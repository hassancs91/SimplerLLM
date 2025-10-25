"""
LLM Feedback Loop - Iterative self-improvement system.

This module provides the LLMFeedbackLoop class for iteratively refining LLM responses
through critique and improvement cycles. Supports multiple architectural patterns:
- Single provider self-critique
- Dual provider (generator + critic)
- Multi-provider rotation
"""

import time
import difflib
from typing import List, Optional, Union, Tuple
from datetime import datetime

from SimplerLLM.language.llm.base import LLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from SimplerLLM.utils.custom_verbose import verbose_print
from .models import (
    Critique,
    IterationResult,
    FeedbackResult,
    FeedbackConfig,
    TemperatureSchedule,
)


class LLMFeedbackLoop:
    """
    Iteratively refines LLM responses through critique and improvement cycles.

    Supports three architectural patterns:
    1. Single Provider Self-Critique: Same LLM generates, critiques, and improves
    2. Dual Provider: One LLM generates, another critiques
    3. Multi-Provider Rotation: Providers rotate through generate/critique roles

    Example:
        ```python
        from SimplerLLM.language import LLM, LLMProvider, LLMFeedbackLoop

        # Single provider self-critique
        llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4")
        feedback = LLMFeedbackLoop(llm=llm, max_iterations=3)
        result = feedback.improve("Explain quantum computing")

        # Dual provider
        generator = LLM.create(LLMProvider.OPENAI, model_name="gpt-4")
        critic = LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4")
        feedback = LLMFeedbackLoop(generator_llm=generator, critic_llm=critic)
        result = feedback.improve("Write a haiku")

        # Multi-provider rotation
        providers = [gpt4, claude, gemini]
        feedback = LLMFeedbackLoop(providers=providers, max_iterations=3)
        result = feedback.improve("Explain AI")
        ```
    """

    # Default prompt templates
    DEFAULT_CRITIQUE_TEMPLATE = """You are an expert critic evaluating an AI-generated response.

ORIGINAL QUESTION:
{original_prompt}

CURRENT ANSWER:
{current_answer}

EVALUATION CRITERIA:
{criteria}

YOUR TASK:
1. Identify strengths in the current answer
2. Identify weaknesses and areas for improvement
3. Provide specific, actionable improvement suggestions
4. Rate the overall quality (1-10)
5. For each criterion, identify specific issues if any

Provide detailed, constructive feedback that will help improve the answer."""

    DEFAULT_IMPROVEMENT_TEMPLATE = """You are improving an AI-generated response based on critique.

ORIGINAL QUESTION:
{original_prompt}

CURRENT ANSWER:
{current_answer}

CRITIQUE AND FEEDBACK:
Strengths: {strengths}
Weaknesses: {weaknesses}
Improvement Suggestions: {suggestions}
Specific Issues: {issues}

YOUR TASK:
Generate an IMPROVED version of the answer that addresses all the weaknesses and implements the improvement suggestions while maintaining the strengths.

{focus_instruction}

Provide a better, more refined answer."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        generator_llm: Optional[LLM] = None,
        critic_llm: Optional[LLM] = None,
        providers: Optional[List[LLM]] = None,
        max_iterations: int = 3,
        convergence_threshold: float = 0.1,
        quality_threshold: Optional[float] = None,
        check_convergence: bool = True,
        default_criteria: Optional[List[str]] = None,
        temperature: float = 0.7,
        temperature_schedule: Optional[Union[str, List[float]]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the LLM Feedback Loop.

        Args:
            llm: Single LLM instance for self-critique pattern
            generator_llm: Generator LLM for dual provider pattern
            critic_llm: Critic LLM for dual provider pattern
            providers: List of LLM instances for multi-provider rotation
            max_iterations: Maximum number of improvement iterations
            convergence_threshold: Stop if improvement < this (e.g., 0.1 = 10%)
            quality_threshold: Stop if quality score >= this (1-10 scale)
            check_convergence: Enable convergence detection
            default_criteria: Default evaluation criteria
            temperature: Base temperature for generation
            temperature_schedule: "fixed", "decreasing", or list of floats
            verbose: Enable detailed logging
        """
        self.llm = llm
        self.generator_llm = generator_llm
        self.critic_llm = critic_llm
        self.providers = providers
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.quality_threshold = quality_threshold
        self.check_convergence = check_convergence
        self.default_criteria = default_criteria or ["accuracy", "clarity", "completeness"]
        self.base_temperature = temperature
        self.temperature_schedule = temperature_schedule
        self.verbose = verbose

        # Detect and validate architecture
        self.architecture = self._detect_architecture()

        if self.verbose:
            verbose_print(
                f"Initialized LLMFeedbackLoop with '{self.architecture}' architecture, "
                f"max_iterations={max_iterations}, convergence_threshold={convergence_threshold}",
                "info"
            )

    def _detect_architecture(self) -> str:
        """Detect which architectural pattern to use based on initialization."""
        if self.llm is not None:
            if self.generator_llm or self.critic_llm or self.providers:
                raise ValueError(
                    "When 'llm' is provided, do not provide generator_llm, critic_llm, or providers"
                )
            return "single"

        elif self.generator_llm is not None and self.critic_llm is not None:
            if self.llm or self.providers:
                raise ValueError(
                    "When 'generator_llm' and 'critic_llm' are provided, do not provide llm or providers"
                )
            return "dual"

        elif self.providers is not None and len(self.providers) >= 2:
            if self.llm or self.generator_llm or self.critic_llm:
                raise ValueError(
                    "When 'providers' is provided, do not provide llm, generator_llm, or critic_llm"
                )
            return "multi_rotation"

        else:
            raise ValueError(
                "Must provide either: (1) llm, (2) generator_llm + critic_llm, or (3) providers list with 2+ items"
            )

    def improve(
        self,
        prompt: str,
        initial_answer: Optional[str] = None,
        focus_on: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        critique_prompt_template: Optional[str] = None,
        improvement_prompt_template: Optional[str] = None,
    ) -> FeedbackResult:
        """
        Iteratively improve an answer through critique and refinement cycles.

        Args:
            prompt: The original question/prompt
            initial_answer: Starting answer (if None, generates one)
            focus_on: Specific criteria to focus improvements on
            system_prompt: Custom system prompt for generation
            critique_prompt_template: Custom template for critique prompts
            improvement_prompt_template: Custom template for improvement prompts

        Returns:
            FeedbackResult with complete history and final answer
        """
        start_time = time.time()

        # Use provided criteria or defaults
        criteria = focus_on if focus_on else self.default_criteria
        criteria_text = ", ".join(criteria)

        if self.verbose:
            verbose_print(f"Starting feedback loop for: '{prompt[:50]}...'", "info")
            verbose_print(f"Architecture: {self.architecture}, Max iterations: {self.max_iterations}", "info")
            verbose_print(f"Evaluation criteria: {criteria_text}", "info")

        # Step 1: Get initial answer
        if initial_answer is None:
            initial_answer = self._generate_initial_answer(prompt, system_prompt)

        # Initialize tracking
        iterations: List[IterationResult] = []
        current_answer = initial_answer
        previous_score = None

        # Step 2: Iterate through improvement cycles
        for iteration_num in range(1, self.max_iterations + 1):
            if self.verbose:
                verbose_print(f"\n=== Iteration {iteration_num}/{self.max_iterations} ===", "info")

            iteration_start = time.time()

            # Get temperature for this iteration
            temperature = self._get_temperature_for_iteration(iteration_num)

            # Run one iteration
            iteration_result = self._run_iteration(
                iteration_num=iteration_num,
                prompt=prompt,
                current_answer=current_answer,
                criteria=criteria,
                criteria_text=criteria_text,
                temperature=temperature,
                system_prompt=system_prompt,
                critique_template=critique_prompt_template,
                improvement_template=improvement_prompt_template,
            )

            # Calculate improvement from previous iteration
            if previous_score is not None:
                improvement_pct = (iteration_result.critique.quality_score - previous_score) / previous_score
                iteration_result.improvement_from_previous = improvement_pct

                if self.verbose:
                    verbose_print(
                        f"Score: {previous_score:.1f} → {iteration_result.critique.quality_score:.1f} "
                        f"(+{improvement_pct:.1%})",
                        "info"
                    )

            iteration_result.execution_time = time.time() - iteration_start
            iterations.append(iteration_result)

            # Check stopping criteria
            should_stop, stop_reason = self._should_stop(
                iteration_num=iteration_num,
                current_score=iteration_result.critique.quality_score,
                previous_score=previous_score,
                current_answer=iteration_result.answer,
                previous_answer=current_answer,
            )

            if should_stop:
                if self.verbose:
                    verbose_print(f"Stopping: {stop_reason}", "info")
                break

            # Update for next iteration
            current_answer = iteration_result.answer
            previous_score = iteration_result.critique.quality_score

        # Step 3: Build final result
        total_time = time.time() - start_time

        result = FeedbackResult(
            final_answer=iterations[-1].answer,
            all_iterations=iterations,
            initial_score=iterations[0].critique.quality_score,
            final_score=iterations[-1].critique.quality_score,
            improvement_trajectory=[it.critique.quality_score for it in iterations],
            total_iterations=len(iterations),
            stopped_reason=stop_reason if should_stop else "max_iterations",
            convergence_detected=(stop_reason == "converged") if should_stop else False,
            total_execution_time=total_time,
            architecture_used=self.architecture,
            timestamp=datetime.now(),
        )

        if self.verbose:
            verbose_print(
                f"\nFeedback loop complete: {result.initial_score:.1f} → {result.final_score:.1f} "
                f"in {result.total_iterations} iterations ({total_time:.2f}s)",
                "info"
            )

        return result

    async def improve_async(
        self,
        prompt: str,
        initial_answer: Optional[str] = None,
        focus_on: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        critique_prompt_template: Optional[str] = None,
        improvement_prompt_template: Optional[str] = None,
    ) -> FeedbackResult:
        """
        Async version of improve().

        Note: Currently runs synchronously. Full async implementation requires
        async support in all LLM wrappers.
        """
        # TODO: Implement true async execution when LLM wrappers support it
        return self.improve(
            prompt,
            initial_answer,
            focus_on,
            system_prompt,
            critique_prompt_template,
            improvement_prompt_template,
        )

    # ==================== Private Methods ====================

    def _generate_initial_answer(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate the initial answer using appropriate LLM."""
        if self.verbose:
            verbose_print("Generating initial answer...", "info")

        # Select which LLM to use for initial generation
        if self.architecture == "single":
            llm = self.llm
        elif self.architecture == "dual":
            llm = self.generator_llm
        else:  # multi_rotation
            llm = self.providers[0]

        system = system_prompt or "You are a helpful AI assistant providing accurate and clear answers."

        answer = llm.generate_response(
            prompt=prompt,
            system_prompt=system,
            temperature=self.base_temperature,
        )

        return answer

    def _run_iteration(
        self,
        iteration_num: int,
        prompt: str,
        current_answer: str,
        criteria: List[str],
        criteria_text: str,
        temperature: float,
        system_prompt: Optional[str],
        critique_template: Optional[str],
        improvement_template: Optional[str],
    ) -> IterationResult:
        """Execute one complete iteration (critique + improvement)."""

        # Step 1: Generate critique
        critique = self._generate_critique(
            prompt=prompt,
            current_answer=current_answer,
            criteria=criteria,
            criteria_text=criteria_text,
            critique_template=critique_template,
        )

        if self.verbose:
            verbose_print(f"Quality score: {critique.quality_score}/10", "info")
            if critique.weaknesses:
                verbose_print(f"Weaknesses: {', '.join(critique.weaknesses[:2])}", "debug")

        # Step 2: Generate improvement (unless this is the last iteration and we're just evaluating)
        if iteration_num < self.max_iterations:
            improved_answer = self._generate_improvement(
                prompt=prompt,
                current_answer=current_answer,
                critique=critique,
                criteria=criteria,
                temperature=temperature,
                system_prompt=system_prompt,
                improvement_template=improvement_template,
            )
        else:
            # Last iteration - just keep current answer
            improved_answer = current_answer

        # Get provider/model info
        provider_info = self._get_provider_info_for_iteration(iteration_num)

        return IterationResult(
            iteration_number=iteration_num,
            answer=improved_answer,
            critique=critique,
            provider_used=provider_info[0],
            model_used=provider_info[1],
            temperature_used=temperature,
            execution_time=0.0,  # Will be set by caller
            improvement_from_previous=None,  # Will be set by caller if not first iteration
            timestamp=datetime.now(),
        )

    def _generate_critique(
        self,
        prompt: str,
        current_answer: str,
        criteria: List[str],
        criteria_text: str,
        critique_template: Optional[str],
    ) -> Critique:
        """Generate structured critique using appropriate LLM."""

        # Select critic LLM based on architecture
        if self.architecture == "single":
            critic = self.llm
        elif self.architecture == "dual":
            critic = self.critic_llm
        else:  # multi_rotation - will be handled by caller
            critic = self._get_critic_for_rotation()

        # Build critique prompt
        template = critique_template or self.DEFAULT_CRITIQUE_TEMPLATE
        critique_prompt = template.format(
            original_prompt=prompt,
            current_answer=current_answer,
            criteria=criteria_text,
        )

        if self.verbose:
            verbose_print("Generating critique...", "debug")

        # Generate structured critique
        try:
            critique = generate_pydantic_json_model(
                model_class=Critique,
                prompt=critique_prompt,
                llm_instance=critic,
                max_retries=3,
                system_prompt="You are an expert critic providing structured, constructive feedback.",
            )

            if isinstance(critique, str):
                raise RuntimeError(f"Critique generation failed: {critique}")

            return critique

        except Exception as e:
            if self.verbose:
                verbose_print(f"Critique generation error: {str(e)}", "error")
            raise RuntimeError(f"Failed to generate critique: {str(e)}")

    def _generate_improvement(
        self,
        prompt: str,
        current_answer: str,
        critique: Critique,
        criteria: List[str],
        temperature: float,
        system_prompt: Optional[str],
        improvement_template: Optional[str],
    ) -> str:
        """Generate improved answer based on critique."""

        # Select generator LLM based on architecture
        if self.architecture == "single":
            generator = self.llm
        elif self.architecture == "dual":
            generator = self.generator_llm
        else:  # multi_rotation
            generator = self._get_generator_for_rotation()

        # Build improvement prompt
        template = improvement_template or self.DEFAULT_IMPROVEMENT_TEMPLATE

        focus_instruction = ""
        if criteria:
            focus_instruction = f"Focus especially on improving: {', '.join(criteria)}"

        improvement_prompt = template.format(
            original_prompt=prompt,
            current_answer=current_answer,
            strengths="\n- ".join(critique.strengths) if critique.strengths else "None identified",
            weaknesses="\n- ".join(critique.weaknesses) if critique.weaknesses else "None identified",
            suggestions="\n- ".join(critique.improvement_suggestions) if critique.improvement_suggestions else "None provided",
            issues="\n".join([f"- {k}: {v}" for k, v in critique.specific_issues.items()]) if critique.specific_issues else "None identified",
            focus_instruction=focus_instruction,
        )

        if self.verbose:
            verbose_print("Generating improved answer...", "debug")

        system = system_prompt or "You are a helpful AI assistant refining answers based on feedback."

        improved = generator.generate_response(
            prompt=improvement_prompt,
            system_prompt=system,
            temperature=temperature,
        )

        return improved

    def _get_provider_info_for_iteration(self, iteration_num: int) -> Tuple[str, str]:
        """Get provider and model name for given iteration."""
        if self.architecture == "single":
            return self.llm.provider.name, self.llm.model_name
        elif self.architecture == "dual":
            return self.generator_llm.provider.name, self.generator_llm.model_name
        else:  # multi_rotation
            idx = (iteration_num - 1) % len(self.providers)
            provider = self.providers[idx]
            return provider.provider.name, provider.model_name

    def _get_critic_for_rotation(self) -> LLM:
        """Get critic LLM for multi-provider rotation (implement rotation logic)."""
        # For now, use next provider in rotation
        # In actual implementation, track current iteration
        return self.providers[0]  # Placeholder

    def _get_generator_for_rotation(self) -> LLM:
        """Get generator LLM for multi-provider rotation."""
        # For now, use first provider
        # In actual implementation, track current iteration
        return self.providers[0]  # Placeholder

    def _get_temperature_for_iteration(self, iteration_num: int) -> float:
        """Get temperature for given iteration based on schedule."""
        if self.temperature_schedule is None or self.temperature_schedule == "fixed":
            return self.base_temperature

        elif self.temperature_schedule == "decreasing":
            # Exponential decay: temp * 0.7^(iteration-1)
            return self.base_temperature * (0.7 ** (iteration_num - 1))

        elif isinstance(self.temperature_schedule, list):
            # Custom list - use index or last value if out of bounds
            idx = iteration_num - 1
            if idx < len(self.temperature_schedule):
                return self.temperature_schedule[idx]
            else:
                return self.temperature_schedule[-1]

        else:
            return self.base_temperature

    def _should_stop(
        self,
        iteration_num: int,
        current_score: float,
        previous_score: Optional[float],
        current_answer: str,
        previous_answer: str,
    ) -> Tuple[bool, str]:
        """Check if stopping criteria are met."""

        # Check 1: Max iterations reached
        if iteration_num >= self.max_iterations:
            return True, "max_iterations"

        # Check 2: Quality threshold met (if set)
        if self.quality_threshold is not None and current_score >= self.quality_threshold:
            return True, "threshold_met"

        # Check 3: Convergence detected (if enabled and not first iteration)
        if self.check_convergence and previous_score is not None:
            converged = self._check_convergence(
                current_answer=current_answer,
                previous_answer=previous_answer,
                current_score=current_score,
                previous_score=previous_score,
            )
            if converged:
                return True, "converged"

        return False, ""

    def _check_convergence(
        self,
        current_answer: str,
        previous_answer: str,
        current_score: float,
        previous_score: float,
    ) -> bool:
        """Detect if the answer has converged (stopped improving significantly)."""

        # Check 1: Score improvement is below threshold
        if previous_score > 0:
            improvement_pct = (current_score - previous_score) / previous_score
            if improvement_pct < self.convergence_threshold:
                if self.verbose:
                    verbose_print(
                        f"Convergence detected: improvement {improvement_pct:.1%} < threshold {self.convergence_threshold:.1%}",
                        "info"
                    )
                return True

        # Check 2: Text similarity is very high (answers are nearly identical)
        similarity = self._calculate_text_similarity(current_answer, previous_answer)
        if similarity > 0.95:  # 95% similar
            if self.verbose:
                verbose_print(
                    f"Convergence detected: text similarity {similarity:.1%} > 95%",
                    "info"
                )
            return True

        return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        return difflib.SequenceMatcher(None, text1, text2).ratio()
