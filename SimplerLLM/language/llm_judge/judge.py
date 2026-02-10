"""
LLM Judge - Multi-provider orchestration and evaluation system.

This module provides the LLMJudge class for orchestrating multiple LLM providers,
evaluating their responses, and generating comparative analyses or synthesized answers.
"""

import time
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from SimplerLLM.language.llm.base import LLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from SimplerLLM.utils.custom_verbose import verbose_print
from .models import (
    JudgeMode,
    ProviderResponse,
    ProviderEvaluation,
    JudgeEvaluation,
    JudgeResult,
    RouterSummary,
    EvaluationReport,
)


class LLMJudge:
    """
    Orchestrates multiple LLM providers to evaluate, compare, or synthesize responses.

    The LLMJudge can operate in three modes:
    - select_best: Pick the best answer from all provider responses
    - synthesize: Combine all answers into an improved response
    - compare: Provide detailed comparative analysis

    Template Customization:
        Templates can be customized at two levels:
        1. Instance level: Pass templates to __init__() for default behavior
        2. Per-call level: Pass templates to generate() to override for specific calls

        Available placeholders:
        - {original_prompt}: The user's question
        - {responses_text}: Formatted provider responses
        - {criteria_text}: Comma-separated evaluation criteria
        - {mode_instructions}: Mode-specific task instructions

    Example:
        ```python
        from SimplerLLM.language import LLM, LLMProvider, LLMJudge

        providers = [
            LLM.create(LLMProvider.OPENAI, model_name="gpt-4"),
            LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4"),
        ]
        judge_llm = LLM.create(LLMProvider.ANTHROPIC, model_name="claude-opus-4")

        judge = LLMJudge(providers=providers, judge_llm=judge_llm)
        result = judge.generate("Explain quantum computing", mode="synthesize")
        print(result.final_answer)

        # With custom template
        custom_instructions = '''YOUR TASK:
        1. Evaluate responses focusing on technical accuracy
        2. Score each 1-10
        3. Provide the best response as final_answer'''

        result = judge.generate(
            "Explain quantum computing",
            mode="select_best",
            mode_instructions=custom_instructions
        )
        ```
    """

    # ==================== Default Templates ====================

    DEFAULT_JUDGE_SYSTEM_PROMPT = "You are an expert AI evaluator. Provide detailed, structured evaluation in JSON format."

    DEFAULT_EVALUATION_TEMPLATE = """You are an expert AI evaluator. Evaluate the following AI responses.

ORIGINAL QUESTION:
{original_prompt}

RESPONSES TO EVALUATE:
{responses_text}

EVALUATION CRITERIA:
{criteria_text}

{mode_instructions}
"""

    DEFAULT_SELECT_BEST_INSTRUCTIONS = """YOUR TASK:
1. Score each response 1-10 for each criterion
2. Calculate overall score (1-10) for each response
3. Rank responses (1 = best, 2 = second best, etc.)
4. Select THE BEST response as your final answer
5. Provide clear reasoning for your evaluation and selection

Your final_answer must be the COMPLETE text of the winning response (the one you ranked #1)."""

    DEFAULT_SYNTHESIZE_INSTRUCTIONS = """YOUR TASK:
1. Score each response 1-10 for each criterion
2. Calculate overall score (1-10) for each response
3. Rank responses (1 = best, 2 = second best, etc.)
4. SYNTHESIZE a NEW, IMPROVED response by combining the best elements from all responses
5. Provide clear reasoning for your evaluation and synthesis process

Your final_answer must be a NEW response that combines the strengths of all responses (not a copy of any existing response)."""

    DEFAULT_COMPARE_INSTRUCTIONS = """YOUR TASK:
1. Score each response 1-10 for each criterion
2. Calculate overall score (1-10) for each response
3. Rank responses (1 = best, 2 = second best, etc.)
4. Provide detailed comparative analysis of strengths and weaknesses
5. Provide clear reasoning

Your final_answer must be a comprehensive comparison summary explaining what each response did well and poorly."""

    def __init__(
        self,
        providers: List[LLM],
        judge_llm: LLM,
        parallel: bool = True,
        default_criteria: Optional[List[str]] = None,
        verbose: bool = False,
        evaluation_template: Optional[str] = None,
        select_best_instructions: Optional[str] = None,
        synthesize_instructions: Optional[str] = None,
        compare_instructions: Optional[str] = None,
        judge_system_prompt: Optional[str] = None,
    ):
        """
        Initialize the LLM Judge.

        Args:
            providers: List of LLM instances to use for generating responses
            judge_llm: LLM instance to use as the judge/evaluator
            parallel: If True, execute providers in parallel; if False, sequential
            default_criteria: Default evaluation criteria (e.g., ["accuracy", "clarity"])
            verbose: Enable verbose logging
            evaluation_template: Custom evaluation template (uses DEFAULT_EVALUATION_TEMPLATE if None)
            select_best_instructions: Custom instructions for select_best mode
            synthesize_instructions: Custom instructions for synthesize mode
            compare_instructions: Custom instructions for compare mode
            judge_system_prompt: Custom system prompt for the judge LLM
        """
        if not providers or len(providers) < 1:
            raise ValueError("At least one provider must be specified")

        if not judge_llm:
            raise ValueError("Judge LLM must be specified")

        self.providers = providers
        self.judge_llm = judge_llm
        self.parallel = parallel
        self.default_criteria = default_criteria or ["accuracy", "clarity", "completeness"]
        self.verbose = verbose

        # Template customization (instance-level defaults)
        self.evaluation_template = evaluation_template
        self.select_best_instructions = select_best_instructions
        self.synthesize_instructions = synthesize_instructions
        self.compare_instructions = compare_instructions
        self.judge_system_prompt = judge_system_prompt

        if self.verbose:
            verbose_print(
                f"Initialized LLMJudge with {len(providers)} providers, "
                f"parallel={parallel}, criteria={self.default_criteria}",
                "info"
            )

    def generate(
        self,
        prompt: str,
        mode: str = "synthesize",
        criteria: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        generate_summary: bool = False,
        evaluation_template: Optional[str] = None,
        mode_instructions: Optional[str] = None,
        judge_system_prompt: Optional[str] = None,
    ) -> JudgeResult:
        """
        Generate responses from all providers and evaluate them using the judge.

        Args:
            prompt: The prompt to send to all providers
            mode: Evaluation mode - "select_best", "synthesize", or "compare"
            criteria: Custom evaluation criteria (uses default_criteria if not provided)
            system_prompt: Optional system prompt for providers
            generate_summary: If True, generate router training summary
            evaluation_template: Custom evaluation template (overrides instance default)
            mode_instructions: Custom mode instructions (overrides mode-specific default)
            judge_system_prompt: Custom system prompt for judge (overrides instance default)

        Returns:
            JudgeResult containing final answer, all responses, evaluations, and metadata
        """
        start_time = time.time()

        # Validate mode
        try:
            judge_mode = JudgeMode(mode)
        except ValueError:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {[m.value for m in JudgeMode]}")

        # Use provided criteria or default
        eval_criteria = criteria or self.default_criteria

        if self.verbose:
            verbose_print(f"Starting evaluation in '{mode}' mode with criteria: {eval_criteria}", "info")

        # Step 1: Execute all providers
        provider_responses = self._execute_providers(prompt, system_prompt)

        if self.verbose:
            verbose_print(f"Received {len(provider_responses)} provider responses", "info")

        # Step 2: Build judge prompt based on mode
        judge_prompt = self._build_judge_prompt(
            original_prompt=prompt,
            provider_responses=provider_responses,
            mode=judge_mode,
            criteria=eval_criteria,
            evaluation_template=evaluation_template,
            mode_instructions=mode_instructions,
        )

        # Step 3: Judge evaluates responses
        judge_evaluation = self._evaluate_responses(judge_prompt, judge_system_prompt)

        # Step 4: Calculate confidence scores (normalize overall_scores to 0-1)
        confidence_scores = self._calculate_confidence_scores(judge_evaluation.evaluations)

        # Step 5: Build final result
        total_time = time.time() - start_time

        result = JudgeResult(
            final_answer=judge_evaluation.final_answer,
            all_responses=provider_responses,
            evaluations=judge_evaluation.evaluations,
            judge_reasoning=judge_evaluation.overall_reasoning,
            confidence_scores=confidence_scores,
            mode=judge_mode,
            criteria_used=eval_criteria,
            total_execution_time=total_time,
            timestamp=datetime.now(),
        )

        if self.verbose:
            verbose_print(f"Evaluation complete in {total_time:.2f}s", "info")
            verbose_print(f"Winner: {judge_evaluation.evaluations[0].provider_name}", "info")

        # Step 6: Generate router summary if requested
        if generate_summary:
            self._router_summary = self._export_router_summary(result, prompt)
            if self.verbose:
                verbose_print(f"Router summary generated: {self._router_summary.recommendation}", "info")

        return result

    async def generate_async(
        self,
        prompt: str,
        mode: str = "synthesize",
        criteria: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        generate_summary: bool = False,
        evaluation_template: Optional[str] = None,
        mode_instructions: Optional[str] = None,
        judge_system_prompt: Optional[str] = None,
    ) -> JudgeResult:
        """
        Async version of generate().

        Note: Currently runs synchronously. Full async implementation requires
        async support in all LLM wrappers.
        """
        # TODO: Implement true async execution when LLM wrappers support it
        return self.generate(
            prompt, mode, criteria, system_prompt, generate_summary,
            evaluation_template, mode_instructions, judge_system_prompt
        )

    def evaluate_batch(
        self,
        prompts: List[str],
        mode: str = "compare",
        criteria: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[JudgeResult]:
        """
        Evaluate multiple prompts in batch for benchmarking.

        Args:
            prompts: List of prompts to evaluate
            mode: Evaluation mode (defaults to "compare" for benchmarking)
            criteria: Evaluation criteria
            system_prompt: Optional system prompt

        Returns:
            List of JudgeResult objects, one per prompt
        """
        results = []
        total_prompts = len(prompts)

        if self.verbose:
            verbose_print(f"Starting batch evaluation of {total_prompts} prompts", "info")

        for idx, prompt in enumerate(prompts, 1):
            if self.verbose:
                verbose_print(f"Evaluating prompt {idx}/{total_prompts}", "info")

            result = self.generate(
                prompt=prompt,
                mode=mode,
                criteria=criteria,
                system_prompt=system_prompt,
                generate_summary=False,
            )
            results.append(result)

        return results

    def generate_evaluation_report(
        self,
        results: List[JudgeResult],
        export_format: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Generate statistical summary from batch evaluation results.

        Args:
            results: List of JudgeResult objects from evaluate_batch()
            export_format: Optional export format - "json" or "csv"

        Returns:
            EvaluationReport with statistical analysis
        """
        if not results:
            raise ValueError("No results provided for report generation")

        # Initialize counters
        provider_win_counts: Dict[str, int] = {}
        provider_score_sums: Dict[str, float] = {}
        provider_score_counts: Dict[str, int] = {}
        criteria_winners: Dict[str, Dict[str, int]] = {}

        # Collect all provider names
        all_providers = set()
        for result in results:
            for eval in result.evaluations:
                all_providers.add(eval.provider_name)

        # Initialize dictionaries
        for provider in all_providers:
            provider_win_counts[provider] = 0
            provider_score_sums[provider] = 0.0
            provider_score_counts[provider] = 0

        # Process each result
        for result in results:
            # Count wins (rank 1)
            for eval in result.evaluations:
                if eval.rank == 1:
                    provider_win_counts[eval.provider_name] += 1

                # Sum scores
                provider_score_sums[eval.provider_name] += eval.overall_score
                provider_score_counts[eval.provider_name] += 1

                # Track criterion winners
                for criterion, score in eval.criterion_scores.items():
                    if criterion not in criteria_winners:
                        criteria_winners[criterion] = {}
                    if eval.provider_name not in criteria_winners[criterion]:
                        criteria_winners[criterion][eval.provider_name] = 0
                    criteria_winners[criterion][eval.provider_name] += 1

        # Calculate averages
        average_scores = {
            provider: provider_score_sums[provider] / provider_score_counts[provider]
            for provider in all_providers
        }

        # Find best overall provider
        best_provider_overall = max(average_scores.items(), key=lambda x: x[1])[0]

        # Find best provider per criterion
        best_provider_by_criteria = {
            criterion: max(winners.items(), key=lambda x: x[1])[0]
            for criterion, winners in criteria_winners.items()
        }

        # Build evaluation data for export
        evaluation_data = []
        for idx, result in enumerate(results):
            for eval in result.evaluations:
                evaluation_data.append({
                    "query_index": idx,
                    "provider": eval.provider_name,
                    "overall_score": eval.overall_score,
                    "rank": eval.rank,
                    "criterion_scores": eval.criterion_scores,
                })

        report = EvaluationReport(
            total_queries=len(results),
            provider_win_counts=provider_win_counts,
            average_scores=average_scores,
            best_provider_overall=best_provider_overall,
            best_provider_by_criteria=best_provider_by_criteria,
            evaluation_data=evaluation_data,
            generated_at=datetime.now(),
        )

        # Export if format specified
        if export_format:
            self._export_report(report, export_format)

        return report

    # ==================== Private Methods ====================

    def _execute_providers(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[ProviderResponse]:
        """Execute all providers and collect their responses."""
        responses = []

        if self.parallel:
            # Parallel execution (using threads since LLM wrappers aren't async yet)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
                futures = {
                    executor.submit(self._execute_single_provider, provider, prompt, system_prompt): provider
                    for provider in self.providers
                }

                for future in concurrent.futures.as_completed(futures):
                    response = future.result()
                    responses.append(response)
        else:
            # Sequential execution
            for provider in self.providers:
                response = self._execute_single_provider(provider, prompt, system_prompt)
                responses.append(response)

        return responses

    def _execute_single_provider(
        self,
        provider: LLM,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> ProviderResponse:
        """Execute a single provider and return its response."""
        start_time = time.time()
        error = None
        response_text = ""

        try:
            if self.verbose:
                verbose_print(f"Executing provider: {provider.provider.name} ({provider.model_name})", "debug")

            response_text = provider.generate_response(
                prompt=prompt,
                system_prompt=system_prompt or "You are a helpful AI assistant.",
            )
        except Exception as e:
            error = str(e)
            if self.verbose:
                verbose_print(f"Error from {provider.provider.name}: {error}", "error")

        execution_time = time.time() - start_time

        return ProviderResponse(
            provider_name=provider.provider.name,
            model_name=provider.model_name,
            response_text=response_text,
            execution_time=execution_time,
            timestamp=datetime.now(),
            error=error,
        )

    def _build_judge_prompt(
        self,
        original_prompt: str,
        provider_responses: List[ProviderResponse],
        mode: JudgeMode,
        criteria: List[str],
        evaluation_template: Optional[str] = None,
        mode_instructions: Optional[str] = None,
    ) -> str:
        """
        Build the judge prompt based on mode and responses.

        Template resolution (fallback chain):
        1. Per-call parameter (evaluation_template, mode_instructions)
        2. Instance attribute (self.evaluation_template, self.*_instructions)
        3. Class constant default (DEFAULT_*)
        """
        # Filter out failed responses
        valid_responses = [r for r in provider_responses if not r.error]

        if not valid_responses:
            raise RuntimeError("All providers failed to generate responses")

        # Build responses text
        responses_text = self._format_responses(valid_responses)

        # Build criteria text
        criteria_text = ", ".join(criteria)

        # Resolve mode instructions (per-call > instance > default)
        if mode_instructions:
            instructions = mode_instructions
        elif mode == JudgeMode.SELECT_BEST:
            instructions = self.select_best_instructions or self.DEFAULT_SELECT_BEST_INSTRUCTIONS
        elif mode == JudgeMode.SYNTHESIZE:
            instructions = self.synthesize_instructions or self.DEFAULT_SYNTHESIZE_INSTRUCTIONS
        else:  # JudgeMode.COMPARE
            instructions = self.compare_instructions or self.DEFAULT_COMPARE_INSTRUCTIONS

        # Resolve evaluation template (per-call > instance > default)
        template = evaluation_template or self.evaluation_template or self.DEFAULT_EVALUATION_TEMPLATE

        # Format the template with all placeholders
        return template.format(
            original_prompt=original_prompt,
            responses_text=responses_text,
            criteria_text=criteria_text,
            mode_instructions=instructions,
        )

    def _format_responses(self, responses: List[ProviderResponse]) -> str:
        """Format provider responses for template insertion."""
        parts = []
        for idx, response in enumerate(responses, 1):
            parts.append(f"--- Response {idx} ({response.provider_name} - {response.model_name}) ---")
            parts.append(response.response_text)
            parts.append("")
        return "\n".join(parts)

    def _evaluate_responses(
        self,
        judge_prompt: str,
        judge_system_prompt: Optional[str] = None,
    ) -> JudgeEvaluation:
        """Use the judge LLM to evaluate responses with structured output."""
        if self.verbose:
            verbose_print("Judge is evaluating responses...", "info")

        # Resolve system prompt (per-call > instance > default)
        system_prompt = (
            judge_system_prompt
            or self.judge_system_prompt
            or self.DEFAULT_JUDGE_SYSTEM_PROMPT
        )

        try:
            judge_result = generate_pydantic_json_model(
                model_class=JudgeEvaluation,
                prompt=judge_prompt,
                llm_instance=self.judge_llm,
                max_retries=3,
                system_prompt=system_prompt,
            )

            if isinstance(judge_result, str):
                # Error occurred
                raise RuntimeError(f"Judge evaluation failed: {judge_result}")

            return judge_result

        except Exception as e:
            if self.verbose:
                verbose_print(f"Judge evaluation error: {str(e)}", "error")
            raise RuntimeError(f"Failed to evaluate responses: {str(e)}")

    def _calculate_confidence_scores(self, evaluations: List[ProviderEvaluation]) -> Dict[str, float]:
        """Convert overall scores (1-10) to confidence scores (0-1)."""
        confidence_scores = {}

        for eval in evaluations:
            # Normalize score from 1-10 scale to 0-1 scale
            confidence_scores[eval.provider_name] = eval.overall_score / 10.0

        return confidence_scores

    def _export_router_summary(self, result: JudgeResult, original_prompt: str) -> RouterSummary:
        """Generate router training summary from evaluation results."""
        # Get winner (rank 1)
        winner_eval = next(e for e in result.evaluations if e.rank == 1)

        # Build provider scores dict
        provider_scores = {
            eval.provider_name: eval.overall_score
            for eval in result.evaluations
        }

        # Find winner per criterion
        criteria_winners = {}
        for criterion in result.criteria_used:
            best_provider = None
            best_score = -1

            for eval in result.evaluations:
                if criterion in eval.criterion_scores:
                    if eval.criterion_scores[criterion] > best_score:
                        best_score = eval.criterion_scores[criterion]
                        best_provider = eval.provider_name

            if best_provider:
                criteria_winners[criterion] = best_provider

        # Infer query type (simple heuristic - could be enhanced with LLM classification)
        query_type = "general"
        prompt_lower = original_prompt.lower()
        if any(word in prompt_lower for word in ["code", "python", "javascript", "program", "debug"]):
            query_type = "coding"
        elif any(word in prompt_lower for word in ["explain", "what is", "how does", "technical"]):
            query_type = "technical_explanation"
        elif any(word in prompt_lower for word in ["write", "story", "creative", "poem"]):
            query_type = "creative_writing"

        recommendation = f"Use {winner_eval.provider_name} for {query_type} tasks"

        return RouterSummary(
            query_type=query_type,
            winning_provider=winner_eval.provider_name,
            provider_scores=provider_scores,
            recommendation=recommendation,
            criteria_winners=criteria_winners,
        )

    def _export_report(self, report: EvaluationReport, format: str):
        """Export evaluation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            filename = f"llm_judge_report_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(report.model_dump(), f, indent=2, default=str)

            if self.verbose:
                verbose_print(f"Report exported to {filename}", "info")

        elif format == "csv":
            import csv
            filename = f"llm_judge_report_{timestamp}.csv"

            with open(filename, 'w', newline='') as f:
                if report.evaluation_data:
                    fieldnames = list(report.evaluation_data[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for row in report.evaluation_data:
                        # Flatten nested dicts for CSV
                        flat_row = {k: str(v) for k, v in row.items()}
                        writer.writerow(flat_row)

            if self.verbose:
                verbose_print(f"Report exported to {filename}", "info")

        else:
            raise ValueError(f"Unsupported export format: {format}")
