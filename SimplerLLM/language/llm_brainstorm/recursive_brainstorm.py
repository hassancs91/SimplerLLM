"""
Recursive Brainstorm - Generate and expand ideas recursively using LLMs.

This module provides a flexible brainstorming system that can generate ideas
in three different modes: tree-based expansion, linear refinement, or hybrid.
"""

import time
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from SimplerLLM.language.llm import LLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from SimplerLLM.utils.custom_verbose import verbose_print
from .models import (
    BrainstormIdea,
    BrainstormLevel,
    BrainstormIteration,
    BrainstormResult,
    BrainstormMode,
    IdeaGeneration,
    IdeaEvaluation,
)


class RecursiveBrainstorm:
    """
    Recursive brainstorming system with multiple generation modes.

    This class enables recursive idea generation and refinement using LLMs,
    supporting three different expansion strategies:
    - Tree mode: Exponential expansion (expand all ideas)
    - Linear mode: Focus on best ideas (single path refinement)
    - Hybrid mode: Selective expansion (top-N ideas)

    Template Customization:
        Templates can be customized at two levels:
        1. Instance level: Pass templates to __init__() for default behavior
        2. Per-call level: Pass templates to brainstorm() to override for specific calls

        Available placeholders for generation_template:
        - {ideas_count}: Number of ideas to generate
        - {prompt}: The brainstorming prompt
        - {context}: Additional context (if provided)

        Available placeholders for evaluation_template:
        - {original_prompt}: The original brainstorming prompt
        - {idea}: The idea text to evaluate
        - {criteria_text}: Formatted evaluation criteria

        Available placeholders for expansion_template:
        - {original_prompt}: The original brainstorming prompt
        - {ideas_count}: Number of ideas to generate
        - {parent_idea}: Parent idea text
        - {parent_reasoning}: Parent idea reasoning
        - {parent_score}: Parent idea quality score
        - {context}: Additional context (if provided)

    Example:
        ```python
        from SimplerLLM.language import LLM, LLMProvider
        from SimplerLLM.language import RecursiveBrainstorm, BrainstormMode

        llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")
        brainstorm = RecursiveBrainstorm(
            llm=llm,
            max_depth=2,
            ideas_per_level=5,
            mode=BrainstormMode.TREE
        )

        result = brainstorm.brainstorm("Ways to reduce carbon emissions")
        print(f"Generated {result.total_ideas} ideas")
        print(f"Best idea: {result.overall_best_idea.text}")

        # With custom template
        custom_template = '''Generate {ideas_count} innovative ideas for:
        {prompt}

        Focus on practical, actionable solutions.'''

        result = brainstorm.brainstorm(
            "Improve team productivity",
            generation_template=custom_template
        )
        ```
    """

    # ==================== Default Templates ====================

    DEFAULT_SYSTEM_PROMPT = "You are a creative brainstorming assistant. Generate diverse, innovative, and actionable ideas."

    DEFAULT_GENERATION_TEMPLATE = """Generate exactly {ideas_count} creative and diverse ideas based on the following prompt:

Prompt: {prompt}{context}

Requirements:
- Generate exactly {ideas_count} distinct ideas
- Make each idea specific and actionable
- Ensure diversity in approaches
- Provide brief reasoning for each idea

Output your ideas in structured JSON format."""

    DEFAULT_EVALUATION_TEMPLATE = """Evaluate the following idea based on the criteria below.

Original Prompt: {original_prompt}

Idea to Evaluate: {idea}

Evaluation Criteria:
{criteria_text}

Provide:
1. An overall quality score from 1-10
2. Strengths of the idea
3. Weaknesses or challenges
4. Individual scores for each criterion (1-10)
5. Whether this idea should be expanded further
6. Overall reasoning for your evaluation

Be objective and constructive in your evaluation."""

    DEFAULT_EXPANSION_TEMPLATE = """Based on the following idea, generate {ideas_count} refined variations, expansions, or sub-ideas:

Original Prompt: {original_prompt}{context}

Parent Idea: {parent_idea}
Reasoning: {parent_reasoning}
Quality Score: {parent_score}/10

Generate {ideas_count} ideas that:
- Build upon or refine the parent idea
- Explore different aspects or approaches
- Add more specificity or detail
- Address potential weaknesses while leveraging strengths

Make each idea distinct and valuable."""

    def __init__(
        self,
        llm: LLM,
        max_depth: int = 3,
        ideas_per_level: int = 5,
        mode: BrainstormMode = BrainstormMode.TREE,
        top_n: int = 3,
        evaluation_criteria: Optional[List[str]] = None,
        min_quality_threshold: float = 5.0,
        verbose: bool = False,
        generation_template: Optional[str] = None,
        evaluation_template: Optional[str] = None,
        expansion_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the RecursiveBrainstorm instance.

        Args:
            llm: LLM instance for generation
            max_depth: Maximum recursion depth (default: 3)
            ideas_per_level: Ideas to generate per expansion (default: 5)
            mode: Default mode - BrainstormMode.TREE, LINEAR, or HYBRID (default: TREE)
            top_n: For hybrid mode, number of top ideas to expand (default: 3)
            evaluation_criteria: List of criteria like ["feasibility", "impact", "novelty"]
            min_quality_threshold: Minimum score to expand (1-10, default: 5.0)
            verbose: Print progress information (default: False)
            generation_template: Custom template for idea generation
            evaluation_template: Custom template for idea evaluation
            expansion_template: Custom template for idea expansion
            system_prompt: Custom system prompt for LLM

        Raises:
            ValueError: If mode is not a valid BrainstormMode
            ValueError: If max_depth < 1 or ideas_per_level < 1
        """
        if not isinstance(mode, BrainstormMode):
            raise ValueError(f"Mode must be a BrainstormMode enum value, got {type(mode)}")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if ideas_per_level < 1:
            raise ValueError("ideas_per_level must be at least 1")
        if not (1.0 <= min_quality_threshold <= 10.0):
            raise ValueError("min_quality_threshold must be between 1.0 and 10.0")

        self.llm = llm
        self.max_depth = max_depth
        self.ideas_per_level = ideas_per_level
        self.mode = mode
        self.top_n = min(top_n, ideas_per_level)  # Can't expand more than we generate
        self.evaluation_criteria = evaluation_criteria or ["quality", "feasibility", "impact"]
        self.min_quality_threshold = min_quality_threshold
        self.verbose = verbose

        # Template customization (instance-level defaults)
        self.generation_template = generation_template
        self.evaluation_template = evaluation_template
        self.expansion_template = expansion_template
        self.system_prompt = system_prompt

        # Internal state
        self._iteration_counter = 0
        self._all_ideas: List[BrainstormIdea] = []
        self._all_iterations: List[BrainstormIteration] = []
        self._tree_structure: Dict[str, List[str]] = {}

        if self.verbose:
            verbose_print(
                f"Initialized RecursiveBrainstorm: mode={mode.value}, "
                f"max_depth={max_depth}, ideas_per_level={ideas_per_level}",
                "info"
            )

    def brainstorm(
        self,
        prompt: str,
        mode: Optional[BrainstormMode] = None,
        context: Optional[str] = None,
        save_csv: bool = False,
        csv_path: str = "brainstorm_results.csv",
        csv_expand_criteria: bool = True,
        generation_template: Optional[str] = None,
        evaluation_template: Optional[str] = None,
        expansion_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> BrainstormResult:
        """
        Execute recursive brainstorming session.

        Args:
            prompt: The initial brainstorming prompt
            mode: Override default mode (BrainstormMode.TREE, LINEAR, or HYBRID)
            context: Additional context to guide idea generation
            save_csv: Whether to automatically save results to CSV file
            csv_path: Path to CSV file (used if save_csv=True)
            csv_expand_criteria: Whether to expand criteria scores into separate columns
            generation_template: Custom template for idea generation (overrides instance default)
            evaluation_template: Custom template for idea evaluation (overrides instance default)
            expansion_template: Custom template for idea expansion (overrides instance default)
            system_prompt: Custom system prompt (overrides instance default)

        Returns:
            BrainstormResult containing all generated ideas and metadata

        Example:
            >>> from SimplerLLM.language import LLM, LLMProvider, RecursiveBrainstorm, BrainstormMode
            >>> llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")
            >>> brainstorm = RecursiveBrainstorm(llm, max_depth=2)
            >>> result = brainstorm.brainstorm("Ways to reduce carbon emissions")
            >>> print(f"Generated {result.total_ideas} ideas")
            >>> print(f"Best idea: {result.overall_best_idea.text}")

            >>> # Auto-save to CSV
            >>> result = brainstorm.brainstorm(
            ...     "Generate domain names",
            ...     save_csv=True,
            ...     csv_path="domains.csv"
            ... )
        """
        start_time = time.time()
        active_mode = mode or self.mode

        # Validate mode
        if not isinstance(active_mode, BrainstormMode):
            raise ValueError(f"Mode must be a BrainstormMode enum value, got {type(active_mode)}")

        # Store per-call template overrides for use in generation methods
        self._active_generation_template = generation_template
        self._active_evaluation_template = evaluation_template
        self._active_expansion_template = expansion_template
        self._active_system_prompt = system_prompt

        # Reset internal state
        self._iteration_counter = 0
        self._all_ideas = []
        self._all_iterations = []
        self._tree_structure = {}

        if self.verbose:
            verbose_print(
                f"Starting Recursive Brainstorm - Mode: {active_mode.value.upper()}",
                "info"
            )
            verbose_print(
                f"Max Depth: {self.max_depth} | Ideas per Level: {self.ideas_per_level}",
                "info"
            )

        # Execute brainstorming based on mode
        if active_mode == BrainstormMode.TREE:
            self._brainstorm_tree(prompt, context)
        elif active_mode == BrainstormMode.LINEAR:
            self._brainstorm_linear(prompt, context)
        elif active_mode == BrainstormMode.HYBRID:
            self._brainstorm_hybrid(prompt, context)

        execution_time = time.time() - start_time

        # Build result
        result = self._build_result(
            initial_prompt=prompt,
            mode=active_mode,
            execution_time=execution_time,
        )

        if self.verbose:
            verbose_print(f"Brainstorm Complete!", "info")
            verbose_print(
                f"Total Ideas: {result.total_ideas} | Total Iterations: {result.total_iterations}",
                "info"
            )
            verbose_print(
                f"Max Depth: {result.max_depth_reached} | Time: {result.execution_time:.2f}s",
                "info"
            )
            if result.overall_best_idea:
                verbose_print(
                    f"Best Idea (Score: {result.overall_best_idea.quality_score:.1f}): "
                    f"{result.overall_best_idea.text[:80]}...",
                    "info"
                )

        # Auto-save to CSV if requested
        if save_csv:
            result.to_csv(csv_path, expand_criteria=csv_expand_criteria)
            if self.verbose:
                verbose_print(f"Results saved to CSV: {csv_path}", "info")

        # Clear per-call overrides
        self._active_generation_template = None
        self._active_evaluation_template = None
        self._active_expansion_template = None
        self._active_system_prompt = None

        return result

    async def brainstorm_async(
        self,
        prompt: str,
        mode: Optional[BrainstormMode] = None,
        context: Optional[str] = None,
        save_csv: bool = False,
        csv_path: str = "brainstorm_results.csv",
        csv_expand_criteria: bool = True,
        generation_template: Optional[str] = None,
        evaluation_template: Optional[str] = None,
        expansion_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> BrainstormResult:
        """
        Execute recursive brainstorming session asynchronously.

        Note: Currently runs synchronously via executor. Full async implementation
        requires async support in generate_pydantic_json_model.

        Args:
            prompt: The initial brainstorming prompt
            mode: Override default mode (BrainstormMode.TREE, LINEAR, or HYBRID)
            context: Additional context to guide idea generation
            save_csv: Whether to automatically save results to CSV file
            csv_path: Path to CSV file (used if save_csv=True)
            csv_expand_criteria: Whether to expand criteria scores into separate columns
            generation_template: Custom template for idea generation (overrides instance default)
            evaluation_template: Custom template for idea evaluation (overrides instance default)
            expansion_template: Custom template for idea expansion (overrides instance default)
            system_prompt: Custom system prompt (overrides instance default)

        Returns:
            BrainstormResult containing all generated ideas and metadata
        """
        start_time = time.time()
        active_mode = mode or self.mode

        # Validate mode
        if not isinstance(active_mode, BrainstormMode):
            raise ValueError(f"Mode must be a BrainstormMode enum value, got {type(active_mode)}")

        # Store per-call template overrides for use in generation methods
        self._active_generation_template = generation_template
        self._active_evaluation_template = evaluation_template
        self._active_expansion_template = expansion_template
        self._active_system_prompt = system_prompt

        # Reset internal state
        self._iteration_counter = 0
        self._all_ideas = []
        self._all_iterations = []
        self._tree_structure = {}

        if self.verbose:
            verbose_print(
                f"Starting Recursive Brainstorm (ASYNC) - Mode: {active_mode.value.upper()}",
                "info"
            )
            verbose_print(
                f"Max Depth: {self.max_depth} | Ideas per Level: {self.ideas_per_level}",
                "info"
            )

        # Execute brainstorming based on mode
        if active_mode == BrainstormMode.TREE:
            await self._brainstorm_tree_async(prompt, context)
        elif active_mode == BrainstormMode.LINEAR:
            await self._brainstorm_linear_async(prompt, context)
        elif active_mode == BrainstormMode.HYBRID:
            await self._brainstorm_hybrid_async(prompt, context)

        execution_time = time.time() - start_time

        # Build result
        result = self._build_result(
            initial_prompt=prompt,
            mode=active_mode,
            execution_time=execution_time,
        )

        if self.verbose:
            verbose_print(f"Async Brainstorm Complete!", "info")
            verbose_print(
                f"Total Ideas: {result.total_ideas} | Total Iterations: {result.total_iterations}",
                "info"
            )
            if result.overall_best_idea:
                verbose_print(
                    f"Best Idea (Score: {result.overall_best_idea.quality_score:.1f}): "
                    f"{result.overall_best_idea.text[:80]}...",
                    "info"
                )

        # Auto-save to CSV if requested
        if save_csv:
            result.to_csv(csv_path, expand_criteria=csv_expand_criteria)
            if self.verbose:
                verbose_print(f"Results saved to CSV: {csv_path}", "info")

        # Clear per-call overrides
        self._active_generation_template = None
        self._active_evaluation_template = None
        self._active_expansion_template = None
        self._active_system_prompt = None

        return result

    # -------------------------------------------------------------------------
    # TREE MODE: Exponential expansion of all ideas
    # -------------------------------------------------------------------------

    def _brainstorm_tree(self, prompt: str, context: Optional[str] = None):
        """Tree mode: Expand all ideas at each level (exponential growth)."""
        # Generate initial ideas at depth 0
        initial_ideas = self._generate_ideas(
            prompt=prompt,
            context=context,
            depth=0,
            parent_id=None,
        )

        # Queue of ideas to expand: (idea, current_depth)
        to_expand: List[Tuple[BrainstormIdea, int]] = [(idea, 0) for idea in initial_ideas]

        while to_expand:
            current_idea, current_depth = to_expand.pop(0)
            next_depth = current_depth + 1

            # Stop if we've reached max depth
            if next_depth >= self.max_depth:
                continue

            # Stop if idea quality is below threshold
            if current_idea.quality_score < self.min_quality_threshold:
                if self.verbose:
                    verbose_print(
                        f"Skipping expansion of idea (score {current_idea.quality_score:.1f} < {self.min_quality_threshold})",
                        "debug"
                    )
                continue

            # Generate child ideas
            expansion_prompt = self._create_expansion_prompt(current_idea, prompt, context)
            child_ideas = self._generate_ideas(
                prompt=expansion_prompt,
                context=context,
                depth=next_depth,
                parent_id=current_idea.id,
            )

            # Add children to expansion queue
            to_expand.extend([(child, next_depth) for child in child_ideas])

    async def _brainstorm_tree_async(self, prompt: str, context: Optional[str] = None):
        """Async tree mode: Expand all ideas at each level."""
        # Generate initial ideas
        initial_ideas = await self._generate_ideas_async(
            prompt=prompt,
            context=context,
            depth=0,
            parent_id=None,
        )

        # Queue of ideas to expand
        to_expand: List[Tuple[BrainstormIdea, int]] = [(idea, 0) for idea in initial_ideas]

        while to_expand:
            current_idea, current_depth = to_expand.pop(0)
            next_depth = current_depth + 1

            if next_depth >= self.max_depth:
                continue

            if current_idea.quality_score < self.min_quality_threshold:
                continue

            expansion_prompt = self._create_expansion_prompt(current_idea, prompt, context)
            child_ideas = await self._generate_ideas_async(
                prompt=expansion_prompt,
                context=context,
                depth=next_depth,
                parent_id=current_idea.id,
            )

            to_expand.extend([(child, next_depth) for child in child_ideas])

    # -------------------------------------------------------------------------
    # LINEAR MODE: Refine only the best idea at each level
    # -------------------------------------------------------------------------

    def _brainstorm_linear(self, prompt: str, context: Optional[str] = None):
        """Linear mode: Generate ideas, pick best, refine it, repeat."""
        current_prompt = prompt
        current_depth = 0

        while current_depth < self.max_depth:
            # Generate ideas at current level
            ideas = self._generate_ideas(
                prompt=current_prompt,
                context=context,
                depth=current_depth,
                parent_id=None if current_depth == 0 else best_idea.id,
            )

            if not ideas:
                break

            # Pick the best idea
            best_idea = max(ideas, key=lambda x: x.quality_score)

            if self.verbose:
                verbose_print(
                    f"Best idea at depth {current_depth} (score: {best_idea.quality_score:.1f}): {best_idea.text[:60]}...",
                    "debug"
                )

            # Stop if quality is below threshold
            if best_idea.quality_score < self.min_quality_threshold:
                if self.verbose:
                    verbose_print(
                        f"Stopping: best idea score {best_idea.quality_score:.1f} < threshold {self.min_quality_threshold}",
                        "debug"
                    )
                break

            # Create refinement prompt for next iteration
            current_prompt = self._create_expansion_prompt(best_idea, prompt, context)
            current_depth += 1

    async def _brainstorm_linear_async(self, prompt: str, context: Optional[str] = None):
        """Async linear mode: Generate ideas, pick best, refine it, repeat."""
        current_prompt = prompt
        current_depth = 0

        while current_depth < self.max_depth:
            ideas = await self._generate_ideas_async(
                prompt=current_prompt,
                context=context,
                depth=current_depth,
                parent_id=None if current_depth == 0 else best_idea.id,
            )

            if not ideas:
                break

            best_idea = max(ideas, key=lambda x: x.quality_score)

            if self.verbose:
                verbose_print(
                    f"Best idea at depth {current_depth} (score: {best_idea.quality_score:.1f}): {best_idea.text[:60]}...",
                    "debug"
                )

            if best_idea.quality_score < self.min_quality_threshold:
                break

            current_prompt = self._create_expansion_prompt(best_idea, prompt, context)
            current_depth += 1

    # -------------------------------------------------------------------------
    # HYBRID MODE: Expand top N ideas at each level
    # -------------------------------------------------------------------------

    def _brainstorm_hybrid(self, prompt: str, context: Optional[str] = None):
        """Hybrid mode: Generate ideas, expand top N."""
        # Generate initial ideas
        initial_ideas = self._generate_ideas(
            prompt=prompt,
            context=context,
            depth=0,
            parent_id=None,
        )

        # Queue: (idea, depth)
        to_expand: List[Tuple[BrainstormIdea, int]] = [(idea, 0) for idea in initial_ideas]

        while to_expand:
            # Get all ideas at current depth
            current_depth = to_expand[0][1]
            ideas_at_depth = [idea for idea, d in to_expand if d == current_depth]
            to_expand = [item for item in to_expand if item[1] != current_depth]

            next_depth = current_depth + 1
            if next_depth >= self.max_depth:
                continue

            # Sort by quality and take top N
            ideas_at_depth.sort(key=lambda x: x.quality_score, reverse=True)
            top_ideas = ideas_at_depth[:self.top_n]

            if self.verbose:
                verbose_print(
                    f"Depth {current_depth}: Expanding top {len(top_ideas)} ideas (scores: {[f'{i.quality_score:.1f}' for i in top_ideas]})",
                    "debug"
                )

            # Expand each top idea
            for idea in top_ideas:
                if idea.quality_score < self.min_quality_threshold:
                    continue

                expansion_prompt = self._create_expansion_prompt(idea, prompt, context)
                child_ideas = self._generate_ideas(
                    prompt=expansion_prompt,
                    context=context,
                    depth=next_depth,
                    parent_id=idea.id,
                )

                to_expand.extend([(child, next_depth) for child in child_ideas])

    async def _brainstorm_hybrid_async(self, prompt: str, context: Optional[str] = None):
        """Async hybrid mode: Generate ideas, expand top N."""
        initial_ideas = await self._generate_ideas_async(
            prompt=prompt,
            context=context,
            depth=0,
            parent_id=None,
        )

        to_expand: List[Tuple[BrainstormIdea, int]] = [(idea, 0) for idea in initial_ideas]

        while to_expand:
            current_depth = to_expand[0][1]
            ideas_at_depth = [idea for idea, d in to_expand if d == current_depth]
            to_expand = [item for item in to_expand if item[1] != current_depth]

            next_depth = current_depth + 1
            if next_depth >= self.max_depth:
                continue

            ideas_at_depth.sort(key=lambda x: x.quality_score, reverse=True)
            top_ideas = ideas_at_depth[:self.top_n]

            for idea in top_ideas:
                if idea.quality_score < self.min_quality_threshold:
                    continue

                expansion_prompt = self._create_expansion_prompt(idea, prompt, context)
                child_ideas = await self._generate_ideas_async(
                    prompt=expansion_prompt,
                    context=context,
                    depth=next_depth,
                    parent_id=idea.id,
                )

                to_expand.extend([(child, next_depth) for child in child_ideas])

    # -------------------------------------------------------------------------
    # CORE GENERATION AND EVALUATION METHODS
    # -------------------------------------------------------------------------

    def _generate_ideas(
        self,
        prompt: str,
        context: Optional[str],
        depth: int,
        parent_id: Optional[str],
    ) -> List[BrainstormIdea]:
        """Generate and evaluate ideas for a given prompt."""
        iteration_start = time.time()
        self._iteration_counter += 1

        if self.verbose:
            verbose_print(
                f"[Iteration {self._iteration_counter}] Depth {depth} | Parent: {parent_id or 'ROOT'}",
                "debug"
            )
            verbose_print(f"Generating {self.ideas_per_level} ideas...", "debug")

        # Build generation prompt using template resolution
        generation_prompt = self._build_generation_prompt(prompt, context)

        # Resolve system prompt (per-call > instance > default)
        system_prompt = (
            getattr(self, '_active_system_prompt', None)
            or self.system_prompt
            or self.DEFAULT_SYSTEM_PROMPT
        )

        # Generate ideas using structured output
        try:
            idea_gen = generate_pydantic_json_model(
                model_class=IdeaGeneration,
                prompt=generation_prompt,
                llm_instance=self.llm,
                max_retries=3,
                system_prompt=system_prompt,
            )

            if isinstance(idea_gen, str):  # Error case
                if self.verbose:
                    verbose_print(f"Failed to generate ideas: {idea_gen}", "error")
                return []

            # Ensure we have matching counts
            ideas_list = idea_gen.ideas[:self.ideas_per_level]
            reasoning_list = idea_gen.reasoning_per_idea[:len(ideas_list)]
            while len(reasoning_list) < len(ideas_list):
                reasoning_list.append("")

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating ideas: {str(e)}", "error")
            return []

        # Evaluate each idea
        brainstorm_ideas = []
        for idx, (idea_text, reasoning) in enumerate(zip(ideas_list, reasoning_list)):
            idea_id = f"idea_{self._iteration_counter}_{idx}"

            # Evaluate the idea
            evaluation = self._evaluate_idea(idea_text, prompt)

            # Create BrainstormIdea object
            brainstorm_idea = BrainstormIdea(
                id=idea_id,
                text=idea_text,
                reasoning=reasoning,
                quality_score=evaluation.quality_score if evaluation else 5.0,
                depth=depth,
                parent_id=parent_id,
                iteration=self._iteration_counter,
                criteria_scores=evaluation.criteria_scores if evaluation else {},
            )

            brainstorm_ideas.append(brainstorm_idea)
            self._all_ideas.append(brainstorm_idea)

            # Update tree structure
            if parent_id:
                if parent_id not in self._tree_structure:
                    self._tree_structure[parent_id] = []
                self._tree_structure[parent_id].append(idea_id)

            if self.verbose:
                verbose_print(f"[{brainstorm_idea.quality_score:.1f}] {idea_text[:70]}...", "debug")

        # Record iteration
        iteration = BrainstormIteration(
            iteration_number=self._iteration_counter,
            depth=depth,
            parent_idea=next((idea for idea in self._all_ideas if idea.id == parent_id), None) if parent_id else None,
            generated_ideas=brainstorm_ideas,
            prompt_used=prompt,
            mode_used=self.mode.value,
            execution_time=time.time() - iteration_start,
            provider_used=str(self.llm.provider.value) if hasattr(self.llm.provider, 'value') else str(self.llm.provider),
            model_used=self.llm.model_name,
        )
        self._all_iterations.append(iteration)

        return brainstorm_ideas

    async def _generate_ideas_async(
        self,
        prompt: str,
        context: Optional[str],
        depth: int,
        parent_id: Optional[str],
    ) -> List[BrainstormIdea]:
        """Async version of _generate_ideas."""
        # Note: generate_pydantic_json_model doesn't have async version yet
        # For now, we'll run it in executor. In future, use async LLM methods
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_ideas,
            prompt,
            context,
            depth,
            parent_id,
        )

    def _evaluate_idea(self, idea_text: str, original_prompt: str) -> Optional[IdeaEvaluation]:
        """Evaluate a single idea against criteria."""
        evaluation_prompt = self._build_evaluation_prompt(idea_text, original_prompt)

        # Resolve system prompt (per-call > instance > default)
        system_prompt = (
            getattr(self, '_active_system_prompt', None)
            or self.system_prompt
            or self.DEFAULT_SYSTEM_PROMPT
        )

        try:
            evaluation = generate_pydantic_json_model(
                model_class=IdeaEvaluation,
                prompt=evaluation_prompt,
                llm_instance=self.llm,
                max_retries=2,
                temperature=0.3,  # Lower temperature for more consistent evaluation
                system_prompt=system_prompt,
            )

            if isinstance(evaluation, str):  # Error case
                return None

            return evaluation

        except Exception as e:
            if self.verbose:
                verbose_print(f"Warning: Evaluation failed: {str(e)}", "error")
            return None

    # -------------------------------------------------------------------------
    # PROMPT BUILDING
    # -------------------------------------------------------------------------

    def _build_generation_prompt(self, prompt: str, context: Optional[str]) -> str:
        """
        Build the prompt for generating ideas.

        Template resolution (fallback chain):
        1. Per-call parameter (_active_generation_template)
        2. Instance attribute (self.generation_template)
        3. Class constant default (DEFAULT_GENERATION_TEMPLATE)
        """
        # Resolve template
        template = (
            getattr(self, '_active_generation_template', None)
            or self.generation_template
            or self.DEFAULT_GENERATION_TEMPLATE
        )

        context_section = f"\n\nContext:\n{context}" if context else ""

        return template.format(
            ideas_count=self.ideas_per_level,
            prompt=prompt,
            context=context_section,
        )

    def _build_evaluation_prompt(self, idea: str, original_prompt: str) -> str:
        """
        Build the prompt for evaluating an idea.

        Template resolution (fallback chain):
        1. Per-call parameter (_active_evaluation_template)
        2. Instance attribute (self.evaluation_template)
        3. Class constant default (DEFAULT_EVALUATION_TEMPLATE)
        """
        # Resolve template
        template = (
            getattr(self, '_active_evaluation_template', None)
            or self.evaluation_template
            or self.DEFAULT_EVALUATION_TEMPLATE
        )

        criteria_text = "\n".join([f"- {criterion}" for criterion in self.evaluation_criteria])

        return template.format(
            original_prompt=original_prompt,
            idea=idea,
            criteria_text=criteria_text,
        )

    def _create_expansion_prompt(
        self,
        parent_idea: BrainstormIdea,
        original_prompt: str,
        context: Optional[str],
    ) -> str:
        """
        Create a prompt for expanding/refining a specific idea.

        Template resolution (fallback chain):
        1. Per-call parameter (_active_expansion_template)
        2. Instance attribute (self.expansion_template)
        3. Class constant default (DEFAULT_EXPANSION_TEMPLATE)
        """
        # Resolve template
        template = (
            getattr(self, '_active_expansion_template', None)
            or self.expansion_template
            or self.DEFAULT_EXPANSION_TEMPLATE
        )

        context_section = f"\n\nOriginal Context:\n{context}" if context else ""

        return template.format(
            original_prompt=original_prompt,
            ideas_count=self.ideas_per_level,
            parent_idea=parent_idea.text,
            parent_reasoning=parent_idea.reasoning,
            parent_score=f"{parent_idea.quality_score:.1f}",
            context=context_section,
        )

    # -------------------------------------------------------------------------
    # RESULT BUILDING
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        initial_prompt: str,
        mode: BrainstormMode,
        execution_time: float,
    ) -> BrainstormResult:
        """Build the final BrainstormResult object."""
        # Organize ideas by depth
        max_depth = max([idea.depth for idea in self._all_ideas], default=0)
        levels = []

        for depth in range(max_depth + 1):
            ideas_at_depth = [idea for idea in self._all_ideas if idea.depth == depth]
            if ideas_at_depth:
                level = BrainstormLevel(
                    depth=depth,
                    ideas=ideas_at_depth,
                    total_ideas=len(ideas_at_depth),
                )
                levels.append(level)

        # Find best ideas
        best_per_level = [level.best_idea for level in levels if level.best_idea]
        overall_best = max(self._all_ideas, key=lambda x: x.quality_score, default=None)

        # Determine stop reason
        if max_depth >= self.max_depth - 1:
            stopped_reason = "max_depth_reached"
        elif overall_best and overall_best.quality_score < self.min_quality_threshold:
            stopped_reason = "quality_threshold_not_met"
        else:
            stopped_reason = "natural_completion"

        # Build config
        config = {
            "max_depth": self.max_depth,
            "ideas_per_level": self.ideas_per_level,
            "mode": mode.value,
            "top_n": self.top_n if mode == BrainstormMode.HYBRID else None,
            "evaluation_criteria": self.evaluation_criteria,
            "min_quality_threshold": self.min_quality_threshold,
        }

        return BrainstormResult(
            initial_prompt=initial_prompt,
            mode=mode,
            total_ideas=len(self._all_ideas),
            total_iterations=self._iteration_counter,
            max_depth_reached=max_depth,
            levels=levels,
            all_ideas=self._all_ideas,
            best_ideas_per_level=best_per_level,
            overall_best_idea=overall_best,
            all_iterations=self._all_iterations,
            execution_time=execution_time,
            stopped_reason=stopped_reason,
            tree_structure=self._tree_structure,
            config_used=config,
            timestamp=datetime.now(),
        )
