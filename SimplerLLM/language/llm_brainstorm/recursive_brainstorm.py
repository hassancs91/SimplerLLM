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
from .models import (
    BrainstormIdea,
    BrainstormLevel,
    BrainstormIteration,
    BrainstormResult,
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

    Attributes:
        llm: The LLM instance to use for generation
        max_depth: Maximum recursion depth
        ideas_per_level: Number of ideas to generate per expansion
        mode: Default generation mode ("tree", "linear", or "hybrid")
        top_n: For hybrid mode, how many top ideas to expand
        evaluation_criteria: List of criteria to evaluate ideas against
        min_quality_threshold: Minimum quality score to continue expanding (1-10)
        verbose: Whether to print progress information
    """

    def __init__(
        self,
        llm: LLM,
        max_depth: int = 3,
        ideas_per_level: int = 5,
        mode: str = "tree",
        top_n: int = 3,
        evaluation_criteria: Optional[List[str]] = None,
        min_quality_threshold: float = 5.0,
        verbose: bool = False,
    ):
        """
        Initialize the RecursiveBrainstorm instance.

        Args:
            llm: LLM instance for generation
            max_depth: Maximum recursion depth (default: 3)
            ideas_per_level: Ideas to generate per expansion (default: 5)
            mode: Default mode - "tree", "linear", or "hybrid" (default: "tree")
            top_n: For hybrid mode, number of top ideas to expand (default: 3)
            evaluation_criteria: List of criteria like ["feasibility", "impact", "novelty"]
            min_quality_threshold: Minimum score to expand (1-10, default: 5.0)
            verbose: Print progress information (default: False)

        Raises:
            ValueError: If mode is not one of "tree", "linear", "hybrid"
            ValueError: If max_depth < 1 or ideas_per_level < 1
        """
        if mode not in ["tree", "linear", "hybrid"]:
            raise ValueError("Mode must be 'tree', 'linear', or 'hybrid'")
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

        # Internal state
        self._iteration_counter = 0
        self._all_ideas: List[BrainstormIdea] = []
        self._all_iterations: List[BrainstormIteration] = []
        self._tree_structure: Dict[str, List[str]] = {}

    def brainstorm(
        self,
        prompt: str,
        mode: Optional[str] = None,
        context: Optional[str] = None,
        save_csv: bool = False,
        csv_path: str = "brainstorm_results.csv",
        csv_expand_criteria: bool = True,
    ) -> BrainstormResult:
        """
        Execute recursive brainstorming session.

        Args:
            prompt: The initial brainstorming prompt
            mode: Override default mode ("tree", "linear", or "hybrid")
            context: Additional context to guide idea generation
            save_csv: Whether to automatically save results to CSV file
            csv_path: Path to CSV file (used if save_csv=True)
            csv_expand_criteria: Whether to expand criteria scores into separate columns

        Returns:
            BrainstormResult containing all generated ideas and metadata

        Example:
            >>> from SimplerLLM.language import LLM, LLMProvider
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
        mode = mode or self.mode

        # Reset internal state
        self._iteration_counter = 0
        self._all_ideas = []
        self._all_iterations = []
        self._tree_structure = {}

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Recursive Brainstorm - Mode: {mode.upper()}")
            print(f"Max Depth: {self.max_depth} | Ideas per Level: {self.ideas_per_level}")
            print(f"{'='*60}\n")

        # Execute brainstorming based on mode
        if mode == "tree":
            self._brainstorm_tree(prompt, context)
        elif mode == "linear":
            self._brainstorm_linear(prompt, context)
        elif mode == "hybrid":
            self._brainstorm_hybrid(prompt, context)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        execution_time = time.time() - start_time

        # Build result
        result = self._build_result(
            initial_prompt=prompt,
            mode=mode,
            execution_time=execution_time,
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Brainstorm Complete!")
            print(f"Total Ideas: {result.total_ideas} | Total Iterations: {result.total_iterations}")
            print(f"Max Depth: {result.max_depth_reached} | Time: {result.execution_time:.2f}s")
            if result.overall_best_idea:
                print(f"Best Idea (Score: {result.overall_best_idea.quality_score:.1f}): {result.overall_best_idea.text[:80]}...")
            print(f"{'='*60}\n")

        # Auto-save to CSV if requested
        if save_csv:
            result.to_csv(csv_path, expand_criteria=csv_expand_criteria)
            if self.verbose:
                print(f"Results saved to CSV: {csv_path}")

        return result

    async def brainstorm_async(
        self,
        prompt: str,
        mode: Optional[str] = None,
        context: Optional[str] = None,
        save_csv: bool = False,
        csv_path: str = "brainstorm_results.csv",
        csv_expand_criteria: bool = True,
    ) -> BrainstormResult:
        """
        Execute recursive brainstorming session asynchronously.

        Args:
            prompt: The initial brainstorming prompt
            mode: Override default mode ("tree", "linear", or "hybrid")
            context: Additional context to guide idea generation
            save_csv: Whether to automatically save results to CSV file
            csv_path: Path to CSV file (used if save_csv=True)
            csv_expand_criteria: Whether to expand criteria scores into separate columns

        Returns:
            BrainstormResult containing all generated ideas and metadata
        """
        start_time = time.time()
        mode = mode or self.mode

        # Reset internal state
        self._iteration_counter = 0
        self._all_ideas = []
        self._all_iterations = []
        self._tree_structure = {}

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Recursive Brainstorm (ASYNC) - Mode: {mode.upper()}")
            print(f"Max Depth: {self.max_depth} | Ideas per Level: {self.ideas_per_level}")
            print(f"{'='*60}\n")

        # Execute brainstorming based on mode
        if mode == "tree":
            await self._brainstorm_tree_async(prompt, context)
        elif mode == "linear":
            await self._brainstorm_linear_async(prompt, context)
        elif mode == "hybrid":
            await self._brainstorm_hybrid_async(prompt, context)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        execution_time = time.time() - start_time

        # Build result
        result = self._build_result(
            initial_prompt=prompt,
            mode=mode,
            execution_time=execution_time,
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Async Brainstorm Complete!")
            print(f"Total Ideas: {result.total_ideas} | Total Iterations: {result.total_iterations}")
            print(f"Best Idea (Score: {result.overall_best_idea.quality_score:.1f}): {result.overall_best_idea.text[:80]}...")
            print(f"{'='*60}\n")

        # Auto-save to CSV if requested
        if save_csv:
            result.to_csv(csv_path, expand_criteria=csv_expand_criteria)
            if self.verbose:
                print(f"Results saved to CSV: {csv_path}")

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
                    print(f"  Skipping expansion of idea (score {current_idea.quality_score:.1f} < {self.min_quality_threshold})")
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
                print(f"  Best idea at depth {current_depth} (score: {best_idea.quality_score:.1f}): {best_idea.text[:60]}...")

            # Stop if quality is below threshold
            if best_idea.quality_score < self.min_quality_threshold:
                if self.verbose:
                    print(f"  Stopping: best idea score {best_idea.quality_score:.1f} < threshold {self.min_quality_threshold}")
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
                print(f"  Best idea at depth {current_depth} (score: {best_idea.quality_score:.1f}): {best_idea.text[:60]}...")

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
                print(f"  Depth {current_depth}: Expanding top {len(top_ideas)} ideas (scores: {[f'{i.quality_score:.1f}' for i in top_ideas]})")

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
            print(f"\n[Iteration {self._iteration_counter}] Depth {depth} | Parent: {parent_id or 'ROOT'}")
            print(f"  Generating {self.ideas_per_level} ideas...")

        # Build generation prompt
        generation_prompt = self._build_generation_prompt(prompt, context)

        # Generate ideas using structured output
        try:
            idea_gen = generate_pydantic_json_model(
                model_class=IdeaGeneration,
                prompt=generation_prompt,
                llm_instance=self.llm,
                max_retries=3,
            )

            if isinstance(idea_gen, str):  # Error case
                if self.verbose:
                    print(f"  ERROR: Failed to generate ideas: {idea_gen}")
                return []

            # Ensure we have matching counts
            ideas_list = idea_gen.ideas[:self.ideas_per_level]
            reasoning_list = idea_gen.reasoning_per_idea[:len(ideas_list)]
            while len(reasoning_list) < len(ideas_list):
                reasoning_list.append("")

        except Exception as e:
            if self.verbose:
                print(f"  ERROR generating ideas: {str(e)}")
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
                print(f"  - [{brainstorm_idea.quality_score:.1f}] {idea_text[:70]}...")

        # Record iteration
        iteration = BrainstormIteration(
            iteration_number=self._iteration_counter,
            depth=depth,
            parent_idea=next((idea for idea in self._all_ideas if idea.id == parent_id), None) if parent_id else None,
            generated_ideas=brainstorm_ideas,
            prompt_used=prompt,
            mode_used=self.mode,
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

        try:
            evaluation = generate_pydantic_json_model(
                model_class=IdeaEvaluation,
                prompt=evaluation_prompt,
                llm_instance=self.llm,
                max_retries=2,
                temperature=0.3,  # Lower temperature for more consistent evaluation
            )

            if isinstance(evaluation, str):  # Error case
                return None

            return evaluation

        except Exception as e:
            if self.verbose:
                print(f"    Warning: Evaluation failed: {str(e)}")
            return None

    # -------------------------------------------------------------------------
    # PROMPT BUILDING
    # -------------------------------------------------------------------------

    def _build_generation_prompt(self, prompt: str, context: Optional[str]) -> str:
        """Build the prompt for generating ideas."""
        context_section = f"\n\nContext:\n{context}" if context else ""

        return f"""Generate exactly {self.ideas_per_level} creative and diverse ideas based on the following prompt:

Prompt: {prompt}{context_section}

Requirements:
- Generate exactly {self.ideas_per_level} distinct ideas
- Make each idea specific and actionable
- Ensure diversity in approaches
- Provide brief reasoning for each idea

Output your ideas in structured JSON format."""

    def _build_evaluation_prompt(self, idea: str, original_prompt: str) -> str:
        """Build the prompt for evaluating an idea."""
        criteria_text = "\n".join([f"- {criterion}" for criterion in self.evaluation_criteria])

        return f"""Evaluate the following idea based on the criteria below.

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

    def _create_expansion_prompt(
        self,
        parent_idea: BrainstormIdea,
        original_prompt: str,
        context: Optional[str],
    ) -> str:
        """Create a prompt for expanding/refining a specific idea."""
        context_section = f"\n\nOriginal Context:\n{context}" if context else ""

        return f"""Based on the following idea, generate {self.ideas_per_level} refined variations, expansions, or sub-ideas:

Original Prompt: {original_prompt}{context_section}

Parent Idea: {parent_idea.text}
Reasoning: {parent_idea.reasoning}
Quality Score: {parent_idea.quality_score:.1f}/10

Generate {self.ideas_per_level} ideas that:
- Build upon or refine the parent idea
- Explore different aspects or approaches
- Add more specificity or detail
- Address potential weaknesses while leveraging strengths

Make each idea distinct and valuable."""

    # -------------------------------------------------------------------------
    # RESULT BUILDING
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        initial_prompt: str,
        mode: str,
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
            "mode": mode,
            "top_n": self.top_n if mode == "hybrid" else None,
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
