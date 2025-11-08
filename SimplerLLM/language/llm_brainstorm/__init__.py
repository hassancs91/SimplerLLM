"""
Recursive Brainstorm - Generate and expand ideas recursively using LLMs.

This module provides flexible brainstorming capabilities with three generation modes:
- Tree mode: Exponential expansion of all ideas
- Linear mode: Focused refinement of the best ideas
- Hybrid mode: Selective expansion of top-N ideas

Example:
    >>> from SimplerLLM.language import LLM, LLMProvider
    >>> from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm
    >>>
    >>> llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4o")
    >>> brainstorm = RecursiveBrainstorm(
    ...     llm=llm,
    ...     max_depth=3,
    ...     ideas_per_level=5,
    ...     mode="tree"
    ... )
    >>>
    >>> result = brainstorm.brainstorm("Ways to reduce carbon emissions")
    >>> print(f"Generated {result.total_ideas} ideas")
    >>> print(f"Best idea: {result.overall_best_idea.text}")
"""

from .recursive_brainstorm import RecursiveBrainstorm
from .models import (
    BrainstormIdea,
    BrainstormLevel,
    BrainstormIteration,
    BrainstormResult,
    IdeaGeneration,
    IdeaEvaluation,
)

__all__ = [
    "RecursiveBrainstorm",
    "BrainstormIdea",
    "BrainstormLevel",
    "BrainstormIteration",
    "BrainstormResult",
    "IdeaGeneration",
    "IdeaEvaluation",
]
