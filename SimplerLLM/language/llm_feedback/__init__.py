"""
LLM Feedback Loop - Iterative self-improvement system.

This module provides tools for iteratively refining LLM responses through
critique and improvement cycles. Supports multiple architectural patterns:
- Single provider self-critique
- Dual provider (generator + critic)
- Multi-provider rotation

Main Classes:
    - LLMFeedbackLoop: Main class for iterative improvement
    - FeedbackResult: Complete result with history and final answer
    - IterationResult: Result from a single iteration
    - Critique: Structured critique model
    - FeedbackConfig: Configuration options

Example:
    ```python
    from SimplerLLM.language import LLM, LLMProvider
    from SimplerLLM.language.llm_feedback import LLMFeedbackLoop

    # Single provider self-critique
    llm = LLM.create(LLMProvider.OPENAI, model_name="gpt-4")
    feedback = LLMFeedbackLoop(llm=llm, max_iterations=3)
    result = feedback.improve("Explain quantum computing")

    print(f"Improvement: {result.initial_score} → {result.final_score}")
    print(result.final_answer)
    ```
"""

from .feedback_loop import LLMFeedbackLoop
from .models import (
    Critique,
    IterationResult,
    FeedbackResult,
    FeedbackConfig,
    TemperatureSchedule,
)

__all__ = [
    "LLMFeedbackLoop",
    "Critique",
    "IterationResult",
    "FeedbackResult",
    "FeedbackConfig",
    "TemperatureSchedule",
]
