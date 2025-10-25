"""
LLM Judge - Multi-provider orchestration and evaluation system.

This module provides tools for orchestrating multiple LLM providers,
evaluating their responses, and generating comparative analyses or synthesized answers.

Main Classes:
    - LLMJudge: Orchestrates multiple providers and evaluates responses
    - JudgeMode: Enum for evaluation modes (select_best, synthesize, compare)
    - JudgeResult: Complete result with evaluations and metadata
    - EvaluationReport: Statistical summary for batch evaluations

Example:
    ```python
    from SimplerLLM.language import LLM, LLMProvider
    from SimplerLLM.language.llm_judge import LLMJudge, JudgeMode

    # Create providers
    providers = [
        LLM.create(LLMProvider.OPENAI, model_name="gpt-4"),
        LLM.create(LLMProvider.ANTHROPIC, model_name="claude-sonnet-4"),
    ]
    judge_llm = LLM.create(LLMProvider.ANTHROPIC, model_name="claude-opus-4")

    # Initialize judge
    judge = LLMJudge(providers=providers, judge_llm=judge_llm)

    # Evaluate
    result = judge.generate("Explain quantum computing", mode="synthesize")
    print(result.final_answer)
    print(result.confidence_scores)
    ```
"""

from .judge import LLMJudge
from .models import (
    JudgeMode,
    JudgeResult,
    ProviderResponse,
    ProviderEvaluation,
    EvaluationReport,
    RouterSummary,
)

__all__ = [
    "LLMJudge",
    "JudgeMode",
    "JudgeResult",
    "ProviderResponse",
    "ProviderEvaluation",
    "EvaluationReport",
    "RouterSummary",
]
