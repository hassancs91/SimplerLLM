"""
LLM Validator - Multi-provider validation system for AI-generated content.

This module provides tools for validating AI-generated content using multiple
LLM providers, with configurable aggregation methods and consensus detection.

Main Classes:
    - LLMValidator: Validates content using multiple LLM providers
    - AggregationMethod: Enum for score aggregation methods
    - ValidationResult: Complete validation result with scores and metadata
    - ValidatorScore: Individual validator's score and explanation

Example:
    ```python
    from SimplerLLM.language import LLM, LLMProvider
    from SimplerLLM.language.llm_validator import LLMValidator

    # Create validators
    validators = [
        LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),
        LLM.create(LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022"),
    ]

    # Initialize validator
    validator = LLMValidator(validators=validators)

    # Validate content
    result = validator.validate(
        content="Paris is the capital of France.",
        validation_prompt="Check if the facts are accurate.",
        original_question="What is the capital of France?",
    )
    print(f"Score: {result.overall_score}")
    print(f"Valid: {result.is_valid}")
    print(f"Consensus: {result.consensus}")
    ```
"""

from .validator import LLMValidator
from .models import (
    AggregationMethod,
    ValidationResult,
    ValidatorScore,
)

__all__ = [
    "LLMValidator",
    "AggregationMethod",
    "ValidationResult",
    "ValidatorScore",
]
