"""
LLM Add-ons Module - Structured generation and utility functions.

This module provides high-level utilities for structured LLM output generation:

- **JSON/Pydantic Generation**: Generate validated Pydantic model instances from LLM responses
  with automatic retry logic and exponential backoff.

- **Pattern Extraction**: Extract structured patterns (emails, phones, URLs, etc.) from LLM
  responses with validation and normalization.

- **Cost Utilities**: Calculate token counts and API costs.

Quick Start:
    >>> from SimplerLLM.language import LLM, LLMProvider
    >>> from SimplerLLM.language.llm_addons import generate_pydantic_json_model
    >>> from pydantic import BaseModel
    >>>
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    ...     city: str
    >>>
    >>> llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    >>> person = generate_pydantic_json_model(
    ...     model_class=Person,
    ...     prompt="Generate a fictional person",
    ...     llm_instance=llm
    ... )
    >>> print(person.name)

Pattern Extraction Example:
    >>> from SimplerLLM.language.llm_addons import generate_structured_pattern
    >>>
    >>> result = generate_structured_pattern(
    ...     pattern="email",
    ...     prompt="What is the support email for OpenAI?",
    ...     llm_instance=llm,
    ...     validate=True
    ... )
    >>> print(result.matches[0].value)

Available Functions:
    JSON Generation:
        - generate_pydantic_json_model: Sync JSON generation
        - generate_pydantic_json_model_async: Async JSON generation
        - generate_pydantic_json_model_reliable: Sync with fallback
        - generate_pydantic_json_model_reliable_async: Async with fallback
        - create_optimized_prompt: Create prompts with JSON format instructions

    Pattern Extraction:
        - generate_structured_pattern: Sync pattern extraction
        - generate_structured_pattern_async: Async pattern extraction
        - generate_structured_pattern_reliable: Sync with fallback
        - generate_structured_pattern_reliable_async: Async with fallback

    Utilities:
        - calculate_text_generation_costs: Calculate API costs
"""

from .json_generation import (
    create_optimized_prompt,
    generate_pydantic_json_model,
    generate_pydantic_json_model_reliable,
    generate_pydantic_json_model_async,
    generate_pydantic_json_model_reliable_async,
)

from .pattern_extraction import (
    generate_structured_pattern,
    generate_structured_pattern_async,
    generate_structured_pattern_reliable,
    generate_structured_pattern_reliable_async,
)

from .cost_utils import calculate_text_generation_costs

__all__ = [
    # JSON Generation
    "create_optimized_prompt",
    "generate_pydantic_json_model",
    "generate_pydantic_json_model_reliable",
    "generate_pydantic_json_model_async",
    "generate_pydantic_json_model_reliable_async",
    # Pattern Extraction
    "generate_structured_pattern",
    "generate_structured_pattern_async",
    "generate_structured_pattern_reliable",
    "generate_structured_pattern_reliable_async",
    # Utilities
    "calculate_text_generation_costs",
]
