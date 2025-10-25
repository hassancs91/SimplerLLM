"""
LLM Provider Router - Smart wrapper for intelligent provider routing.

This module provides automatic query classification and provider routing,
making it easy to use multiple LLM providers intelligently.

Main Classes:
    - LLMProviderRouter: Main router class for automatic provider selection
    - ProviderConfig: Configuration for individual providers
    - RoutingResult: Complete result with answer and metadata
    - QueryClassification: Query classification details

Example:
    ```python
    from SimplerLLM.language import LLM, LLMProvider
    from SimplerLLM.language.llm_provider_router import (
        LLMProviderRouter,
        ProviderConfig
    )

    # Configure providers
    providers = [
        ProviderConfig(
            llm_provider="OPENAI",
            llm_model="gpt-4",
            specialties=["coding", "technical"],
            description="Best for code"
        ),
    ]

    # Create LLM instances
    llm_instances = [LLM.create(LLMProvider.OPENAI, model_name="gpt-4")]

    # Initialize router
    router = LLMProviderRouter(
        provider_configs=providers,
        llm_instances=llm_instances
    )

    # Route and execute
    result = router.route("Write a Python function to reverse a string")
    print(result.answer)
    print(f"Used: {result.provider_used}")
    ```
"""

from .provider_router import LLMProviderRouter
from .query_classifier import QueryClassifier
from .models import (
    ProviderConfig,
    QueryClassification,
    RoutingResult,
    RouterConfig,
)

__all__ = [
    "LLMProviderRouter",
    "QueryClassifier",
    "ProviderConfig",
    "QueryClassification",
    "RoutingResult",
    "RouterConfig",
]
