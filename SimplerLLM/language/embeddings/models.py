"""
Embeddings models and enumerations.

This module defines the data structures and enumerations used by the
embeddings module.
"""

from enum import Enum


class EmbeddingsProvider(Enum):
    """
    Enumeration of supported embedding providers.

    Each provider has different models, capabilities, and pricing.
    Choose based on your requirements for quality, speed, and cost.

    Attributes:
        OPENAI: OpenAI's text-embedding models. Best general-purpose choice.
            Models: text-embedding-3-small, text-embedding-3-large
            Dimensions: 1536 (small), 3072 (large)

        VOYAGE: Voyage AI embeddings. Excellent for retrieval tasks.
            Models: voyage-3, voyage-3-lite, voyage-code-3
            Features: input_type optimization, custom dimensions

        COHERE: Cohere embeddings. Strong multilingual support.
            Models: embed-english-v3.0, embed-multilingual-v3.0
            Features: input_type for search optimization, truncation control

        OPENROUTER: OpenRouter unified gateway. Access embedding models
            from multiple providers through a single API.
            Models use 'provider/model' format:
            openai/text-embedding-3-small, qwen/qwen3-embedding-8b

        COMETAPI: CometAPI all-in-one aggregator. Access OpenAI embedding
            models (and more) through a single CometAPI key, using native
            model names: text-embedding-3-small, text-embedding-3-large

    Example:
        >>> from SimplerLLM.language.embeddings import EmbeddingsProvider
        >>> provider = EmbeddingsProvider.OPENAI
        >>> print(provider.name)
        'OPENAI'
    """
    OPENAI = 1
    VOYAGE = 2
    COHERE = 3
    OPENROUTER = 4
    COMETAPI = 5
