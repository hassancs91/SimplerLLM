"""
Embeddings Module - Unified Interface for Text Embeddings.

This module provides a factory-based interface for generating text embeddings
using multiple providers (OpenAI, Voyage AI, Cohere). It supports both
synchronous and asynchronous operations with provider-specific optimizations.

Supported Providers:
    OpenAI:
        - text-embedding-3-small (default): 1536 dimensions, fast and cost-effective
        - text-embedding-3-large: 3072 dimensions, highest quality
        - text-embedding-ada-002: Legacy model, 1536 dimensions

    Voyage AI:
        - voyage-3: Latest multilingual model (1024 dimensions)
        - voyage-3-lite: Fast, cost-effective (512 dimensions)
        - voyage-code-3: Code-optimized embeddings
        - voyage-finance-2: Finance domain specialist

    Cohere:
        - embed-english-v3.0: English-optimized (1024 dimensions)
        - embed-multilingual-v3.0: 100+ languages (1024 dimensions)
        - embed-v4.0: Latest model with configurable dimensions

Environment Variables:
    OPENAI_API_KEY: Required for OpenAI embeddings
    VOYAGE_API_KEY: Required for Voyage AI embeddings
    COHERE_API_KEY: Required for Cohere embeddings

Quick Start:
    >>> from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider
    >>>
    >>> # Create OpenAI embeddings instance
    >>> embeddings = EmbeddingsLLM.create(
    ...     provider=EmbeddingsProvider.OPENAI,
    ...     model_name="text-embedding-3-small"
    ... )
    >>>
    >>> # Generate embedding for a single text
    >>> vector = embeddings.generate_embeddings("Hello, world!")
    >>> print(f"Embedding dimension: {len(vector)}")
    >>>
    >>> # Generate embeddings for multiple texts
    >>> vectors = embeddings.generate_embeddings(["Text 1", "Text 2"])
    >>> print(f"Generated {len(vectors)} embeddings")

Retrieval-Optimized Usage (Voyage/Cohere):
    >>> # For search queries (finding similar content)
    >>> query_emb = embeddings.generate_embeddings(
    ...     "What is machine learning?",
    ...     input_type="query"
    ... )
    >>>
    >>> # For documents (content to be searched)
    >>> doc_emb = embeddings.generate_embeddings(
    ...     "Machine learning is a subset of AI...",
    ...     input_type="document"
    ... )

Async Usage:
    >>> import asyncio
    >>>
    >>> async def embed_texts():
    ...     embeddings = EmbeddingsLLM.create(
    ...         provider=EmbeddingsProvider.OPENAI
    ...     )
    ...     return await embeddings.generate_embeddings_async("Hello")
    >>>
    >>> vector = asyncio.run(embed_texts())

See Also:
    - SimplerLLM.language.llm_providers.llm_response_models.LLMEmbeddingsResponse
    - SimplerLLM.vectors for vector database integration
"""

from .models import EmbeddingsProvider
from .base import EmbeddingsLLM
from .providers import (
    BaseEmbeddings,
    OpenAIEmbeddings,
    VoyageEmbeddings,
    CohereEmbeddings,
)

__all__ = [
    # Main factory and enum
    "EmbeddingsLLM",
    "EmbeddingsProvider",
    # Provider implementations
    "BaseEmbeddings",
    "OpenAIEmbeddings",
    "VoyageEmbeddings",
    "CohereEmbeddings",
]
