"""
Vectors Module - Unified Interface for Vector Databases.

This module provides a factory-based interface for vector storage and
similarity search using multiple database providers.

Supported Providers:
    LOCAL:
        - In-memory storage with NumPy arrays
        - Pickle-based file persistence (.svdb format)
        - Fast cosine similarity search
        - Optional vector compression
        - Best for: development, testing, small datasets

    QDRANT:
        - Self-hosted or Qdrant Cloud deployment
        - Automatic persistence and recovery
        - Advanced metadata filtering
        - Efficient batch operations
        - Best for: production, large datasets, distributed systems

Environment Variables:
    QDRANT_API_KEY: API key for Qdrant Cloud (optional)

Quick Start:
    >>> from SimplerLLM.vectors import VectorDB, VectorProvider
    >>>
    >>> # Create local vector database
    >>> db = VectorDB.create(
    ...     provider=VectorProvider.LOCAL,
    ...     db_folder="./my_vectors"
    ... )
    >>>
    >>> # Add vectors
    >>> vector_id = db.add_vector(
    ...     vector=[0.1, 0.2, 0.3, 0.4],
    ...     meta={"text": "Hello world", "source": "user"}
    ... )
    >>>
    >>> # Search for similar vectors
    >>> results = db.top_cosine_similarity(
    ...     target_vector=[0.1, 0.2, 0.3, 0.4],
    ...     top_n=5
    ... )
    >>> for vid, meta, score in results:
    ...     print(f"Score: {score:.3f}, Text: {meta.get('text')}")
    >>>
    >>> # Save to disk
    >>> db.save_to_disk("my_collection")

RAG (Retrieval-Augmented Generation) Example:
    >>> from SimplerLLM.vectors import VectorDB, VectorProvider
    >>> from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider
    >>>
    >>> # Create embeddings instance
    >>> embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)
    >>>
    >>> # Create vector database
    >>> db = VectorDB.create(provider=VectorProvider.LOCAL, db_folder="./rag_db")
    >>>
    >>> # Add documents with embeddings
    >>> documents = ["Machine learning is...", "Neural networks are..."]
    >>> for doc in documents:
    ...     embedding = embeddings.generate_embeddings(doc)
    ...     db.add_text_with_embedding(text=doc, embedding=embedding)
    >>>
    >>> # Search by text
    >>> results = db.search_by_text(
    ...     query_text="What is deep learning?",
    ...     embeddings_llm_instance=embeddings,
    ...     top_n=3
    ... )

Qdrant Cloud Example:
    >>> db = VectorDB.create(
    ...     provider=VectorProvider.QDRANT,
    ...     url="https://your-cluster.qdrant.io",
    ...     api_key="your-api-key",
    ...     collection_name="production"
    ... )

See Also:
    - SimplerLLM.language.embeddings for generating embeddings
    - SimplerLLM.tools.text_chunker for semantic text chunking
"""

# Core factory and enum
from .base import VectorDB, VectorProvider

# Response models
from .models import (
    VectorSearchResult,
    VectorStats,
    VectorOperationResult,
    VectorQueryResult,
)

# Exceptions
from .exceptions import (
    VectorDBError,
    VectorNotFoundError,
    DimensionMismatchError,
    VectorDBConnectionError,
    VectorDBOperationError,
)

# Wrapper implementations (high-level)
from .wrappers import LocalVectorDB, QdrantVectorDB

# Provider implementations (low-level)
from .providers import SimplerVectors, SerializationFormat

__all__ = [
    # Core factory and enum
    'VectorDB',
    'VectorProvider',
    # Response models
    'VectorSearchResult',
    'VectorStats',
    'VectorOperationResult',
    'VectorQueryResult',
    # Exceptions
    'VectorDBError',
    'VectorNotFoundError',
    'DimensionMismatchError',
    'VectorDBConnectionError',
    'VectorDBOperationError',
    # Wrapper implementations
    'LocalVectorDB',
    'QdrantVectorDB',
    # Provider implementations
    'SimplerVectors',
    'SerializationFormat',
]
