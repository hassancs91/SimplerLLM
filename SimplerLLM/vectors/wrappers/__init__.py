"""
Vector Database Wrappers.

High-level wrapper classes for vector database providers, providing
unified interfaces with comprehensive documentation and logging.

Available Wrappers:
    LocalVectorDB: In-memory vector database with file persistence
    QdrantVectorDB: Qdrant vector database (self-hosted or cloud)

Example:
    >>> from SimplerLLM.vectors import VectorDB, VectorProvider
    >>>
    >>> # Using factory (recommended)
    >>> db = VectorDB.create(provider=VectorProvider.LOCAL, db_folder="./vectors")
    >>>
    >>> # Direct wrapper import
    >>> from SimplerLLM.vectors.wrappers import LocalVectorDB
    >>> db = LocalVectorDB(db_folder="./vectors")
"""

from .local_wrapper import LocalVectorDB
from .qdrant_wrapper import QdrantVectorDB

__all__ = [
    'LocalVectorDB',
    'QdrantVectorDB',
]
