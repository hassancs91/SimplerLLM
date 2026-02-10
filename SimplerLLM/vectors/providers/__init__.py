"""
Vector Database Providers.

Low-level provider implementations for vector database operations.
These modules contain the core logic for each database type.

Available Providers:
    SimplerVectors: In-memory vector storage with NumPy
    SerializationFormat: Enum for serialization formats
    QdrantProvider: Qdrant database operations

Note:
    For most use cases, use the high-level wrappers in
    SimplerLLM.vectors.wrappers instead of these low-level providers.

Example:
    >>> from SimplerLLM.vectors.providers import SimplerVectors
    >>> db = SimplerVectors(db_folder="./vectors")
"""

from .local_provider import SimplerVectors, SerializationFormat
from .qdrant_provider import QdrantProvider

__all__ = [
    'SimplerVectors',
    'SerializationFormat',
    'QdrantProvider',
]
