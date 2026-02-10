"""
Vector Database Response Models.

Pydantic models for vector database operations, providing type-safe
response objects with metadata for all CRUD and search operations.

Models:
    VectorSearchResult: Result from similarity search
    VectorStats: Database statistics
    VectorOperationResult: Result from CRUD operations

Example:
    >>> from SimplerLLM.vectors import VectorDB, VectorProvider
    >>>
    >>> db = VectorDB.create(provider=VectorProvider.LOCAL, db_folder="./vectors")
    >>> results = db.top_cosine_similarity(query_vector, top_n=5, full_response=True)
    >>> for result in results:
    ...     print(f"ID: {result.vector_id}, Score: {result.similarity:.3f}")
"""

from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict, Union
from datetime import datetime


class VectorSearchResult(BaseModel):
    """
    Result from a vector similarity search.

    Contains the matched vector's ID, metadata, similarity score,
    and optionally the vector itself.

    Attributes:
        vector_id: Unique identifier of the matched vector
        metadata: Metadata associated with the vector (dict, str, or any type)
        similarity: Cosine similarity score (0 to 1, higher is more similar)
        vector: The actual vector values (optional, included if requested)

    Example:
        >>> results = db.top_cosine_similarity(query_vec, top_n=3, full_response=True)
        >>> for result in results:
        ...     print(f"ID: {result.vector_id}")
        ...     print(f"Similarity: {result.similarity:.4f}")
        ...     print(f"Metadata: {result.metadata}")
    """
    vector_id: str = Field(description="Unique identifier of the matched vector")
    metadata: Any = Field(description="Metadata associated with the vector")
    similarity: float = Field(description="Cosine similarity score (0-1)")
    vector: Optional[List[float]] = Field(
        default=None,
        description="The actual vector values (optional)"
    )

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True


class VectorStats(BaseModel):
    """
    Statistics about a vector database.

    Provides overview information about the database including
    vector count, dimension, and provider-specific details.

    Attributes:
        total_vectors: Number of vectors stored in the database
        dimension: Vector dimension (None if no vectors added yet)
        provider: Provider name ("local" or "qdrant")
        size_in_memory_mb: Memory usage in MB (local provider only)
        collection_name: Collection name (qdrant provider only)
        metadata_keys: List of unique metadata keys used (optional)
        created_at: When the database was created (optional)

    Example:
        >>> stats = db.get_stats(full_response=True)
        >>> print(f"Vectors: {stats.total_vectors}")
        >>> print(f"Dimension: {stats.dimension}")
        >>> print(f"Provider: {stats.provider}")
    """
    total_vectors: int = Field(description="Number of vectors in the database")
    dimension: Optional[int] = Field(
        default=None,
        description="Vector dimension (None if empty)"
    )
    provider: str = Field(description="Provider name (local, qdrant)")
    size_in_memory_mb: Optional[float] = Field(
        default=None,
        description="Memory usage in MB (local only)"
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Collection name (qdrant only)"
    )
    metadata_keys: Optional[List[str]] = Field(
        default=None,
        description="Unique metadata keys used"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Database creation timestamp"
    )


class VectorOperationResult(BaseModel):
    """
    Result from a vector CRUD operation.

    Provides feedback about add, update, or delete operations
    including success status and affected vector ID.

    Attributes:
        success: Whether the operation completed successfully
        vector_id: ID of the affected vector (None for batch operations)
        operation: Type of operation ("add", "update", "delete", "batch_add")
        message: Optional message with details or error information
        count: Number of vectors affected (for batch operations)

    Example:
        >>> result = db.add_vector(vector, meta, full_response=True)
        >>> if result.success:
        ...     print(f"Added vector: {result.vector_id}")
        ... else:
        ...     print(f"Failed: {result.message}")
    """
    success: bool = Field(description="Whether the operation succeeded")
    vector_id: Optional[str] = Field(
        default=None,
        description="ID of the affected vector"
    )
    operation: str = Field(description="Type of operation performed")
    message: Optional[str] = Field(
        default=None,
        description="Optional message or error details"
    )
    count: Optional[int] = Field(
        default=None,
        description="Number of vectors affected (batch operations)"
    )


class VectorQueryResult(BaseModel):
    """
    Result from a metadata query.

    Contains vectors matching the metadata query criteria.

    Attributes:
        vector_id: Unique identifier of the matched vector
        vector: The vector values as a list of floats
        metadata: Metadata associated with the vector

    Example:
        >>> results = db.query_by_metadata(source="wikipedia", full_response=True)
        >>> for result in results:
        ...     print(f"ID: {result.vector_id}, Meta: {result.metadata}")
    """
    vector_id: str = Field(description="Unique identifier of the vector")
    vector: List[float] = Field(description="The vector values")
    metadata: Any = Field(description="Metadata associated with the vector")

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True
