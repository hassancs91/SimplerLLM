from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np
from .vector_providers import VectorProvider


# Custom Exception Classes
class VectorDBError(Exception):
    """Base exception for VectorDB operations."""
    pass


class VectorNotFoundError(VectorDBError):
    """Raised when a vector ID is not found in the database."""
    pass


class DimensionMismatchError(VectorDBError):
    """Raised when vector dimensions don't match the database dimension."""
    pass


class VectorDBConnectionError(VectorDBError):
    """Raised when connection to vector database fails."""
    pass


class VectorDBOperationError(VectorDBError):
    """Raised when a database operation fails."""
    pass


class VectorDB(ABC):
    """
    Abstract base class for unified vector database interface.

    This class defines the contract that all vector database implementations must follow.
    It provides a factory method for creating instances and defines all core operations
    that any vector database provider should support.

    Core Operations:
    - CRUD: add_vector, add_vectors_batch, add_text_with_embedding, delete_vector, update_vector
    - Search: top_cosine_similarity, search_by_text, query_by_metadata
    - Retrieval: get_vector_by_id, list_all_ids, get_vector_count
    - Utility: clear_database, get_stats

    Usage:
        # Create a vector database instance using the factory method
        db = VectorDB.create(provider=VectorProvider.LOCAL, db_folder="./vectors")

        # Add vectors
        vector_id = db.add_vector(vector=[0.1, 0.2, 0.3], meta={"text": "example"})

        # Search
        results = db.top_cosine_similarity(target_vector=[0.1, 0.2, 0.3], top_n=5)
    """

    def __init__(self, provider: VectorProvider, **config):
        """
        Initialize the vector database.

        Args:
            provider: VectorProvider enum indicating the database type
            **config: Provider-specific configuration parameters
        """
        self.provider = provider
        self.config = config

    @staticmethod
    def create(provider: VectorProvider = VectorProvider.LOCAL, **config) -> 'VectorDB':
        """
        Factory method to create a vector database instance.

        Args:
            provider: VectorProvider enum (LOCAL or QDRANT)
            **config: Provider-specific configuration

        Provider-specific config parameters:

        LOCAL:
            - db_folder (str): Path to store database files (required)
            - dimension (int): Vector dimension for validation (optional)

        QDRANT:
            - url (str): Qdrant server URL (default: 'localhost')
            - port (int): Qdrant server port (default: 6333)
            - collection_name (str): Collection name (default: 'default_collection')
            - dimension (int): Vector dimension (optional)
            - api_key (str): API key for Qdrant Cloud (optional)

        Returns:
            VectorDB instance for the specified provider

        Raises:
            ValueError: If provider is not supported
        """
        if provider == VectorProvider.LOCAL:
            from .local_vector_db import LocalVectorDB
            return LocalVectorDB(provider, **config)
        elif provider == VectorProvider.QDRANT:
            from .qdrant_vector_db import QdrantVectorDB
            return QdrantVectorDB(provider, **config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def set_provider(self, provider: VectorProvider):
        """Set the vector database provider."""
        if not isinstance(provider, VectorProvider):
            raise ValueError("Provider must be an instance of VectorProvider Enum")
        self.provider = provider

    # ==================== Core CRUD Operations ====================

    @abstractmethod
    def add_vector(self, vector: np.ndarray, meta: Any, normalize: bool = True, id: Optional[str] = None) -> str:
        """
        Add a single vector to the database.

        Args:
            vector: Numpy array representing the vector
            meta: Metadata associated with the vector (dict preferred)
            normalize: Whether to normalize the vector (default: True)
            id: Optional custom ID for the vector (auto-generated if None)

        Returns:
            str: The vector ID

        Raises:
            DimensionMismatchError: If vector dimension doesn't match database dimension
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def add_vectors_batch(self, vectors_with_meta: List[Tuple], normalize: bool = False) -> List[str]:
        """
        Add multiple vectors to the database in batch.

        Args:
            vectors_with_meta: List of tuples (vector, metadata) or (vector, metadata, id)
            normalize: Whether to normalize the vectors (default: False)

        Returns:
            List[str]: List of vector IDs

        Raises:
            DimensionMismatchError: If any vector dimension doesn't match
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def add_text_with_embedding(self, text: str, embedding: np.ndarray, metadata: Optional[Dict] = None,
                                normalize: bool = True, id: Optional[str] = None) -> str:
        """
        Add a text with its embedding to the database.

        Convenience method for RAG use cases where you want to store both the text
        and its embedding vector.

        Args:
            text: The text content
            embedding: The embedding vector for the text
            metadata: Additional metadata (optional)
            normalize: Whether to normalize the embedding (default: True)
            id: Optional custom ID (auto-generated if None)

        Returns:
            str: The vector ID

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the database.

        Args:
            vector_id: The ID of the vector to delete

        Returns:
            bool: True if deleted successfully, False if vector not found

        Raises:
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def update_vector(self, vector_id: str, new_vector: Optional[np.ndarray] = None,
                     new_metadata: Optional[Any] = None, normalize: bool = True) -> bool:
        """
        Update a vector and/or its metadata.

        Args:
            vector_id: The ID of the vector to update
            new_vector: New vector values (optional, keeps existing if None)
            new_metadata: New metadata (optional, keeps existing if None)
            normalize: Whether to normalize the new vector (default: True)

        Returns:
            bool: True if updated successfully, False if vector not found

        Raises:
            DimensionMismatchError: If new vector dimension doesn't match
            VectorDBOperationError: If the operation fails
        """
        pass

    # ==================== Search Operations ====================

    @abstractmethod
    def top_cosine_similarity(self, target_vector: np.ndarray, top_n: int = 3,
                             filter_func: Optional[Callable] = None) -> List[Tuple[str, Any, float]]:
        """
        Find vectors with highest cosine similarity to the target vector.

        Args:
            target_vector: The vector to compare against
            top_n: Number of top results to return (default: 3)
            filter_func: Optional function to filter results by metadata

        Returns:
            List of tuples (id, metadata, similarity_score) sorted by similarity (highest first)

        Raises:
            DimensionMismatchError: If target vector dimension doesn't match
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def search_by_text(self, query_text: str, embeddings_llm_instance: Any, top_n: int = 3,
                      filter_func: Optional[Callable] = None) -> List[Tuple[str, Any, float]]:
        """
        Search for similar vectors using a text query.

        Args:
            query_text: The text query
            embeddings_llm_instance: LLM instance to generate embeddings from text
            top_n: Number of top results to return (default: 3)
            filter_func: Optional function to filter results by metadata

        Returns:
            List of tuples (id, metadata, similarity_score) sorted by similarity (highest first)

        Raises:
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def query_by_metadata(self, **kwargs) -> List[Tuple[str, np.ndarray, Any]]:
        """
        Query vectors by metadata fields.

        Args:
            **kwargs: Metadata field key-value pairs to match

        Returns:
            List of tuples (id, vector, metadata) matching the query

        Raises:
            VectorDBOperationError: If the operation fails
        """
        pass

    # ==================== Retrieval Operations ====================

    @abstractmethod
    def get_vector_by_id(self, vector_id: str) -> Optional[Tuple[np.ndarray, Any]]:
        """
        Retrieve a vector and its metadata by ID.

        Args:
            vector_id: The ID of the vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None if not found

        Raises:
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def list_all_ids(self) -> List[str]:
        """
        Get a list of all vector IDs in the database.

        Returns:
            List of all vector IDs

        Raises:
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the database.

        Returns:
            int: Number of vectors

        Raises:
            VectorDBOperationError: If the operation fails
        """
        pass

    # ==================== Utility Operations ====================

    @abstractmethod
    def clear_database(self) -> None:
        """
        Clear all vectors from the database.

        Raises:
            VectorDBOperationError: If the operation fails
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.

        All implementations must return these required keys:
        - total_vectors (int): Total number of vectors
        - dimension (int): Vector dimension
        - provider (str): Provider name (e.g., "local", "qdrant")

        Implementations may include additional provider-specific keys.

        Returns:
            Dictionary containing database statistics

        Raises:
            VectorDBOperationError: If the operation fails
        """
        pass


class VectorDBOptional(ABC):
    """
    Optional features that may not be supported by all vector database providers.

    Implementations should inherit from this class if they support these features.
    Default implementations raise NotImplementedError with helpful messages.
    """

    def compress_vectors(self, bits: int = 16) -> float:
        """
        Compress vectors to reduce memory usage (provider-specific feature).

        Args:
            bits: Number of bits for compression (16 or 32)

        Returns:
            float: Compression ratio achieved

        Raises:
            NotImplementedError: If provider doesn't support compression
        """
        raise NotImplementedError(
            f"Vector compression is not supported by {self.provider.value} provider. "
            f"This feature is only available for local vector databases."
        )

    def save_to_disk(self, collection_name: str, serialization_format: Optional[str] = None) -> None:
        """
        Manually save the database to disk (provider-specific feature).

        Args:
            collection_name: Name for the saved collection
            serialization_format: Format to use for serialization

        Raises:
            NotImplementedError: If provider doesn't support manual persistence
        """
        raise NotImplementedError(
            f"Manual save to disk is not supported by {self.provider.value} provider. "
            f"This provider handles persistence automatically."
        )

    def load_from_disk(self, collection_name: str, serialization_format: Optional[str] = None) -> None:
        """
        Load the database from disk (provider-specific feature).

        Args:
            collection_name: Name of the collection to load
            serialization_format: Format used for serialization

        Raises:
            NotImplementedError: If provider doesn't support manual loading
        """
        raise NotImplementedError(
            f"Manual load from disk is not supported by {self.provider.value} provider. "
            f"This provider handles persistence automatically."
        )
