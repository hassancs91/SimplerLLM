"""
Qdrant Vector Database Wrapper - High-level interface for Qdrant.

This wrapper provides a unified interface for Qdrant vector database
with verbose logging support and comprehensive documentation.

Features:
    - Self-hosted or Qdrant Cloud support
    - Automatic collection management
    - DOT distance metric (cosine when normalized)
    - Efficient batch operations
    - Metadata filtering with Qdrant's native filters
    - Verbose logging for debugging

Requirements:
    pip install qdrant-client

Example:
    >>> from SimplerLLM.vectors import VectorDB, VectorProvider
    >>>
    >>> # Create using factory (recommended)
    >>> db = VectorDB.create(
    ...     provider=VectorProvider.QDRANT,
    ...     url="localhost",
    ...     port=6333,
    ...     collection_name="my_collection",
    ...     verbose=True
    ... )
    >>>
    >>> # Add vectors
    >>> vector_id = db.add_vector(
    ...     vector=[0.1, 0.2, 0.3],
    ...     meta={"text": "Example text"}
    ... )
    >>>
    >>> # Search
    >>> results = db.top_cosine_similarity(
    ...     target_vector=[0.1, 0.2, 0.3],
    ...     top_n=5
    ... )
"""

from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import numpy as np

from ..providers.qdrant_provider import QdrantProvider
from ..exceptions import (
    VectorDBError,
    VectorNotFoundError,
    DimensionMismatchError,
    VectorDBOperationError,
    VectorDBConnectionError,
)
from ..models import VectorSearchResult, VectorStats, VectorOperationResult

try:
    from SimplerLLM.utils.custom_verbose import verbose_print
except ImportError:
    def verbose_print(message, level="info"):
        print(f"[{level.upper()}] {message}")


class QdrantVectorDB:
    """
    Qdrant vector database wrapper for self-hosted or cloud deployments.

    This class provides a high-level interface for the Qdrant vector
    database, handling parameter formatting and verbose logging while
    delegating operations to the QdrantProvider.

    Attributes:
        url: Qdrant server URL
        port: Qdrant server port
        collection_name: Name of the collection
        dimension: Vector dimension
        verbose: Whether verbose logging is enabled

    Example:
        >>> from SimplerLLM.vectors import VectorDB, VectorProvider
        >>>
        >>> # Local Qdrant instance
        >>> db = VectorDB.create(
        ...     provider=VectorProvider.QDRANT,
        ...     url="localhost",
        ...     port=6333,
        ...     collection_name="my_vectors"
        ... )
        >>>
        >>> # Qdrant Cloud
        >>> db = VectorDB.create(
        ...     provider=VectorProvider.QDRANT,
        ...     url="https://your-cluster.qdrant.io",
        ...     api_key="your-api-key",
        ...     collection_name="my_vectors"
        ... )
    """

    def __init__(
        self,
        url: str = 'localhost',
        port: int = 6333,
        collection_name: str = 'default_collection',
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
        **config
    ):
        """
        Initialize the QdrantVectorDB wrapper.

        Args:
            url: Qdrant server URL (default: 'localhost')
            port: Qdrant server port (default: 6333)
            collection_name: Name of the collection to use
            dimension: Vector dimension (auto-detected if None)
            api_key: API key for Qdrant Cloud authentication
            verbose: Enable verbose logging (default: False)
            **config: Additional configuration

        Raises:
            ImportError: If qdrant-client is not installed
            VectorDBConnectionError: If connection to Qdrant fails

        Example:
            >>> # Local instance
            >>> db = QdrantVectorDB(url="localhost", port=6333)
            >>>
            >>> # Qdrant Cloud with verbose logging
            >>> db = QdrantVectorDB(
            ...     url="https://your-cluster.qdrant.io",
            ...     api_key="your-api-key",
            ...     collection_name="production",
            ...     verbose=True
            ... )
        """
        self.url = url
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.api_key = api_key
        self.verbose = verbose

        if self.verbose:
            verbose_print(f"Connecting to Qdrant at {url}:{port}", "info")

        self._provider = QdrantProvider(
            url=url,
            port=port,
            collection_name=collection_name,
            dimension=dimension,
            api_key=api_key
        )

        if self.verbose:
            verbose_print(f"Connected to Qdrant, collection: {collection_name}", "info")

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "qdrant"

    def add_vector(
        self,
        vector: Union[np.ndarray, List[float]],
        meta: Any,
        normalize: bool = True,
        id: Optional[str] = None,
        full_response: bool = False
    ) -> Union[str, VectorOperationResult]:
        """
        Add a single vector with metadata to the database.

        Args:
            vector: The vector to add (numpy array or list of floats)
            meta: Metadata to associate with the vector
            normalize: Whether to normalize the vector to unit length
            id: Optional custom ID (auto-generated UUID if None)
            full_response: Return VectorOperationResult instead of just ID

        Returns:
            Vector ID (str) or VectorOperationResult if full_response=True

        Raises:
            DimensionMismatchError: If vector dimension doesn't match
            VectorDBOperationError: If the operation fails

        Note:
            Uses DOT distance metric. When normalize=True, dot product of
            normalized vectors is equivalent to cosine similarity.

        Example:
            >>> vector_id = db.add_vector(
            ...     vector=[0.1, 0.2, 0.3, 0.4],
            ...     meta={"text": "Hello world", "source": "user"}
            ... )
        """
        if self.verbose:
            verbose_print(f"Adding vector with metadata: {meta}", "debug")

        try:
            vector_id = self._provider.add_vector(vector, meta, normalize, id)

            if self.verbose:
                verbose_print(f"Added vector: {vector_id}", "info")

            if full_response:
                return VectorOperationResult(
                    success=True,
                    vector_id=vector_id,
                    operation="add",
                    message="Vector added successfully"
                )
            return vector_id
        except Exception as e:
            if full_response:
                return VectorOperationResult(
                    success=False,
                    operation="add",
                    message=str(e)
                )
            raise

    def add_vectors_batch(
        self,
        vectors_with_meta: List[Tuple],
        normalize: bool = False,
        full_response: bool = False
    ) -> Union[List[str], VectorOperationResult]:
        """
        Add multiple vectors with metadata in batch.

        Args:
            vectors_with_meta: List of tuples: (vector, metadata) or (vector, metadata, id)
            normalize: Whether to normalize the vectors
            full_response: Return VectorOperationResult instead of just IDs

        Returns:
            List of vector IDs or VectorOperationResult if full_response=True

        Example:
            >>> batch = [
            ...     ([0.1, 0.2, 0.3], {"text": "Doc 1"}),
            ...     ([0.2, 0.3, 0.4], {"text": "Doc 2"}),
            ... ]
            >>> ids = db.add_vectors_batch(batch, normalize=True)
        """
        if self.verbose:
            verbose_print(f"Adding batch of {len(vectors_with_meta)} vectors", "info")

        try:
            ids = self._provider.add_vectors_batch(vectors_with_meta, normalize)

            if self.verbose:
                verbose_print(f"Added {len(ids)} vectors in batch", "info")

            if full_response:
                return VectorOperationResult(
                    success=True,
                    operation="batch_add",
                    message=f"Added {len(ids)} vectors",
                    count=len(ids)
                )
            return ids
        except Exception as e:
            if full_response:
                return VectorOperationResult(
                    success=False,
                    operation="batch_add",
                    message=str(e)
                )
            raise

    def add_text_with_embedding(
        self,
        text: str,
        embedding: Union[np.ndarray, List[float]],
        metadata: Optional[Dict] = None,
        normalize: bool = True,
        id: Optional[str] = None
    ) -> str:
        """
        Add text content along with its embedding for RAG use cases.

        Args:
            text: The original text content
            embedding: The embedding vector for the text
            metadata: Additional metadata (text will be added automatically)
            normalize: Whether to normalize the embedding
            id: Optional custom ID

        Returns:
            The ID of the added vector

        Example:
            >>> embedding = embeddings_llm.generate_embeddings("Hello world")
            >>> vector_id = db.add_text_with_embedding(
            ...     text="Hello world",
            ...     embedding=embedding,
            ...     metadata={"source": "user_input"}
            ... )
        """
        if self.verbose:
            verbose_print(f"Adding text with embedding: {text[:50]}...", "debug")

        return self._provider.add_text_with_embedding(text, embedding, metadata, normalize, id)

    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector by its ID.

        Args:
            vector_id: The ID of the vector to delete

        Returns:
            True if operation completed

        Raises:
            VectorDBOperationError: If the operation fails

        Example:
            >>> db.delete_vector(vector_id)
        """
        if self.verbose:
            verbose_print(f"Deleting vector: {vector_id}", "debug")

        result = self._provider.delete_vector(vector_id)

        if self.verbose:
            verbose_print(f"Vector {vector_id}: deleted", "info")

        return result

    def update_vector(
        self,
        vector_id: str,
        new_vector: Optional[Union[np.ndarray, List[float]]] = None,
        new_metadata: Optional[Any] = None,
        normalize: bool = True
    ) -> bool:
        """
        Update a vector and/or its metadata by ID.

        Args:
            vector_id: The ID of the vector to update
            new_vector: New vector values (optional)
            new_metadata: New metadata (optional)
            normalize: Whether to normalize the new vector

        Returns:
            True if updated successfully, False if not found

        Raises:
            DimensionMismatchError: If new vector dimension doesn't match
            VectorDBOperationError: If the operation fails

        Example:
            >>> db.update_vector(vector_id, new_metadata={"text": "Updated content"})
        """
        if self.verbose:
            verbose_print(f"Updating vector: {vector_id}", "debug")

        result = self._provider.update_vector(vector_id, new_vector, new_metadata, normalize)

        if self.verbose:
            status = "updated" if result else "not found"
            verbose_print(f"Vector {vector_id}: {status}", "info")

        return result

    def top_cosine_similarity(
        self,
        target_vector: Union[np.ndarray, List[float]],
        top_n: int = 3,
        filter_func: Optional[Callable[[str, Any], bool]] = None,
        full_response: bool = False
    ) -> Union[List[Tuple[str, Any, float]], List[VectorSearchResult]]:
        """
        Find vectors with highest cosine similarity to the target.

        Args:
            target_vector: The vector to compare against
            top_n: Number of top results to return
            filter_func: Optional function (id, metadata) -> bool to filter results
            full_response: Return VectorSearchResult objects instead of tuples

        Returns:
            List of (id, metadata, similarity) tuples or VectorSearchResult objects

        Raises:
            DimensionMismatchError: If target vector dimension doesn't match
            VectorDBOperationError: If the operation fails

        Example:
            >>> results = db.top_cosine_similarity(
            ...     target_vector=[0.1, 0.2, 0.3, 0.4],
            ...     top_n=5
            ... )
            >>> for vid, meta, score in results:
            ...     print(f"Score: {score:.3f}, Text: {meta.get('text')}")
        """
        if self.verbose:
            verbose_print(f"Searching for top {top_n} similar vectors", "debug")

        results = self._provider.top_cosine_similarity(target_vector, top_n, filter_func)

        if self.verbose:
            verbose_print(f"Found {len(results)} matching vectors", "info")

        if full_response:
            return [
                VectorSearchResult(
                    vector_id=vid,
                    metadata=meta,
                    similarity=score
                )
                for vid, meta, score in results
            ]
        return results

    def search_by_text(
        self,
        query_text: str,
        embeddings_llm_instance: Any,
        top_n: int = 3,
        filter_func: Optional[Callable[[str, Any], bool]] = None,
        full_response: bool = False
    ) -> Union[List[Tuple[str, Any, float]], List[VectorSearchResult]]:
        """
        Search using text query - converts to embedding internally.

        Args:
            query_text: The text query to search for
            embeddings_llm_instance: EmbeddingsLLM instance to generate embeddings
            top_n: Number of top results to return
            filter_func: Optional filter function
            full_response: Return VectorSearchResult objects instead of tuples

        Returns:
            List of (id, metadata, similarity) tuples or VectorSearchResult objects

        Example:
            >>> results = db.search_by_text(
            ...     query_text="What is machine learning?",
            ...     embeddings_llm_instance=embeddings,
            ...     top_n=5
            ... )
        """
        if self.verbose:
            verbose_print(f"Searching by text: {query_text[:50]}...", "debug")

        results = self._provider.search_by_text(query_text, embeddings_llm_instance, top_n, filter_func)

        if self.verbose:
            verbose_print(f"Found {len(results)} matching vectors", "info")

        if full_response:
            return [
                VectorSearchResult(
                    vector_id=vid,
                    metadata=meta,
                    similarity=score
                )
                for vid, meta, score in results
            ]
        return results

    def query_by_metadata(self, **kwargs) -> List[Tuple[str, np.ndarray, Any]]:
        """
        Query vectors by metadata fields.

        Args:
            **kwargs: Key-value pairs to match in metadata

        Returns:
            List of (id, vector, metadata) tuples matching all criteria

        Example:
            >>> results = db.query_by_metadata(source="wikipedia")
        """
        if self.verbose:
            verbose_print(f"Querying by metadata: {kwargs}", "debug")

        results = self._provider.query_by_metadata(**kwargs)

        if self.verbose:
            verbose_print(f"Found {len(results)} matching vectors", "info")

        return results

    def get_vector_by_id(self, vector_id: str) -> Optional[Tuple[np.ndarray, Any]]:
        """
        Retrieve a vector and its metadata by ID.

        Args:
            vector_id: The ID of the vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None if not found

        Example:
            >>> result = db.get_vector_by_id(vector_id)
            >>> if result:
            ...     vector, metadata = result
        """
        return self._provider.get_vector_by_id(vector_id)

    def list_all_ids(self) -> List[str]:
        """
        Get all vector IDs in the database.

        Returns:
            List of all vector IDs

        Example:
            >>> ids = db.list_all_ids()
            >>> print(f"Database contains {len(ids)} vectors")
        """
        return self._provider.list_all_ids()

    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the database.

        Returns:
            Number of vectors

        Example:
            >>> count = db.get_vector_count()
        """
        return self._provider.get_vector_count()

    def clear_database(self) -> None:
        """
        Remove all vectors from the database.

        Example:
            >>> db.clear_database()
        """
        if self.verbose:
            verbose_print("Clearing database", "info")

        self._provider.clear_database()

        if self.verbose:
            verbose_print("Database cleared", "info")

    def get_stats(self, full_response: bool = False) -> Union[Dict[str, Any], VectorStats]:
        """
        Get statistics about the database.

        Args:
            full_response: Return VectorStats object instead of dict

        Returns:
            Dictionary or VectorStats with: total_vectors, dimension, provider,
            collection_name, status

        Example:
            >>> stats = db.get_stats()
            >>> print(f"Vectors: {stats['total_vectors']}")
        """
        stats = self._provider.get_stats()

        if full_response:
            return VectorStats(
                total_vectors=stats["total_vectors"],
                dimension=stats["dimension"],
                provider=stats["provider"],
                collection_name=stats.get("collection_name")
            )
        return stats

    def compress_vectors(self, bits: int = 16) -> float:
        """
        Compress vectors (not supported by Qdrant).

        Note:
            Qdrant manages its own storage optimization.

        Returns:
            Always 1.0 (no compression applied)
        """
        if self.verbose:
            verbose_print("Vector compression not supported for Qdrant", "warning")

        return self._provider.compress_vectors(bits)

    def save_to_disk(self, collection_name: str, serialization_format: Any = None) -> None:
        """
        Save to disk (Qdrant handles persistence automatically).

        Note:
            Qdrant persists data automatically. This method exists for
            API compatibility.
        """
        if self.verbose:
            verbose_print("Qdrant handles persistence automatically", "info")

        self._provider.save_to_disk(collection_name, serialization_format)

    def load_from_disk(self, collection_name: str, serialization_format: Any = None) -> None:
        """
        Load from disk (Qdrant loads collections automatically).

        Args:
            collection_name: Name of the collection to switch to

        Note:
            Updates the current collection name and ensures it exists.
        """
        if self.verbose:
            verbose_print(f"Switching to collection: {collection_name}", "info")

        self._provider.load_from_disk(collection_name, serialization_format)
        self.collection_name = collection_name
