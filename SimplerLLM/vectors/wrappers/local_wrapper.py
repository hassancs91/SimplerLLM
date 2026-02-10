"""
Local Vector Database Wrapper - High-level interface for in-memory vectors.

This wrapper provides a unified interface for the local in-memory vector
database with verbose logging support and comprehensive documentation.

Features:
    - In-memory storage with fast NumPy operations
    - Pickle-based file persistence (.svdb format)
    - Automatic dimension validation
    - Metadata indexing for fast lookups
    - Vector normalization for cosine similarity
    - Optional vector compression
    - Verbose logging for debugging

Example:
    >>> from SimplerLLM.vectors import VectorDB, VectorProvider
    >>>
    >>> # Create using factory (recommended)
    >>> db = VectorDB.create(
    ...     provider=VectorProvider.LOCAL,
    ...     db_folder="./vectors",
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
    >>> for vid, meta, score in results:
    ...     print(f"Score: {score:.3f}, Text: {meta.get('text')}")
    >>>
    >>> # Save to disk
    >>> db.save_to_disk("my_collection")
"""

from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import numpy as np

from ..providers.local_provider import SimplerVectors, SerializationFormat
from ..exceptions import (
    VectorDBError,
    VectorNotFoundError,
    DimensionMismatchError,
    VectorDBOperationError,
)
from ..models import VectorSearchResult, VectorStats, VectorOperationResult

try:
    from SimplerLLM.utils.custom_verbose import verbose_print
except ImportError:
    def verbose_print(message, level="info"):
        print(f"[{level.upper()}] {message}")


class LocalVectorDB:
    """
    Local vector database wrapper with in-memory storage.

    This class provides a high-level interface for the local vector
    database, handling parameter formatting and verbose logging while
    delegating operations to the SimplerVectors provider.

    Attributes:
        db_folder: Path to the folder for storing database files
        dimension: Vector dimension (auto-detected or explicit)
        verbose: Whether verbose logging is enabled

    Example:
        >>> from SimplerLLM.vectors import VectorDB, VectorProvider
        >>>
        >>> db = VectorDB.create(
        ...     provider=VectorProvider.LOCAL,
        ...     db_folder="./my_vectors",
        ...     verbose=True
        ... )
        >>>
        >>> # Add a vector
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
    """

    def __init__(
        self,
        db_folder: str = "./vectors",
        dimension: Optional[int] = None,
        verbose: bool = False,
        **config
    ):
        """
        Initialize the LocalVectorDB wrapper.

        Args:
            db_folder: Path to store database files (default: "./vectors")
            dimension: Expected vector dimension (auto-detected if None)
            verbose: Enable verbose logging (default: False)
            **config: Additional configuration (passed to provider)

        Example:
            >>> # Basic usage
            >>> db = LocalVectorDB(db_folder="./vectors")
            >>>
            >>> # With dimension validation
            >>> db = LocalVectorDB(db_folder="./vectors", dimension=1536)
            >>>
            >>> # With verbose logging
            >>> db = LocalVectorDB(db_folder="./vectors", verbose=True)
        """
        self.db_folder = db_folder
        self.dimension = dimension
        self.verbose = verbose
        self._provider = SimplerVectors(db_folder=db_folder, dimension=dimension)

        if self.verbose:
            verbose_print(f"Initialized LocalVectorDB at {db_folder}", "info")

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "local"

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
            meta: Metadata to associate with the vector (dict preferred)
            normalize: Whether to normalize the vector to unit length
            id: Optional custom ID (auto-generated UUID if None)
            full_response: Return VectorOperationResult instead of just ID

        Returns:
            Vector ID (str) or VectorOperationResult if full_response=True

        Raises:
            DimensionMismatchError: If vector dimension doesn't match
            VectorDBOperationError: If the operation fails

        Example:
            >>> # Basic usage
            >>> vector_id = db.add_vector(
            ...     vector=[0.1, 0.2, 0.3, 0.4],
            ...     meta={"text": "Hello world", "source": "user"}
            ... )
            >>>
            >>> # With full response
            >>> result = db.add_vector(
            ...     vector=[0.1, 0.2, 0.3, 0.4],
            ...     meta={"text": "Hello world"},
            ...     full_response=True
            ... )
            >>> if result.success:
            ...     print(f"Added: {result.vector_id}")
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
            ...     ([0.3, 0.4, 0.5], {"text": "Doc 3"}, "custom-id"),
            ... ]
            >>> ids = db.add_vectors_batch(batch, normalize=True)
            >>> print(f"Added {len(ids)} vectors")
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
            >>> from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider
            >>> embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)
            >>>
            >>> text = "Machine learning is a subset of artificial intelligence."
            >>> embedding = embeddings.generate_embeddings(text)
            >>>
            >>> vector_id = db.add_text_with_embedding(
            ...     text=text,
            ...     embedding=embedding,
            ...     metadata={"source": "wikipedia", "category": "AI"}
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
            True if deleted successfully, False if not found

        Example:
            >>> if db.delete_vector(vector_id):
            ...     print("Vector deleted")
            ... else:
            ...     print("Vector not found")
        """
        if self.verbose:
            verbose_print(f"Deleting vector: {vector_id}", "debug")

        result = self._provider.delete_vector(vector_id)

        if self.verbose:
            status = "deleted" if result else "not found"
            verbose_print(f"Vector {vector_id}: {status}", "info")

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

        Example:
            >>> # Update just metadata
            >>> db.update_vector(vector_id, new_metadata={"text": "Updated content"})
            >>>
            >>> # Update vector and metadata
            >>> db.update_vector(
            ...     vector_id,
            ...     new_vector=[0.5, 0.6, 0.7, 0.8],
            ...     new_metadata={"text": "New content"}
            ... )
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
            >>> # Basic search
            >>> results = db.top_cosine_similarity(
            ...     target_vector=[0.1, 0.2, 0.3, 0.4],
            ...     top_n=5
            ... )
            >>> for vid, meta, score in results:
            ...     print(f"Score: {score:.3f}, Text: {meta.get('text')}")
            >>>
            >>> # With filter
            >>> results = db.top_cosine_similarity(
            ...     target_vector=query_vec,
            ...     top_n=5,
            ...     filter_func=lambda id, meta: meta.get('source') == 'wikipedia'
            ... )
            >>>
            >>> # With full response
            >>> results = db.top_cosine_similarity(
            ...     target_vector=query_vec,
            ...     top_n=5,
            ...     full_response=True
            ... )
            >>> for result in results:
            ...     print(f"ID: {result.vector_id}, Score: {result.similarity:.3f}")
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
            >>> from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider
            >>> embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)
            >>>
            >>> results = db.search_by_text(
            ...     query_text="What is machine learning?",
            ...     embeddings_llm_instance=embeddings,
            ...     top_n=5
            ... )
            >>> for vid, meta, score in results:
            ...     print(f"Score: {score:.3f}, Text: {meta.get('text')}")
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
        Query vectors by metadata fields (AND logic).

        Args:
            **kwargs: Key-value pairs to match in metadata

        Returns:
            List of (id, vector, metadata) tuples matching all criteria

        Example:
            >>> # Find all vectors with source="wikipedia"
            >>> results = db.query_by_metadata(source="wikipedia")
            >>>
            >>> # Find vectors matching multiple criteria
            >>> results = db.query_by_metadata(source="wikipedia", category="science")
            >>> for vid, vec, meta in results:
            ...     print(f"ID: {vid}, Meta: {meta}")
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
            ...     print(f"Found: {metadata}")
            ... else:
            ...     print("Not found")
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
            >>> print(f"Vectors: {count}")
        """
        return self._provider.get_vector_count()

    def clear_database(self) -> None:
        """
        Remove all vectors from the database.

        Example:
            >>> db.clear_database()
            >>> print(db.get_vector_count())  # 0
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
            size_in_memory_mb, metadata_keys

        Example:
            >>> stats = db.get_stats()
            >>> print(f"Vectors: {stats['total_vectors']}")
            >>> print(f"Dimension: {stats['dimension']}")
            >>> print(f"Memory: {stats['size_in_memory_mb']:.2f} MB")
            >>>
            >>> # With full response
            >>> stats = db.get_stats(full_response=True)
            >>> print(f"Provider: {stats.provider}")
        """
        stats = self._provider.get_stats()

        if full_response:
            return VectorStats(
                total_vectors=stats["total_vectors"],
                dimension=stats["dimension"],
                provider=stats["provider"],
                size_in_memory_mb=stats.get("size_in_memory_mb"),
                metadata_keys=stats.get("metadata_keys")
            )
        return stats

    def compress_vectors(self, bits: int = 16) -> float:
        """
        Compress vectors to lower precision to save memory.

        Args:
            bits: Target bit precision (16 or 32)

        Returns:
            Compression ratio achieved

        Example:
            >>> original_stats = db.get_stats()
            >>> ratio = db.compress_vectors(bits=16)
            >>> print(f"Compressed {ratio:.1f}x")
        """
        if self.verbose:
            verbose_print(f"Compressing vectors to {bits}-bit precision", "info")

        ratio = self._provider.compress_vectors(bits)

        if self.verbose:
            verbose_print(f"Compression ratio: {ratio:.2f}x", "info")

        return ratio

    def save_to_disk(
        self,
        collection_name: str,
        serialization_format: SerializationFormat = SerializationFormat.BINARY
    ) -> None:
        """
        Save the collection to disk.

        Args:
            collection_name: Name for the saved collection (without extension)
            serialization_format: Format to use for serialization

        Example:
            >>> db.add_vector([0.1, 0.2, 0.3], {"text": "example"})
            >>> db.save_to_disk("my_collection")
        """
        if self.verbose:
            verbose_print(f"Saving to disk: {collection_name}", "info")

        self._provider.save_to_disk(collection_name, serialization_format)

        if self.verbose:
            verbose_print(f"Saved collection: {collection_name}", "info")

    def load_from_disk(
        self,
        collection_name: str,
        serialization_format: SerializationFormat = SerializationFormat.BINARY
    ) -> None:
        """
        Load a collection from disk.

        Args:
            collection_name: Name of the collection (without extension)
            serialization_format: Format used for serialization

        Example:
            >>> db = VectorDB.create(provider=VectorProvider.LOCAL, db_folder="./vectors")
            >>> db.load_from_disk("my_collection")
            >>> print(f"Loaded {db.get_vector_count()} vectors")
        """
        if self.verbose:
            verbose_print(f"Loading from disk: {collection_name}", "info")

        self._provider.load_from_disk(collection_name, serialization_format)

        if self.verbose:
            count = self._provider.get_vector_count()
            verbose_print(f"Loaded {count} vectors from {collection_name}", "info")
