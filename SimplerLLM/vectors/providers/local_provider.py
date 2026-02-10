"""
Local Vector Database Provider.

Low-level in-memory vector storage implementation using NumPy arrays
with pickle-based file persistence.

This module provides the core SimplerVectors class that handles all
vector operations for the local provider. For high-level usage, prefer
the LocalVectorDB wrapper or VectorDB.create() factory.

Features:
    - In-memory storage with fast NumPy operations
    - Pickle-based file persistence (.svdb format)
    - Automatic dimension validation
    - Metadata indexing for fast lookups
    - Vector normalization for cosine similarity
    - Optional vector compression

Example:
    >>> from SimplerLLM.vectors.providers import SimplerVectors
    >>>
    >>> db = SimplerVectors(db_folder="./vectors")
    >>> vector_id = db.add_vector([0.1, 0.2, 0.3], {"text": "example"})
    >>> results = db.top_cosine_similarity([0.1, 0.2, 0.3], top_n=5)
"""

import numpy as np
import os
import pickle
import enum
import uuid
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

from ..exceptions import (
    VectorDBError,
    VectorNotFoundError,
    DimensionMismatchError,
    VectorDBOperationError,
)


class SerializationFormat(enum.Enum):
    """
    Serialization formats for saving vector databases to disk.

    Attributes:
        BINARY: Pickle-based binary format (.svdb files)

    Example:
        >>> db.save_to_disk("my_collection", SerializationFormat.BINARY)
    """
    BINARY = 'pickle'


class SimplerVectors:
    """
    In-memory vector database with NumPy-based operations.

    This class provides core vector storage and similarity search functionality
    using NumPy arrays for efficient computation. Supports file persistence
    via pickle serialization.

    Attributes:
        db_folder: Path to the folder for storing database files
        dimension: Vector dimension (set by first vector or explicitly)
        vectors: List of stored vectors as NumPy arrays
        metadata: List of metadata associated with each vector
        ids: List of unique vector IDs

    Example:
        >>> db = SimplerVectors(db_folder="./my_vectors")
        >>>
        >>> # Add vectors
        >>> id1 = db.add_vector([0.1, 0.2, 0.3], {"text": "Hello"})
        >>> id2 = db.add_vector([0.2, 0.3, 0.4], {"text": "World"})
        >>>
        >>> # Search
        >>> results = db.top_cosine_similarity([0.15, 0.25, 0.35], top_n=2)
        >>> for vid, meta, score in results:
        ...     print(f"{meta['text']}: {score:.3f}")
        >>>
        >>> # Save to disk
        >>> db.save_to_disk("my_collection")
    """

    def __init__(self, db_folder: str, dimension: Optional[int] = None):
        """
        Initialize the SimplerVectors database.

        Args:
            db_folder: Path to the folder for storing database files.
                      Will be created if it doesn't exist.
            dimension: Expected vector dimension. If None, will be set
                      automatically by the first vector added.

        Raises:
            VectorDBOperationError: If the folder cannot be created.

        Example:
            >>> # Auto-detect dimension
            >>> db = SimplerVectors(db_folder="./vectors")
            >>>
            >>> # Explicit dimension (validates all vectors)
            >>> db = SimplerVectors(db_folder="./vectors", dimension=1536)
        """
        self.db_folder = db_folder
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Any] = []
        self.ids: List[str] = []
        self.dimension = dimension
        self._index: Dict[str, List[int]] = defaultdict(list)

        try:
            if not os.path.exists(self.db_folder):
                os.makedirs(self.db_folder)
        except Exception as e:
            raise VectorDBOperationError(f"Failed to create database folder: {e}")

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
            >>> db = SimplerVectors(db_folder="./vectors")
            >>> db.load_from_disk("my_collection")
            >>> print(f"Loaded {db.get_vector_count()} vectors")
        """
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if serialization_format == SerializationFormat.BINARY:
            self._load_pickle(file_path)

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
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if serialization_format == SerializationFormat.BINARY:
            self._save_pickle(file_path)

    def _load_pickle(self, file_path: str) -> None:
        """Load database from pickle file."""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                # Handle both old (2-tuple) and new (4-tuple) format
                if len(data) == 2:
                    self.vectors, self.metadata = data
                    self.ids = [str(uuid.uuid4()) for _ in range(len(self.vectors))]
                else:
                    self.vectors, self.metadata, self.ids, self.dimension = data
                self._rebuild_index()
        else:
            self.vectors, self.metadata, self.ids = [], [], []

    def _save_pickle(self, file_path: str) -> None:
        """Save database to pickle file."""
        with open(file_path, 'wb') as file:
            pickle.dump((self.vectors, self.metadata, self.ids, self.dimension), file)

    def _rebuild_index(self) -> None:
        """Rebuild the metadata index after loading from disk."""
        self._index = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        self._index[f"{key}:{value}"].append(i)
            else:
                self._index[str(meta)].append(i)

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            vector: The vector to normalize

        Returns:
            Normalized vector with unit length, or original if zero-length

        Example:
            >>> vec = np.array([3.0, 4.0])
            >>> normalized = SimplerVectors.normalize_vector(vec)
            >>> print(np.linalg.norm(normalized))  # 1.0
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _validate_dimension(self, vector: np.ndarray) -> bool:
        """Validate that vector matches the expected dimension."""
        if self.dimension is None:
            self.dimension = len(vector)
            return True

        if len(vector) != self.dimension:
            raise DimensionMismatchError(
                f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}"
            )
        return True

    def add_vector(
        self,
        vector: Union[np.ndarray, List[float]],
        meta: Any,
        normalize: bool = True,
        id: Optional[str] = None
    ) -> str:
        """
        Add a single vector with metadata to the database.

        Args:
            vector: The vector to add (numpy array or list of floats)
            meta: Metadata to associate with the vector (dict preferred)
            normalize: Whether to normalize the vector to unit length
            id: Optional custom ID (auto-generated UUID if None)

        Returns:
            The ID of the added vector

        Raises:
            DimensionMismatchError: If vector dimension doesn't match
            VectorDBOperationError: If the operation fails

        Example:
            >>> vector_id = db.add_vector(
            ...     vector=[0.1, 0.2, 0.3, 0.4],
            ...     meta={"text": "Hello world", "source": "user"},
            ...     normalize=True
            ... )
            >>> print(f"Added: {vector_id}")
        """
        try:
            vector = np.array(vector, dtype=np.float32)
            self._validate_dimension(vector)

            if normalize:
                vector = self.normalize_vector(vector)

            vector_id = id if id is not None else str(uuid.uuid4())

            self.vectors.append(vector)
            self.metadata.append(meta)
            self.ids.append(vector_id)

            # Update index
            idx = len(self.vectors) - 1
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        self._index[f"{key}:{value}"].append(idx)
            else:
                self._index[str(meta)].append(idx)

            return vector_id
        except DimensionMismatchError:
            raise
        except Exception as e:
            raise VectorDBOperationError(f"Failed to add vector: {e}")

    def add_vectors_batch(
        self,
        vectors_with_meta: List[Tuple],
        normalize: bool = False
    ) -> List[str]:
        """
        Add multiple vectors with metadata in batch.

        Args:
            vectors_with_meta: List of tuples: (vector, metadata) or (vector, metadata, id)
            normalize: Whether to normalize the vectors

        Returns:
            List of IDs for the added vectors

        Example:
            >>> batch = [
            ...     ([0.1, 0.2, 0.3], {"text": "Doc 1"}),
            ...     ([0.2, 0.3, 0.4], {"text": "Doc 2"}),
            ...     ([0.3, 0.4, 0.5], {"text": "Doc 3"}, "custom-id"),
            ... ]
            >>> ids = db.add_vectors_batch(batch, normalize=True)
        """
        added_ids = []
        for item in vectors_with_meta:
            if len(item) == 2:
                vector, meta = item
                id = None
            else:
                vector, meta, id = item

            added_id = self.add_vector(vector, meta, normalize=normalize, id=id)
            added_ids.append(added_id)

        return added_ids

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
        if isinstance(metadata, dict):
            metadata = metadata.copy()
            metadata['text'] = text
        elif metadata is None:
            metadata = {'text': text}
        else:
            metadata = {'text': text, 'original_metadata': metadata}

        return self.add_vector(embedding, metadata, normalize, id)

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
        if vector_id in self.ids:
            idx = self.ids.index(vector_id)
            self.vectors.pop(idx)
            self.metadata.pop(idx)
            self.ids.pop(idx)
            self._rebuild_index()
            return True
        return False

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
            new_vector: New vector values (optional, keeps existing if None)
            new_metadata: New metadata (optional, keeps existing if None)
            normalize: Whether to normalize the new vector

        Returns:
            True if updated successfully, False if not found

        Example:
            >>> # Update just metadata
            >>> db.update_vector(vector_id, new_metadata={"text": "Updated"})
            >>>
            >>> # Update vector and metadata
            >>> db.update_vector(
            ...     vector_id,
            ...     new_vector=[0.5, 0.6, 0.7],
            ...     new_metadata={"text": "New content"}
            ... )
        """
        if vector_id in self.ids:
            idx = self.ids.index(vector_id)

            if new_vector is not None:
                new_vector = np.array(new_vector, dtype=np.float32)
                self._validate_dimension(new_vector)
                if normalize:
                    new_vector = self.normalize_vector(new_vector)
                self.vectors[idx] = new_vector

            if new_metadata is not None:
                self.metadata[idx] = new_metadata
                self._rebuild_index()

            return True
        return False

    def top_cosine_similarity(
        self,
        target_vector: Union[np.ndarray, List[float]],
        top_n: int = 3,
        filter_func: Optional[Callable[[str, Any], bool]] = None
    ) -> List[Tuple[str, Any, float]]:
        """
        Find vectors with highest cosine similarity to the target.

        Args:
            target_vector: The vector to compare against
            top_n: Number of top results to return
            filter_func: Optional function (id, metadata) -> bool to filter results

        Returns:
            List of (id, metadata, similarity_score) tuples, sorted by similarity

        Raises:
            DimensionMismatchError: If target vector dimension doesn't match
            VectorDBOperationError: If the operation fails

        Example:
            >>> results = db.top_cosine_similarity(
            ...     target_vector=[0.1, 0.2, 0.3],
            ...     top_n=5
            ... )
            >>> for vid, meta, score in results:
            ...     print(f"Score: {score:.3f}, Text: {meta.get('text', 'N/A')}")
            >>>
            >>> # With filter
            >>> results = db.top_cosine_similarity(
            ...     target_vector=[0.1, 0.2, 0.3],
            ...     top_n=5,
            ...     filter_func=lambda id, meta: meta.get('source') == 'wikipedia'
            ... )
        """
        try:
            target_vector = np.array(target_vector, dtype=np.float32)
            target_vector = self.normalize_vector(target_vector)

            if self.dimension and len(target_vector) != self.dimension:
                raise DimensionMismatchError(
                    f"Query vector dimension mismatch. Expected {self.dimension}, got {len(target_vector)}"
                )

            if not self.vectors:
                return []

            # Create mask for filtered vectors if filter provided
            mask = None
            if filter_func and self.vectors:
                mask = np.array([
                    filter_func(self.ids[i], self.metadata[i])
                    for i in range(len(self.vectors))
                ])
                if not np.any(mask):
                    return []

            # Calculate cosine similarities
            vectors_array = np.array(self.vectors)
            similarities = np.dot(vectors_array, target_vector)

            # Apply filter if provided
            if mask is not None:
                similarities = np.where(mask, similarities, -1)

            # Get the indices of the top N similar vectors
            top_indices = np.argsort(-similarities)[:top_n]

            return [
                (self.ids[i], self.metadata[i], float(similarities[i]))
                for i in top_indices if similarities[i] > -1
            ]
        except DimensionMismatchError:
            raise
        except Exception as e:
            raise VectorDBOperationError(f"Failed to search vectors: {e}")

    def search_by_text(
        self,
        query_text: str,
        embeddings_llm_instance: Any,
        top_n: int = 3,
        filter_func: Optional[Callable[[str, Any], bool]] = None
    ) -> List[Tuple[str, Any, float]]:
        """
        Search using text query - converts to embedding internally.

        Args:
            query_text: The text query to search for
            embeddings_llm_instance: EmbeddingsLLM instance to generate embeddings
            top_n: Number of top results to return
            filter_func: Optional filter function

        Returns:
            List of (id, metadata, similarity_score) tuples

        Example:
            >>> from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider
            >>> embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)
            >>>
            >>> results = db.search_by_text(
            ...     query_text="What is machine learning?",
            ...     embeddings_llm_instance=embeddings,
            ...     top_n=5
            ... )
        """
        try:
            if not query_text or not query_text.strip():
                raise VectorDBOperationError("Query text cannot be empty")

            query_embedding = embeddings_llm_instance.generate_embeddings(query_text)
            query_embedding = np.array(query_embedding, dtype=np.float32)

            if query_embedding.size == 0:
                raise VectorDBOperationError("Empty embedding returned")

            return self.top_cosine_similarity(query_embedding, top_n, filter_func)
        except (DimensionMismatchError, VectorDBOperationError):
            raise
        except Exception as e:
            raise VectorDBOperationError(f"Failed to search by text: {e}")

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
        """
        matching_indices = set()
        first_key = True

        for key, value in kwargs.items():
            indices = set(self._index.get(f"{key}:{value}", []))
            if first_key:
                matching_indices = indices
                first_key = False
            else:
                matching_indices &= indices

        return [
            (self.ids[i], self.vectors[i], self.metadata[i])
            for i in matching_indices
        ]

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
        """
        if vector_id in self.ids:
            idx = self.ids.index(vector_id)
            return (self.vectors[idx], self.metadata[idx])
        return None

    def list_all_ids(self) -> List[str]:
        """
        Get all vector IDs in the database.

        Returns:
            List of all vector IDs

        Example:
            >>> ids = db.list_all_ids()
            >>> print(f"Database contains {len(ids)} vectors")
        """
        return self.ids.copy()

    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the database.

        Returns:
            Number of vectors

        Example:
            >>> count = db.get_vector_count()
            >>> print(f"Vectors: {count}")
        """
        return len(self.vectors)

    def clear_database(self) -> None:
        """
        Remove all vectors from the database.

        Example:
            >>> db.clear_database()
            >>> print(db.get_vector_count())  # 0
        """
        self.vectors.clear()
        self.metadata.clear()
        self.ids.clear()
        self._index.clear()
        self.dimension = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.

        Returns:
            Dictionary with: total_vectors, dimension, provider,
            size_in_memory_mb, metadata_keys

        Example:
            >>> stats = db.get_stats()
            >>> print(f"Vectors: {stats['total_vectors']}")
            >>> print(f"Dimension: {stats['dimension']}")
            >>> print(f"Memory: {stats['size_in_memory_mb']:.2f} MB")
        """
        try:
            return {
                "total_vectors": len(self.vectors),
                "dimension": self.dimension,
                "provider": "local",
                "size_in_memory_mb": sum(v.nbytes for v in self.vectors) / (1024 * 1024) if self.vectors else 0,
                "metadata_keys": self._get_metadata_keys(),
            }
        except Exception as e:
            raise VectorDBOperationError(f"Failed to get stats: {e}")

    def _get_metadata_keys(self) -> List[str]:
        """Get unique metadata keys across all entries."""
        keys = set()
        for meta in self.metadata:
            if isinstance(meta, dict):
                keys.update(meta.keys())
        return list(keys)

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
        if not self.vectors:
            return 1.0

        original_size = sum(v.nbytes for v in self.vectors)
        dtype = np.float16 if bits == 16 else np.float32

        for i in range(len(self.vectors)):
            self.vectors[i] = np.array(self.vectors[i], dtype=dtype)

        new_size = sum(v.nbytes for v in self.vectors)
        return original_size / new_size if new_size > 0 else 1.0
