"""
Qdrant Vector Database Provider.

Low-level Qdrant vector database implementation for self-hosted or
Qdrant Cloud deployments.

This module provides the core QdrantProvider class that handles all
vector operations for the Qdrant provider. For high-level usage, prefer
the QdrantVectorDB wrapper or VectorDB.create() factory.

Features:
    - Self-hosted or Qdrant Cloud support
    - Automatic collection management
    - DOT distance metric (cosine when normalized)
    - Efficient batch operations
    - Metadata filtering with Qdrant's native filters

Requirements:
    pip install qdrant-client

Example:
    >>> from SimplerLLM.vectors.providers import QdrantProvider
    >>>
    >>> db = QdrantProvider(
    ...     url="localhost",
    ...     port=6333,
    ...     collection_name="my_collection"
    ... )
    >>> vector_id = db.add_vector([0.1, 0.2, 0.3], {"text": "example"})
"""

import numpy as np
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

from ..exceptions import (
    VectorDBError,
    VectorNotFoundError,
    DimensionMismatchError,
    VectorDBOperationError,
    VectorDBConnectionError,
)

# Lazy import for qdrant-client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantProvider:
    """
    Qdrant vector database provider for self-hosted or cloud deployments.

    This class provides core vector storage and similarity search functionality
    using Qdrant's vector database. Supports both local Qdrant instances and
    Qdrant Cloud with API key authentication.

    Attributes:
        url: Qdrant server URL
        port: Qdrant server port
        collection_name: Name of the collection
        dimension: Vector dimension
        api_key: API key for Qdrant Cloud (optional)

    Example:
        >>> # Local Qdrant instance
        >>> db = QdrantProvider(
        ...     url="localhost",
        ...     port=6333,
        ...     collection_name="my_vectors"
        ... )
        >>>
        >>> # Qdrant Cloud
        >>> db = QdrantProvider(
        ...     url="https://your-cluster.qdrant.io",
        ...     api_key="your-api-key",
        ...     collection_name="my_vectors"
        ... )
        >>>
        >>> # Add vectors
        >>> id1 = db.add_vector([0.1, 0.2, 0.3], {"text": "Hello"})
        >>>
        >>> # Search
        >>> results = db.top_cosine_similarity([0.1, 0.2, 0.3], top_n=5)
    """

    def __init__(
        self,
        url: str = 'localhost',
        port: int = 6333,
        collection_name: str = 'default_collection',
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Qdrant provider.

        Args:
            url: Qdrant server URL (default: 'localhost')
            port: Qdrant server port (default: 6333)
            collection_name: Name of the collection to use
            dimension: Vector dimension (optional, detected from first vector)
            api_key: API key for Qdrant Cloud authentication

        Raises:
            ImportError: If qdrant-client is not installed
            VectorDBConnectionError: If connection to Qdrant fails

        Example:
            >>> # Local instance
            >>> db = QdrantProvider(url="localhost", port=6333)
            >>>
            >>> # Qdrant Cloud
            >>> db = QdrantProvider(
            ...     url="https://your-cluster.qdrant.io",
            ...     api_key="your-api-key",
            ...     collection_name="production"
            ... )
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required for QdrantProvider. "
                "Install it with: pip install qdrant-client"
            )

        self.url = url
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.api_key = api_key

        try:
            if self.api_key:
                self.client = QdrantClient(
                    url=self.url,
                    port=self.port,
                    api_key=self.api_key,
                    timeout=10
                )
            else:
                self.client = QdrantClient(
                    host=self.url,
                    port=self.port,
                    timeout=10
                )
            self._ensure_collection_exists()
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to connect to Qdrant: {e}")

    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                if self.dimension is None:
                    return  # Will create when we know the dimension
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.DOT)
                )
        except Exception as e:
            print(f"Warning: Could not check/create collection: {e}")

    def _create_collection_if_needed(self, vector_size: int) -> None:
        """Create collection with the given vector size if it doesn't exist."""
        if self.dimension is None:
            self.dimension = vector_size
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.DOT)
                )
            except Exception:
                pass  # Collection might already exist

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
            meta: Metadata to associate with the vector
            normalize: Whether to normalize the vector to unit length
            id: Optional custom ID (auto-generated UUID if None)

        Returns:
            The ID of the added vector

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
        try:
            vector = np.array(vector, dtype=np.float32)

            if self.dimension is not None and len(vector) != self.dimension:
                raise DimensionMismatchError(
                    f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}"
                )

            self._create_collection_if_needed(len(vector))

            if normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

            vector_id = id if id is not None else str(uuid.uuid4())

            payload = meta.copy() if isinstance(meta, dict) else {"metadata": meta}

            point = PointStruct(
                id=vector_id,
                vector=vector.tolist(),
                payload=payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

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
            ... ]
            >>> ids = db.add_vectors_batch(batch, normalize=True)
        """
        points = []
        added_ids = []

        for item in vectors_with_meta:
            if len(item) == 2:
                vector, meta = item
                vector_id = None
            else:
                vector, meta, vector_id = item

            vector = np.array(vector, dtype=np.float32)

            if not points:
                self._create_collection_if_needed(len(vector))

            if normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

            if vector_id is None:
                vector_id = str(uuid.uuid4())

            payload = meta.copy() if isinstance(meta, dict) else {"metadata": meta}

            point = PointStruct(
                id=vector_id,
                vector=vector.tolist(),
                payload=payload
            )

            points.append(point)
            added_ids.append(vector_id)

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

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
            True if operation completed (Qdrant doesn't report if ID existed)

        Raises:
            VectorDBOperationError: If the operation fails

        Example:
            >>> db.delete_vector(vector_id)
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[vector_id]
            )
            return True
        except Exception as e:
            raise VectorDBOperationError(f"Failed to delete vector: {e}")

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
            >>> db.update_vector(vector_id, new_metadata={"text": "Updated"})
        """
        try:
            existing_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True
            )

            if not existing_points:
                return False

            existing_point = existing_points[0]
            vector = existing_point.vector
            payload = existing_point.payload

            if new_vector is not None:
                new_vector = np.array(new_vector, dtype=np.float32)

                if self.dimension is not None and len(new_vector) != self.dimension:
                    raise DimensionMismatchError(
                        f"Vector dimension mismatch. Expected {self.dimension}, got {len(new_vector)}"
                    )

                if normalize:
                    norm = np.linalg.norm(new_vector)
                    if norm > 0:
                        new_vector = new_vector / norm
                vector = new_vector.tolist()

            if new_metadata is not None:
                payload = new_metadata.copy() if isinstance(new_metadata, dict) else {"metadata": new_metadata}

            point = PointStruct(
                id=vector_id,
                vector=vector,
                payload=payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            return True
        except DimensionMismatchError:
            raise
        except Exception as e:
            raise VectorDBOperationError(f"Failed to update vector: {e}")

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
            ...     print(f"Score: {score:.3f}, Text: {meta.get('text')}")
        """
        try:
            target_vector = np.array(target_vector, dtype=np.float32)

            if self.dimension is not None and len(target_vector) != self.dimension:
                raise DimensionMismatchError(
                    f"Query vector dimension mismatch. Expected {self.dimension}, got {len(target_vector)}"
                )

            norm = np.linalg.norm(target_vector)
            if norm > 0:
                target_vector = target_vector / norm

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=target_vector.tolist(),
                limit=top_n * 2 if filter_func else top_n,
                with_payload=True
            )

            results = []
            for point in search_result:
                point_id = str(point.id)
                metadata = point.payload
                similarity = point.score

                if filter_func is None or filter_func(point_id, metadata):
                    results.append((point_id, metadata, similarity))
                    if len(results) >= top_n:
                        break

            return results
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
        Query vectors by metadata fields.

        Args:
            **kwargs: Key-value pairs to match in metadata

        Returns:
            List of (id, vector, metadata) tuples matching all criteria

        Example:
            >>> results = db.query_by_metadata(source="wikipedia")
        """
        try:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in kwargs.items()
            ]

            if not conditions:
                return []

            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=conditions),
                with_payload=True,
                with_vectors=True,
                limit=1000
            )

            results = []
            for point in search_result[0]:
                point_id = str(point.id)
                vector = np.array(point.vector, dtype=np.float32)
                metadata = point.payload
                results.append((point_id, vector, metadata))

            return results
        except Exception as e:
            print(f"Error in metadata query: {e}")
            return []

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
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True
            )

            if points:
                point = points[0]
                vector = np.array(point.vector, dtype=np.float32)
                metadata = point.payload
                return (vector, metadata)

            return None
        except Exception as e:
            print(f"Error retrieving vector: {e}")
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
        try:
            all_ids = []
            offset = None

            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=offset,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False
                )

                points, next_offset = result

                for point in points:
                    all_ids.append(str(point.id))

                if next_offset is None:
                    break
                offset = next_offset

            return all_ids
        except Exception as e:
            print(f"Error listing IDs: {e}")
            return []

    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the database.

        Returns:
            Number of vectors

        Example:
            >>> count = db.get_vector_count()
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            print(f"Error getting vector count: {e}")
            return 0

    def clear_database(self) -> None:
        """
        Remove all vectors from the database.

        Example:
            >>> db.clear_database()
        """
        try:
            self.client.delete_collection(self.collection_name)
            if self.dimension:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.DOT)
                )
        except Exception as e:
            print(f"Error clearing database: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.

        Returns:
            Dictionary with: total_vectors, dimension, provider,
            collection_name, status

        Example:
            >>> stats = db.get_stats()
            >>> print(f"Vectors: {stats['total_vectors']}")
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "total_vectors": info.points_count,
                "dimension": self.dimension,
                "provider": "qdrant",
                "collection_name": self.collection_name,
                "status": str(info.status),
            }
        except Exception as e:
            raise VectorDBOperationError(f"Failed to get stats: {e}")

    def compress_vectors(self, bits: int = 16) -> float:
        """
        Compress vectors (not supported by Qdrant).

        Note:
            Qdrant manages its own storage optimization.

        Returns:
            Always 1.0 (no compression applied)
        """
        print("Vector compression is not supported for Qdrant.")
        return 1.0

    def save_to_disk(self, collection_name: str, serialization_format: Any = None) -> None:
        """
        Save to disk (Qdrant handles persistence automatically).

        Note:
            Qdrant persists data automatically. This method exists for
            API compatibility.
        """
        print("Qdrant handles persistence automatically.")

    def load_from_disk(self, collection_name: str, serialization_format: Any = None) -> None:
        """
        Load from disk (Qdrant loads collections automatically).

        Args:
            collection_name: Name of the collection to switch to

        Note:
            Updates the current collection name and ensures it exists.
        """
        self.collection_name = collection_name
        self._ensure_collection_exists()
