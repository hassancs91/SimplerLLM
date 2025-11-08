import numpy as np
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple
from .vector_db import VectorDB, VectorDBOptional, VectorDBError, DimensionMismatchError, VectorDBOperationError, VectorDBConnectionError, VectorNotFoundError

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantVectorDB(VectorDB, VectorDBOptional):
    def __init__(self, provider, **config):
        super().__init__(provider, **config)

        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for QdrantVectorDB. Install it with: pip install qdrant-client")

        # Extract configuration
        self.url = config.get('url', 'localhost')
        self.port = config.get('port', 6333)
        self.collection_name = config.get('collection_name', 'default_collection')
        self.dimension = config.get('dimension', None)
        self.api_key = config.get('api_key', None)

        # Initialize Qdrant client
        try:
            if self.api_key:
                # When using API key with URL, include port and timeout
                self.client = QdrantClient(
                    url=self.url,
                    port=self.port,
                    api_key=self.api_key,
                    timeout=10
                )
            else:
                self.client = QdrantClient(host=self.url, port=self.port, timeout=10)

            # Create collection if it doesn't exist
            self._ensure_collection_exists()
        except Exception as e:
            raise VectorDBConnectionError(f"Failed to connect to Qdrant: {e}")

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                if self.dimension is None:
                    # We'll create the collection when we know the dimension
                    return
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.DOT)
                )
        except Exception as e:
            print(f"Warning: Could not check/create collection: {e}")

    def _create_collection_if_needed(self, vector_size):
        """Create collection with the given vector size if it doesn't exist"""
        if self.dimension is None:
            self.dimension = vector_size
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.DOT)
                )
            except Exception as e:
                # Collection might already exist, that's okay
                pass

    def add_vector(self, vector, meta, normalize=True, id=None):
        """
        Add a single vector with metadata to the database.

        Parameters:
            vector (array-like): The vector to add
            meta (any): Metadata to associate with the vector
            normalize (bool): Whether to normalize the vector
                - True: Normalizes to unit length, uses cosine similarity for search
                - False: Keeps original magnitude, uses dot product for search
            id (str): Optional custom ID, will generate UUID if not provided

        Returns:
            str: The ID of the added vector

        Note:
            Uses DOT distance metric. When normalize=True, dot product of normalized
            vectors is equivalent to cosine similarity. When normalize=False, vector
            magnitudes are preserved for use cases where scale matters.
        """
        try:
            # Ensure vector is a numpy array
            vector = np.array(vector, dtype=np.float32)

            # Validate dimension
            if self.dimension is not None and len(vector) != self.dimension:
                raise DimensionMismatchError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")

            # Create collection if needed
            self._create_collection_if_needed(len(vector))

            if normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

            # Generate or use provided ID
            vector_id = id if id is not None else str(uuid.uuid4())

            # Prepare metadata for Qdrant
            payload = {}
            if isinstance(meta, dict):
                payload = meta.copy()
            else:
                payload = {"metadata": meta}

            # Create point
            point = PointStruct(
                id=vector_id,
                vector=vector.tolist(),
                payload=payload
            )

            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            return vector_id
        except DimensionMismatchError:
            raise  # Re-raise dimension errors as-is
        except Exception as e:
            raise VectorDBOperationError(f"Failed to add vector: {e}")

    def add_vectors_batch(self, vectors_with_meta, normalize=False):
        """
        Add multiple vectors with metadata in batch.

        Parameters:
            vectors_with_meta (list): List of (vector, metadata, [id]) tuples
            normalize (bool): Whether to normalize the vectors
                - True: Normalizes to unit length, uses cosine similarity for search
                - False: Keeps original magnitude, uses dot product for search

        Returns:
            list: The IDs of the added vectors

        Note:
            Uses DOT distance metric. When normalize=True, enables cosine similarity.
            When normalize=False, preserves vector magnitudes for dot product search.
        """
        points = []
        added_ids = []
        
        for item in vectors_with_meta:
            if len(item) == 2:
                vector, meta = item
                vector_id = None
            else:
                vector, meta, vector_id = item
            
            # Ensure vector is a numpy array
            vector = np.array(vector, dtype=np.float32)
            
            # Create collection if needed (using first vector)
            if not points:
                self._create_collection_if_needed(len(vector))
            
            if normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            
            # Generate or use provided ID
            if vector_id is None:
                vector_id = str(uuid.uuid4())
            
            # Prepare metadata for Qdrant
            payload = {}
            if isinstance(meta, dict):
                payload = meta.copy()
            else:
                payload = {"metadata": meta}
            
            # Create point
            point = PointStruct(
                id=vector_id,
                vector=vector.tolist(),
                payload=payload
            )
            
            points.append(point)
            added_ids.append(vector_id)
        
        # Batch insert into Qdrant
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        return added_ids

    def add_text_with_embedding(self, text, embedding, metadata=None, normalize=True, id=None):
        """
        Add text content along with its embedding for easy retrieval
        
        Parameters:
            text (str): The original text content
            embedding (array-like): The embedding vector
            metadata (any): Additional metadata (will include text if dict)
            normalize (bool): Whether to normalize the vector
            id (str): Optional custom ID
            
        Returns:
            str: The ID of the added vector
        """
        # If metadata is a dict, add the text to it
        if isinstance(metadata, dict):
            metadata = metadata.copy()
            metadata['text'] = text
        elif metadata is None:
            metadata = {'text': text}
        else:
            # If metadata is not a dict, create a new dict with both
            metadata = {'text': text, 'original_metadata': metadata}
        
        return self.add_vector(embedding, metadata, normalize, id)

    def delete_vector(self, vector_id):
        """
        Delete a vector by its ID.

        Parameters:
            vector_id (str): The ID of the vector to delete

        Returns:
            bool: True if successful, False if vector not found
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[vector_id]
            )
            return True
        except Exception as e:
            raise VectorDBOperationError(f"Failed to delete vector: {e}")

    def update_vector(self, vector_id, new_vector=None, new_metadata=None, normalize=True):
        """
        Update a vector or its metadata by ID.

        Parameters:
            vector_id (str): The ID of the vector to update
            new_vector (array-like): The new vector (optional)
            new_metadata (any): The new metadata (optional)
            normalize (bool): Whether to normalize the new vector
                - True: Normalizes to unit length for cosine similarity
                - False: Keeps original magnitude for dot product

        Returns:
            bool: True if successful, False if vector not found

        Note:
            When updating vector data, ensure normalization matches the original
            storage strategy for consistent search results.
        """
        try:
            # Get existing point
            existing_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True
            )

            if not existing_points:
                return False

            existing_point = existing_points[0]

            # Prepare updated point
            vector = existing_point.vector
            payload = existing_point.payload

            if new_vector is not None:
                new_vector = np.array(new_vector, dtype=np.float32)

                # Validate dimension
                if self.dimension is not None and len(new_vector) != self.dimension:
                    raise DimensionMismatchError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(new_vector)}")

                if normalize:
                    norm = np.linalg.norm(new_vector)
                    if norm > 0:
                        new_vector = new_vector / norm
                vector = new_vector.tolist()

            if new_metadata is not None:
                if isinstance(new_metadata, dict):
                    payload = new_metadata.copy()
                else:
                    payload = {"metadata": new_metadata}

            # Update point
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
            raise  # Re-raise dimension errors as-is
        except Exception as e:
            raise VectorDBOperationError(f"Failed to update vector: {e}")

    def top_cosine_similarity(self, target_vector, top_n=3, filter_func=None):
        """
        Calculate the cosine similarity and return the top N most similar vectors.

        Parameters:
            target_vector (array-like): The vector to compare against the database
            top_n (int): The number of top results to return
            filter_func (callable): Optional filter function that takes (id, metadata) and returns bool

        Returns:
            list: Tuples of (id, metadata, similarity) for the top N most similar vectors
        """
        try:
            # Ensure vector is a numpy array and normalized
            target_vector = np.array(target_vector, dtype=np.float32)

            # Validate dimension
            if self.dimension is not None and len(target_vector) != self.dimension:
                raise DimensionMismatchError(f"Query vector dimension mismatch. Expected {self.dimension}, got {len(target_vector)}")

            norm = np.linalg.norm(target_vector)
            if norm > 0:
                target_vector = target_vector / norm

            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=target_vector.tolist(),
                limit=top_n * 2 if filter_func else top_n,  # Get more if we need to filter
                with_payload=True
            )

            results = []
            for point in search_result:
                point_id = str(point.id)
                metadata = point.payload
                similarity = point.score

                # Apply filter if provided
                if filter_func is None or filter_func(point_id, metadata):
                    results.append((point_id, metadata, similarity))

                    # Stop if we have enough results
                    if len(results) >= top_n:
                        break

            return results
        except DimensionMismatchError:
            raise  # Re-raise dimension errors as-is
        except Exception as e:
            raise VectorDBOperationError(f"Failed to search vectors: {e}")

    def search_by_text(self, query_text, embeddings_llm_instance, top_n=3, filter_func=None):
        """
        Search using text query - converts to embedding internally

        Parameters:
            query_text (str): The text query to search for
            embeddings_llm_instance: Instance of EmbeddingsLLM to generate embeddings
            top_n (int): Number of top results to return
            filter_func (callable): Optional filter function that takes (id, metadata) and returns bool

        Returns:
            list: Tuples of (id, metadata, similarity) for the top N most similar vectors
        """
        try:
            if not query_text or not query_text.strip():
                raise VectorDBOperationError("Query text cannot be empty")

            # Generate embedding for the query text
            query_embedding = embeddings_llm_instance.generate_embeddings(query_text)

            # Convert to numpy array to ensure consistent format
            query_embedding = np.array(query_embedding, dtype=np.float32)

            # Validate that we have a valid embedding
            if query_embedding.size == 0:
                raise VectorDBOperationError("Empty embedding returned from embeddings_llm_instance")

            return self.top_cosine_similarity(query_embedding, top_n, filter_func)
        except (DimensionMismatchError, VectorDBOperationError):
            raise  # Re-raise VectorDB errors as-is
        except Exception as e:
            raise VectorDBOperationError(f"Failed to search by text: {e}")

    def query_by_metadata(self, **kwargs):
        """
        Query vectors by metadata fields.
        
        Parameters:
            **kwargs: Key-value pairs to match in metadata
            
        Returns:
            list: List of (id, vector, metadata) tuples that match the query
        """
        try:
            # Build Qdrant filter
            conditions = []
            for key, value in kwargs.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            
            if not conditions:
                return []
            
            # Search with filter
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=conditions),
                with_payload=True,
                with_vectors=True,
                limit=1000  # Adjust as needed
            )
            
            results = []
            for point in search_result[0]:  # scroll returns (points, next_page_offset)
                point_id = str(point.id)
                vector = np.array(point.vector, dtype=np.float32)
                metadata = point.payload
                results.append((point_id, vector, metadata))
            
            return results
        except Exception as e:
            print(f"Error in metadata query: {e}")
            return []

    def get_vector_by_id(self, vector_id):
        """
        Retrieve a specific vector and its metadata by ID
        
        Parameters:
            vector_id (str): The ID of the vector to retrieve
            
        Returns:
            tuple: (vector, metadata) if found, None if not found
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

    def list_all_ids(self):
        """
        Get all vector IDs in the database
        
        Returns:
            list: List of all vector IDs
        """
        try:
            # Scroll through all points to get IDs
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

    def get_vector_count(self):
        """
        Get the total number of vectors in the database
        
        Returns:
            int: Number of vectors
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            print(f"Error getting vector count: {e}")
            return 0

    def clear_database(self):
        """
        Remove all vectors from the database
        """
        try:
            self.client.delete_collection(self.collection_name)
            # Recreate empty collection
            if self.dimension:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.DOT)
                )
        except Exception as e:
            print(f"Error clearing database: {e}")

    def get_stats(self):
        """
        Get statistics about the vector database.

        Returns:
            dict: Dictionary with statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                # Required fields
                "total_vectors": info.points_count,
                "dimension": self.dimension,
                "provider": "qdrant",
                # Qdrant-specific fields
                "collection_name": self.collection_name,
                "status": info.status,
            }
        except Exception as e:
            raise VectorDBOperationError(f"Failed to get stats: {e}")

    def compress_vectors(self, bits=16):
        """
        Compress vectors to lower precision to save memory.
        Note: This is not directly supported by Qdrant, returning 1.0
        
        Parameters:
            bits (int): Target bit precision (16 or 32, default 16 for compression)
            
        Returns:
            float: Compression ratio (always 1.0 for Qdrant)
        """
        print("Vector compression is not supported for Qdrant. Vectors are stored in Qdrant's optimized format.")
        return 1.0

    def save_to_disk(self, collection_name, serialization_format=None):
        """
        Save the vector database to disk.
        Note: Qdrant handles persistence automatically
        
        Parameters:
            collection_name (str): Name of the collection to save
            serialization_format: Serialization format (ignored for Qdrant)
        """
        print("Qdrant handles persistence automatically. No manual save required.")

    def load_from_disk(self, collection_name, serialization_format=None):
        """
        Load the vector database from disk.
        Note: Qdrant loads collections automatically
        
        Parameters:
            collection_name (str): Name of the collection to load
            serialization_format: Serialization format (ignored for Qdrant)
        """
        # Update collection name and ensure it exists
        self.collection_name = collection_name
        self._ensure_collection_exists()
