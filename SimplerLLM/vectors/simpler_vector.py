import numpy as np
import os
import pickle
import enum
import uuid
from collections import defaultdict
from .vector_db import VectorDB, VectorDBOptional, VectorDBError, DimensionMismatchError, VectorDBOperationError, VectorNotFoundError
from .vector_providers import VectorProvider


class SerializationFormat(enum.Enum):
    BINARY = 'pickle'


class SimplerVectors(VectorDB, VectorDBOptional):
    def __init__(self, db_folder, dimension=None):
        # Initialize base class
        super().__init__(provider=VectorProvider.LOCAL, db_folder=db_folder, dimension=dimension)

        self.db_folder = db_folder
        self.vectors = []  # Initialize the vectors list
        self.metadata = []  # Initialize the metadata list
        self.ids = []  # Store unique IDs for each vector
        self.dimension = dimension  # Store expected dimension for validation
        self._index = defaultdict(list)  # Simple index for metadata lookup

        try:
            if not os.path.exists(self.db_folder):
                os.makedirs(self.db_folder)
        except Exception as e:
            raise VectorDBOperationError(f"Failed to create database folder: {e}")

    def load_from_disk(self, collection_name, serialization_format=SerializationFormat.BINARY):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if serialization_format == SerializationFormat.BINARY:
            self._load_pickle(file_path)

    def save_to_disk(self, collection_name, serialization_format=SerializationFormat.BINARY):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if serialization_format == SerializationFormat.BINARY:
            self._save_pickle(file_path)

    def _load_pickle(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                # Handle both old and new format
                if len(data) == 2:
                    self.vectors, self.metadata = data
                    self.ids = [str(uuid.uuid4()) for _ in range(len(self.vectors))]
                else:
                    self.vectors, self.metadata, self.ids, self.dimension = data
                self._rebuild_index()
        else:
            self.vectors, self.metadata, self.ids = [], [], []

    def _save_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.vectors, self.metadata, self.ids, self.dimension), file)

    def _rebuild_index(self):
        """Rebuild the metadata index after loading from disk"""
        self._index = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            # Create an index by each metadata field
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        self._index[f"{key}:{value}"].append(i)
            else:
                # For simple metadata
                self._index[str(meta)].append(i)

    @staticmethod
    def normalize_vector(vector):
        """
        Normalize a vector to unit length; return the original vector if it is zero-length.

        Parameters:
            vector (array-like): The vector to be normalized.

        Returns:
            array-like: A normalized vector with unit length.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector  # Handle zero-length vector to avoid division by zero
        return vector / norm
    
    def _validate_dimension(self, vector):
        """Validate that vector matches the expected dimension"""
        if self.dimension is None:
            # First vector sets the dimension
            self.dimension = len(vector)
            return True

        if len(vector) != self.dimension:
            raise DimensionMismatchError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")

        return True
    
    def add_vector(self, vector, meta, normalize=True, id=None):
        """
        Add a single vector with metadata to the database.

        Parameters:
            vector (array-like): The vector to add
            meta (any): Metadata to associate with the vector
            normalize (bool): Whether to normalize the vector
            id (str): Optional custom ID, will generate UUID if not provided

        Returns:
            str: The ID of the added vector
        """
        try:
            # Ensure vector is a numpy array
            vector = np.array(vector, dtype=np.float32)

            # Validate dimension
            self._validate_dimension(vector)

            if normalize:
                vector = self.normalize_vector(vector)

            # Generate or use provided ID
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
            raise  # Re-raise dimension errors as-is
        except Exception as e:
            raise VectorDBOperationError(f"Failed to add vector: {e}")

    def add_vectors_batch(self, vectors_with_meta, normalize=False):
        """
        Add multiple vectors with metadata in batch.
        
        Parameters:
            vectors_with_meta (list): List of (vector, metadata, [id]) tuples
            normalize (bool): Whether to normalize the vectors
            
        Returns:
            list: The IDs of the added vectors
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

    def delete_vector(self, vector_id):
        """
        Delete a vector by its ID.
        
        Parameters:
            vector_id (str): The ID of the vector to delete
            
        Returns:
            bool: True if successful, False if vector not found
        """
        if vector_id in self.ids:
            idx = self.ids.index(vector_id)
            
            # Remove the vector, metadata, and ID
            self.vectors.pop(idx)
            self.metadata.pop(idx)
            self.ids.pop(idx)
            
            # Rebuild index as indices have changed
            self._rebuild_index()
            return True
        
        return False
    
    def update_vector(self, vector_id, new_vector=None, new_metadata=None, normalize=True):
        """
        Update a vector or its metadata by ID.
        
        Parameters:
            vector_id (str): The ID of the vector to update
            new_vector (array-like): The new vector (optional)
            new_metadata (any): The new metadata (optional)
            normalize (bool): Whether to normalize the vector
            
        Returns:
            bool: True if successful, False if vector not found
        """
        if vector_id in self.ids:
            idx = self.ids.index(vector_id)
            
            if new_vector is not None:
                # Ensure vector is a numpy array
                new_vector = np.array(new_vector, dtype=np.float32)
                
                # Validate dimension
                self._validate_dimension(new_vector)
                
                if normalize:
                    new_vector = self.normalize_vector(new_vector)
                
                self.vectors[idx] = new_vector
            
            if new_metadata is not None:
                self.metadata[idx] = new_metadata
            
            # Rebuild index if metadata was updated
            if new_metadata is not None:
                self._rebuild_index()
                
            return True
        
        return False

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
            target_vector = self.normalize_vector(target_vector)

            # Validate dimension
            if self.dimension and len(target_vector) != self.dimension:
                raise DimensionMismatchError(f"Query vector dimension mismatch. Expected {self.dimension}, got {len(target_vector)}")

            # Return empty list if no vectors in database
            if not self.vectors:
                return []

            # Create mask for filtered vectors if filter provided
            mask = None
            if filter_func and self.vectors:
                mask = np.array([
                    filter_func(self.ids[i], self.metadata[i])
                    for i in range(len(self.vectors))
                ])

                # Return empty if no vectors match filter
                if not np.any(mask):
                    return []

            # Calculate cosine similarities
            vectors_array = np.array(self.vectors)
            similarities = np.dot(vectors_array, target_vector)

            # Apply filter if provided
            if mask is not None:
                # Set similarities to -1 for filtered-out vectors
                similarities = np.where(mask, similarities, -1)

            # Get the indices of the top N similar vectors
            top_indices = np.argsort(-similarities)[:top_n]

            # Return ID, metadata and similarity for the top N entries
            return [(self.ids[i], self.metadata[i], similarities[i]) for i in top_indices if similarities[i] > -1]
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

            # The embeddings API now returns numpy arrays directly
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
        matching_indices = set()
        first_key = True
        
        for key, value in kwargs.items():
            indices = set(self._index.get(f"{key}:{value}", []))
            
            if first_key:
                matching_indices = indices
                first_key = False
            else:
                # Intersection with previous matches (AND logic)
                matching_indices &= indices
                
        return [(self.ids[i], self.vectors[i], self.metadata[i]) for i in matching_indices]
    
    def get_stats(self):
        """
        Get statistics about the vector database.

        Returns:
            dict: Dictionary with statistics
        """
        try:
            return {
                # Required fields
                "total_vectors": len(self.vectors),
                "dimension": self.dimension,
                "provider": "local",
                # Local-specific fields
                "size_in_memory_mb": sum(v.nbytes for v in self.vectors) / (1024 * 1024) if self.vectors else 0,
                "metadata_keys": self._get_metadata_keys(),
            }
        except Exception as e:
            raise VectorDBOperationError(f"Failed to get stats: {e}")
    
    def _get_metadata_keys(self):
        """Get unique metadata keys across all entries"""
        keys = set()
        for meta in self.metadata:
            if isinstance(meta, dict):
                keys.update(meta.keys())
        return list(keys)
            
    def compress_vectors(self, bits=16):
        """
        Compress vectors to lower precision to save memory.
        
        Parameters:
            bits (int): Target bit precision (16 or 32, default 16 for compression)
            
        Returns:
            float: Compression ratio
        """
        if not self.vectors:
            return 1.0
            
        # Calculate size before compression
        original_size = sum(v.nbytes for v in self.vectors)
        
        # Choose dtype based on bits
        if bits == 16:
            dtype = np.float16
        else:
            dtype = np.float32
            
        # Convert vectors to the lower precision
        for i in range(len(self.vectors)):
            self.vectors[i] = np.array(self.vectors[i], dtype=dtype)
            
        # Calculate size after compression
        new_size = sum(v.nbytes for v in self.vectors)
        
        return original_size / new_size if new_size > 0 else 1.0

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

    def get_vector_by_id(self, vector_id):
        """
        Retrieve a specific vector and its metadata by ID
        
        Parameters:
            vector_id (str): The ID of the vector to retrieve
            
        Returns:
            tuple: (vector, metadata) if found, None if not found
        """
        if vector_id in self.ids:
            idx = self.ids.index(vector_id)
            return (self.vectors[idx], self.metadata[idx])
        return None

    def list_all_ids(self):
        """
        Get all vector IDs in the database
        
        Returns:
            list: List of all vector IDs
        """
        return self.ids.copy()

    def get_vector_count(self):
        """
        Get the total number of vectors in the database
        
        Returns:
            int: Number of vectors
        """
        return len(self.vectors)

    def clear_database(self):
        """
        Remove all vectors from the database
        """
        self.vectors.clear()
        self.metadata.clear()
        self.ids.clear()
        self._index.clear()
        self.dimension = None
