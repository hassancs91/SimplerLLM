from .vector_db import VectorDB, VectorDBOptional
from .simpler_vector import SimplerVectors


class LocalVectorDB(VectorDB, VectorDBOptional):
    def __init__(self, provider, **config):
        super().__init__(provider, **config)
        # Initialize the underlying SimplerVectors instance
        self._vector_db = SimplerVectors(**config)

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
        return self._vector_db.add_vector(vector, meta, normalize, id)

    def add_vectors_batch(self, vectors_with_meta, normalize=False):
        """
        Add multiple vectors with metadata in batch.
        
        Parameters:
            vectors_with_meta (list): List of (vector, metadata, [id]) tuples
            normalize (bool): Whether to normalize the vectors
            
        Returns:
            list: The IDs of the added vectors
        """
        return self._vector_db.add_vectors_batch(vectors_with_meta, normalize)

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
        return self._vector_db.add_text_with_embedding(text, embedding, metadata, normalize, id)

    def delete_vector(self, vector_id):
        """
        Delete a vector by its ID.
        
        Parameters:
            vector_id (str): The ID of the vector to delete
            
        Returns:
            bool: True if successful, False if vector not found
        """
        return self._vector_db.delete_vector(vector_id)

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
        return self._vector_db.update_vector(vector_id, new_vector, new_metadata, normalize)

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
        return self._vector_db.top_cosine_similarity(target_vector, top_n, filter_func)

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
        return self._vector_db.search_by_text(query_text, embeddings_llm_instance, top_n, filter_func)

    def query_by_metadata(self, **kwargs):
        """
        Query vectors by metadata fields.
        
        Parameters:
            **kwargs: Key-value pairs to match in metadata
            
        Returns:
            list: List of (id, vector, metadata) tuples that match the query
        """
        return self._vector_db.query_by_metadata(**kwargs)

    def get_vector_by_id(self, vector_id):
        """
        Retrieve a specific vector and its metadata by ID
        
        Parameters:
            vector_id (str): The ID of the vector to retrieve
            
        Returns:
            tuple: (vector, metadata) if found, None if not found
        """
        return self._vector_db.get_vector_by_id(vector_id)

    def list_all_ids(self):
        """
        Get all vector IDs in the database
        
        Returns:
            list: List of all vector IDs
        """
        return self._vector_db.list_all_ids()

    def get_vector_count(self):
        """
        Get the total number of vectors in the database
        
        Returns:
            int: Number of vectors
        """
        return self._vector_db.get_vector_count()

    def clear_database(self):
        """
        Remove all vectors from the database
        """
        return self._vector_db.clear_database()

    def get_stats(self):
        """
        Get statistics about the vector database.
        
        Returns:
            dict: Dictionary with statistics
        """
        return self._vector_db.get_stats()

    def compress_vectors(self, bits=16):
        """
        Compress vectors to lower precision to save memory.
        
        Parameters:
            bits (int): Target bit precision (16 or 32, default 16 for compression)
            
        Returns:
            float: Compression ratio
        """
        return self._vector_db.compress_vectors(bits)

    def save_to_disk(self, collection_name, serialization_format=None):
        """
        Save the vector database to disk.
        
        Parameters:
            collection_name (str): Name of the collection to save
            serialization_format: Serialization format (optional)
        """
        if serialization_format is None:
            return self._vector_db.save_to_disk(collection_name)
        else:
            return self._vector_db.save_to_disk(collection_name, serialization_format)

    def load_from_disk(self, collection_name, serialization_format=None):
        """
        Load the vector database from disk.
        
        Parameters:
            collection_name (str): Name of the collection to load
            serialization_format: Serialization format (optional)
        """
        if serialization_format is None:
            return self._vector_db.load_from_disk(collection_name)
        else:
            return self._vector_db.load_from_disk(collection_name, serialization_format)
