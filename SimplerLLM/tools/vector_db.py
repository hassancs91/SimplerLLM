import os
import chromadb
from chromadb.utils import embedding_functions

class VectorDB:
    def __init__(self):
        persistence_directory = "./chroma_db"
        self.client = chromadb.PersistentClient(path=persistence_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name="responses",
            embedding_function=self.embedding_function
        )

    def store_vectors(self, texts):
        self.collection.add(documents=texts, ids=[f"id_{i}" for i in range(len(texts))])

    def query_vectors(self, query_text):
        results = self.collection.query(query_texts=[query_text], n_results=5)
        return results['documents'][0]

    def store_response(self, text):
        self.collection.add(documents=[text], ids=[f"id_{self.collection.count()}"])

    def query_similar(self, query_text):
        return self.query_vectors(query_text)

