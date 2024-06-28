# import os
# import chromadb
# from chromadb.utils import embedding_functions

# class VectorDB:
#     def __init__(self):
#         persistence_directory = "./chroma_db"
#         self.client = chromadb.PersistentClient(path=persistence_directory)
#         self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
#         self.collection = self.client.get_or_create_collection(
#             name="responses",
#             embedding_function=self.embedding_function
#         )

#     def store_vectors(self, texts):
#         self.collection.add(documents=texts, ids=[f"id_{i}" for i in range(len(texts))])

#     def query_vectors(self, query_text):
#         results = self.collection.query(query_texts=[query_text], n_results=5)
#         return results['documents'][0]

#     def store_response(self, text):
#         self.collection.add(documents=[text], ids=[f"id_{self.collection.count()}"])

#     def query_similar(self, query_text):
#         return self.query_vectors(query_text)

import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

class VectorDB:
    def __init__(self):
        persistence_directory = "./chroma_db"
        self.client = chromadb.PersistentClient(path=persistence_directory)
       
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-ada-002"
        )
       
        # Generate a unique name for the new collection
        collection_name = f"responses_openai_{uuid.uuid4().hex[:8]}"
        
        print(f"Creating new collection: {collection_name}")
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def store_vectors(self, texts):
        # Use a unique identifier for each text
        ids = [f"id_{uuid.uuid4().hex}" for _ in texts]
        self.collection.add(documents=texts, ids=ids)

    def query_vectors(self, query_text):
        results = self.collection.query(query_texts=[query_text], n_results=5)
        return results['documents'][0]

    def store_response(self, text):
        self.collection.add(documents=[text], ids=[f"id_{uuid.uuid4().hex}"])

    def query_similar(self, query_text):
        return self.query_vectors(query_text)