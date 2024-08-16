import numpy as np
import os
import pickle
import json
import enum
from typing import List, Tuple, Dict, Any, Union
from scipy.spatial import cKDTree
import asyncio
from concurrent.futures import ThreadPoolExecutor


class SerializationFormat(enum.Enum):
    BINARY = 'pickle'
    JSON = 'json'

class VectorDatabase:
    def __init__(self, db_name: str, use_semantic_connections: bool = False,  top_k: int = 5):
        self.db_name: str = db_name
        self.vectors: np.ndarray = np.array([])
        self.metadata: List[Dict[str, Any]] = []
        self.index: Union[cKDTree, None] = None
        self.use_semantic_connections: bool = use_semantic_connections
        self.next_id: int = 1  
        self.top_k: int = top_k 

        

    async def load_from_disk(self, file_path: str, serialization_format: SerializationFormat = SerializationFormat.BINARY) -> None:
        try:
            self.db_file = file_path
            if serialization_format == SerializationFormat.BINARY:
                await self.__load_pickle()
            elif serialization_format == SerializationFormat.JSON:
                await self.__load_json()
            self.__build_index()
            # Update next_id based on the highest id in the loaded metadata
            if self.metadata:
                self.next_id = max(meta['id'] for meta in self.metadata) + 1
            else:
                self.next_id = 1
        except Exception as e:
            print(f"Error loading from disk: {e}")

    async def save_to_disk(self, serialization_format: SerializationFormat = SerializationFormat.BINARY) -> None:
        try:
            if serialization_format == SerializationFormat.BINARY:
                self.db_file = f"{self.db_name}.svdb"
                await self.__save_pickle()
            elif serialization_format == SerializationFormat.JSON:
                self.db_file = f"{self.db_name}.json"
                await self.__save_json()
        except Exception as e:
            print(f"Error saving to disk: {e}")

    async def __load_pickle(self) -> None:
        if os.path.exists(self.db_file):
            with ThreadPoolExecutor() as executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self.__load_pickle_sync
                )
                self.vectors, self.metadata = result
        else:
            raise FileNotFoundError(f"File not found: {self.db_file}")

    def __load_pickle_sync(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        with open(self.db_file, 'rb') as file:
            return pickle.load(file)

    async def __save_pickle(self) -> None:
        with ThreadPoolExecutor() as executor:
            await asyncio.get_event_loop().run_in_executor(
                executor,
                self.__save_pickle_sync
            )

    def __save_pickle_sync(self) -> None:
        with open(self.db_file, 'wb') as file:
            pickle.dump((self.vectors, self.metadata), file)

    async def __load_json(self) -> None:
        if os.path.exists(self.db_file):
            with ThreadPoolExecutor() as executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self.__load_json_sync
                )
                self.vectors, self.metadata = result
        else:
            raise FileNotFoundError(f"File not found: {self.db_file}")

    def __load_json_sync(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        with open(self.db_file, 'r') as file:
            data = json.load(file)
            return np.array(data['vectors']), data['metadata']

    async def __save_json(self) -> None:
        with ThreadPoolExecutor() as executor:
            await asyncio.get_event_loop().run_in_executor(
                executor,
                self.__save_json_sync
            )

    def __save_json_sync(self) -> None:
        with open(self.db_file, 'w') as file:
            json.dump({'vectors': self.vectors.tolist(), 'metadata': self.metadata}, file)

    def __build_index(self) -> None:
        if len(self.vectors) > 0:
            self.index = cKDTree(self.vectors)

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


    def add_vector(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any], normalize: bool = True) -> int:
        if normalize:
            embedding = self.normalize_vector(embedding)
        
        if self.vectors.size == 0:
            self.vectors = np.array([embedding])
        else:
            self.vectors = np.vstack([self.vectors, embedding])
        
        record_id = self.next_id
        self.next_id += 1  # Increment the next available ID
        
        record = {
            "id": record_id,
            "text": text,
            "embedding": embedding.tolist(),
            "metadata": metadata,
            "connections": []  # Initialize connections as an empty list
        }
        
        self.metadata.append(record)
        self.__build_index()

        if self.use_semantic_connections:
            self._update_connections(len(self.metadata) - 1)
        
        return record_id

    def add_vectors_batch(self, records: List[Dict[str, Any]], normalize: bool = True) -> None:
        vectors = []
        for record in records:
            vector = np.array(record['embedding'])
            if normalize:
                vector = self.normalize_vector(vector)
            vectors.append(vector)
            metadata = {k: v for k, v in record.items() if k != 'embedding'}
            metadata['id'] = self.next_id
            self.next_id += 1
            metadata['connections'] = []  # Initialize connections as an empty list
            self.metadata.append(metadata)
        
        if self.vectors.size == 0:
            self.vectors = np.array(vectors)
        else:
            self.vectors = np.vstack([self.vectors] + vectors)
        self.__build_index()

        if self.use_semantic_connections:
            for i in range(len(self.metadata) - len(records), len(self.metadata)):
                self._update_connections(i)

    def _update_connections(self, index: int) -> None:
        if len(self.vectors) < 2:  # Need at least 2 vectors to create connections
            return

        vector = self.vectors[index]
        similarities = self.top_cosine_similarity(vector, self.top_k + 1)  # +1 to account for self-similarity
        connections = [(sim[0]['id'], sim[1]) for sim in similarities if sim[0]['id'] != self.metadata[index]['id']]
        self.metadata[index]['connections'] = connections

        # Update connections for other vectors
        for sim in similarities[1:]:  # Skip the first one as it's the vector itself
            other_index = next(i for i, meta in enumerate(self.metadata) if meta['id'] == sim[0]['id'])
            other_connections = self.metadata[other_index]['connections']
            other_connections.append((self.metadata[index]['id'], sim[1]))
            other_connections.sort(key=lambda x: x[1], reverse=True)
            self.metadata[other_index]['connections'] = other_connections[:self.top_k]

    def update_vector(self, id: int, new_record: Dict[str, Any], normalize: bool = True) -> None:
        index = next((i for i, meta in enumerate(self.metadata) if meta['id'] == id), None)
        if index is None:
            raise ValueError(f"No record found with id: {id}")

        new_vector = np.array(new_record['embedding'])
        if normalize:
            new_vector = self.normalize_vector(new_vector)
        self.vectors[index] = new_vector
        new_metadata = {k: v for k, v in new_record.items() if k != 'embedding'}
        new_metadata['id'] = id  # Preserve the original ID
        new_metadata['connections'] = self.metadata[index]['connections']  # Preserve existing connections
        self.metadata[index] = new_metadata
        self.__build_index()

        if self.use_semantic_connections:
            self._update_connections(index)

    def delete_vector(self, id: int) -> None:
        index = next((i for i, meta in enumerate(self.metadata) if meta['id'] == id), None)
        if index is None:
            raise ValueError(f"No record found with id: {id}")

        self.vectors = np.delete(self.vectors, index, axis=0)
        del self.metadata[index]
        self.__build_index()

        if self.use_semantic_connections:
            # Remove connections to the deleted vector
            for meta in self.metadata:
                meta['connections'] = [(conn_id, score) for conn_id, score in meta['connections'] if conn_id != id]

    def get_connected_chunks(self, id: int, depth: int = 1) -> List[Dict[str, Any]]:
        if not self.use_semantic_connections or depth < 1:
            return []

        index = next((i for i, meta in enumerate(self.metadata) if meta['id'] == id), None)
        if index is None:
            return []

        connected_ids = set([conn[0] for conn in self.metadata[index]['connections']])
        result = [self.get_by_id(connected_id) for connected_id in connected_ids]

        if depth > 1:
            for connected_id in connected_ids:
                result.extend(self.get_connected_chunks(connected_id, depth - 1))

        return list({chunk['id']: chunk for chunk in result}.values())  # Remove duplicates

    def get_by_id(self, id: int) -> Dict[str, Any]:
        for meta in self.metadata:
            if meta['id'] == id:
                return meta
        return None

    def top_cosine_similarity(self, target_vector: np.ndarray, top_n: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        try:
            if self.index is None or len(self.vectors) == 0:
                print("The database is empty.")
                return []
            
            target_vector = self.normalize_vector(target_vector)
            
            # Adjust top_n if it's greater than the number of vectors
            top_n = min(top_n, len(self.vectors))
            
            distances, indices = self.index.query(target_vector.reshape(1, -1), k=top_n)
            
            # Ensure distances and indices are 1D
            distances = distances.flatten()
            indices = indices.flatten()
            
            similarities = 1 - distances**2 / 2
            
            return [(self.metadata[i], float(s)) for i, s in zip(indices, similarities)]
        except Exception as e:
            print(f"An error occurred during similarity search: {e}")
            return []

    def semantic_search(self, query_embedding: np.ndarray, depth: int = 1) -> List[Dict[str, Any]]:
        if len(self.vectors) == 0:
            print("The database is empty.")
            return []

        initial_results = self.top_cosine_similarity(query_embedding, self.top_k)
        
        if not self.use_semantic_connections:
            return [result[0] for result in initial_results]

        expanded_results = []
        for result, similarity in initial_results:
            expanded_results.append(result)
            expanded_results.extend(self.get_connected_chunks(result['id'], depth))

        return list({chunk['id']: chunk for chunk in expanded_results}.values())  # Remove duplicates

    def query_by_metadata(self, query: Dict[str, Any]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        results = []
        for i, meta in enumerate(self.metadata):
            if all(self._nested_get(meta, k) == v for k, v in query.items()):
                results.append((self.vectors[i], meta))
        return results

    def _nested_get(self, d: Dict[str, Any], key: str) -> Any:
        keys = key.split('.')
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return None
        return d if d != {} else None

    def get_by_id(self, id: str) -> Dict[str, Any]:
        for meta in self.metadata:
            if meta['id'] == id:
                return meta
        return None

    def enable_semantic_connections(self):
        self.use_semantic_connections = True
        # Rebuild connections for all existing vectors
        for i in range(len(self.metadata)):
            self._update_connections(i)

    def disable_semantic_connections(self):
        self.use_semantic_connections = False
        for meta in self.metadata:
            meta['connections'] = []

    def clear_database(self):
            """
            Clears all records from the database.
            """
            try:
                # Reset the in-memory data structures
                self.vectors = np.array([])
                self.metadata = []
                self.index = None
                self.next_id = 1

                print("Database has been cleared successfully.")
            except Exception as e:
                print(f"Error clearing database: {e}")

    def check_database_records(self) -> int:
        """
        Checks the number of records in the database.
        """
        return len(self.metadata)

    def add_image_embedding(self, image_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        return self.add_vector(image_id, embedding, metadata)

    def get_similar_images(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        return self.top_cosine_similarity(query_embedding, top_k)

    def image_semantic_search(self, query_embedding: np.ndarray, top_k: int = 5, depth: int = 1) -> List[Dict[str, Any]]:
        return self.semantic_search(query_embedding, top_k, depth)


