"""
Chunk storage backends for efficient memory management.

This module provides different storage strategies for text chunks, enabling
efficient handling of large datasets through lazy loading.
"""

import sqlite3
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from .models import ChunkReference


class ChunkStore(ABC):
    """
    Abstract base class for chunk storage backends.

    This interface allows different storage strategies (in-memory, SQLite, etc.)
    while maintaining a consistent API for retrieving chunks.
    """

    @abstractmethod
    def get_chunk(self, chunk_id: int) -> Optional[ChunkReference]:
        """
        Retrieve a single chunk by ID.

        Args:
            chunk_id: Unique identifier for the chunk

        Returns:
            ChunkReference if found, None otherwise
        """
        pass

    @abstractmethod
    def get_chunks(self, chunk_ids: List[int]) -> List[ChunkReference]:
        """
        Retrieve multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of ChunkReferences (may be shorter than input if some not found)
        """
        pass

    @abstractmethod
    def add_chunk(self, chunk: ChunkReference) -> bool:
        """
        Add a single chunk to storage.

        Args:
            chunk: ChunkReference to store

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[ChunkReference]) -> bool:
        """
        Add multiple chunks to storage (batch operation).

        Args:
            chunks: List of ChunkReferences to store

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_chunk_count(self) -> int:
        """Get total number of chunks in storage."""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources (close connections, etc.)."""
        pass


class InMemoryChunkStore(ChunkStore):
    """
    In-memory chunk storage using a dictionary.

    This is the default storage mode, suitable for small to medium datasets
    (< 1000 chunks). All chunks are kept in memory for fast access.

    Example:
        ```python
        store = InMemoryChunkStore()
        store.add_chunk(ChunkReference(chunk_id=0, text="Example"))
        chunk = store.get_chunk(0)
        ```
    """

    def __init__(self):
        """Initialize empty in-memory storage."""
        self._chunks: Dict[int, ChunkReference] = {}

    def get_chunk(self, chunk_id: int) -> Optional[ChunkReference]:
        """Retrieve chunk from memory."""
        return self._chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: List[int]) -> List[ChunkReference]:
        """Retrieve multiple chunks from memory."""
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self._chunks.get(chunk_id)
            if chunk:
                chunks.append(chunk)
        return chunks

    def add_chunk(self, chunk: ChunkReference) -> bool:
        """Add chunk to memory."""
        try:
            self._chunks[chunk.chunk_id] = chunk
            return True
        except Exception:
            return False

    def add_chunks(self, chunks: List[ChunkReference]) -> bool:
        """Add multiple chunks to memory."""
        try:
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
            return True
        except Exception:
            return False

    def get_chunk_count(self) -> int:
        """Get number of chunks in memory."""
        return len(self._chunks)

    def close(self):
        """No cleanup needed for in-memory storage."""
        pass


class SQLiteChunkStore(ChunkStore):
    """
    SQLite-based chunk storage with lazy loading.

    This storage backend uses SQLite for persisting chunks to disk,
    enabling efficient handling of large datasets (1000+ chunks) without
    loading everything into memory.

    Features:
    - Lazy loading of chunks on demand
    - Efficient batch operations
    - Indexed lookups for fast retrieval
    - Minimal memory footprint

    Example:
        ```python
        store = SQLiteChunkStore("chunks.db")
        store.add_chunks(all_chunks)  # Batch insert

        # Later, lazy load specific chunks
        chunk = store.get_chunk(42)

        store.close()
        ```
    """

    def __init__(self, db_path: str, read_only: bool = False):
        """
        Initialize SQLite chunk store.

        Args:
            db_path: Path to SQLite database file
            read_only: If True, opens database in read-only mode
        """
        self.db_path = Path(db_path)

        # Connection string with read-only option if specified
        if read_only and self.db_path.exists():
            uri = f"file:{self.db_path}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True)
        else:
            self.conn = sqlite3.connect(str(self.db_path))

        # Enable WAL mode for better concurrency (if not read-only)
        if not read_only:
            self.conn.execute("PRAGMA journal_mode=WAL")

        # Create schema if needed
        if not read_only:
            self._create_schema()

    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on chunk_id (though it's PRIMARY KEY, explicit for clarity)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)
        """)

        self.conn.commit()

    def get_chunk(self, chunk_id: int) -> Optional[ChunkReference]:
        """
        Lazy-load a single chunk from SQLite.

        Args:
            chunk_id: ID of chunk to retrieve

        Returns:
            ChunkReference if found, None otherwise
        """
        cursor = self.conn.execute(
            "SELECT chunk_id, text, metadata FROM chunks WHERE chunk_id = ?",
            (chunk_id,)
        )
        row = cursor.fetchone()

        if row:
            return ChunkReference(
                chunk_id=row[0],
                text=row[1],
                metadata=json.loads(row[2]) if row[2] else {}
            )
        return None

    def get_chunks(self, chunk_ids: List[int]) -> List[ChunkReference]:
        """
        Lazy-load multiple chunks from SQLite (batch operation).

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of ChunkReferences (in order of chunk_ids)
        """
        if not chunk_ids:
            return []

        # Use parameterized query with IN clause
        placeholders = ','.join('?' * len(chunk_ids))
        query = f"SELECT chunk_id, text, metadata FROM chunks WHERE chunk_id IN ({placeholders})"

        cursor = self.conn.execute(query, chunk_ids)
        rows = cursor.fetchall()

        # Build dictionary for fast lookup
        chunks_dict = {}
        for row in rows:
            chunks_dict[row[0]] = ChunkReference(
                chunk_id=row[0],
                text=row[1],
                metadata=json.loads(row[2]) if row[2] else {}
            )

        # Return chunks in the order of chunk_ids
        return [chunks_dict[cid] for cid in chunk_ids if cid in chunks_dict]

    def add_chunk(self, chunk: ChunkReference) -> bool:
        """
        Add a single chunk to SQLite.

        Args:
            chunk: ChunkReference to store

        Returns:
            True if successful, False otherwise
        """
        try:
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks (chunk_id, text, metadata) VALUES (?, ?, ?)",
                (chunk.chunk_id, chunk.text, metadata_json)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error adding chunk {chunk.chunk_id}: {e}")
            return False

    def add_chunks(self, chunks: List[ChunkReference]) -> bool:
        """
        Add multiple chunks to SQLite (efficient batch operation).

        Args:
            chunks: List of ChunkReferences to store

        Returns:
            True if successful, False otherwise
        """
        try:
            data = [
                (
                    chunk.chunk_id,
                    chunk.text,
                    json.dumps(chunk.metadata) if chunk.metadata else None
                )
                for chunk in chunks
            ]

            self.conn.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, text, metadata) VALUES (?, ?, ?)",
                data
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error adding chunks in batch: {e}")
            return False

    def get_chunk_count(self) -> int:
        """Get total number of chunks in database."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        return cursor.fetchone()[0]

    def close(self):
        """Close SQLite connection and clean up resources."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


def create_chunk_store(
    mode: str = "memory",
    db_path: Optional[str] = None,
    read_only: bool = False
) -> ChunkStore:
    """
    Factory function to create appropriate chunk store.

    Args:
        mode: Storage mode ("memory" or "sqlite")
        db_path: Path to SQLite database (required if mode="sqlite")
        read_only: If True, opens SQLite in read-only mode

    Returns:
        ChunkStore instance

    Raises:
        ValueError: If invalid mode or missing db_path for SQLite

    Example:
        ```python
        # In-memory store
        store = create_chunk_store(mode="memory")

        # SQLite store
        store = create_chunk_store(mode="sqlite", db_path="chunks.db")
        ```
    """
    if mode == "memory":
        return InMemoryChunkStore()
    elif mode == "sqlite":
        if db_path is None:
            raise ValueError("db_path required for SQLite mode")
        return SQLiteChunkStore(db_path, read_only=read_only)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'memory' or 'sqlite'")
