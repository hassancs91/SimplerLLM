"""
Persistence utilities for saving and loading clustering results.

This module provides functions to save and load clustering results to/from JSON files,
enabling reuse of expensive clustering operations without re-running them.

Supports multiple storage backends:
- JSON: Simple, human-readable (good for < 1000 chunks)
- SQLite: Efficient, lazy-loading (recommended for >= 1000 chunks)
"""

import json
import gzip
from pathlib import Path
from typing import Union, Optional, Tuple
from .models import ClusteringResult, ClusterTree, Cluster
from .chunk_store import SQLiteChunkStore, ChunkStore


def save_clustering_result(
    result: ClusteringResult,
    file_path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save a ClusteringResult to a JSON file.

    Args:
        result: The clustering result to save
        file_path: Path where the JSON file will be saved
        indent: Number of spaces for JSON indentation (default: 2)

    Example:
        ```python
        from SimplerLLM.language.llm_clustering import save_clustering_result

        # After clustering
        result = clusterer.cluster(chunks)

        # Save to file
        save_clustering_result(result, "my_clusters.json")
        ```
    """
    file_path = Path(file_path)

    # Convert to JSON-serializable dict
    data = result.model_dump(mode='json')

    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"✓ Clustering result saved to {file_path}")


def load_clustering_result(file_path: Union[str, Path]) -> ClusteringResult:
    """
    Load a ClusteringResult from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        The loaded ClusteringResult

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid data

    Example:
        ```python
        from SimplerLLM.language.llm_clustering import load_clustering_result

        # Load saved clusters
        result = load_clustering_result("my_clusters.json")

        # Use the clusters
        print(f"Loaded {len(result.clusters)} clusters")
        ```
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Clustering result file not found: {file_path}")

    # Read from file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate and convert to Pydantic model
    try:
        result = ClusteringResult.model_validate(data)
        print(f"✓ Clustering result loaded from {file_path}")
        return result
    except Exception as e:
        raise ValueError(f"Invalid clustering result file: {e}")


def save_cluster_tree(
    tree: ClusterTree,
    file_path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save a ClusterTree to a JSON file.

    Args:
        tree: The cluster tree to save
        file_path: Path where the JSON file will be saved
        indent: Number of spaces for JSON indentation (default: 2)

    Example:
        ```python
        from SimplerLLM.language.llm_clustering import save_cluster_tree

        # After clustering with hierarchy
        result = clusterer.cluster(chunks, build_hierarchy=True)

        # Save just the tree
        save_cluster_tree(result.tree, "my_tree.json")
        ```
    """
    file_path = Path(file_path)

    # Convert to JSON-serializable dict
    data = tree.model_dump(mode='json')

    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"✓ Cluster tree saved to {file_path}")


def load_cluster_tree(file_path: Union[str, Path]) -> ClusterTree:
    """
    Load a ClusterTree from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        The loaded ClusterTree

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid data

    Example:
        ```python
        from SimplerLLM.language.llm_clustering import load_cluster_tree
        from SimplerLLM.language.llm_retrieval import LLMRetriever

        # Load saved tree
        tree = load_cluster_tree("my_tree.json")

        # Use with retriever
        retriever = LLMRetriever(router, tree)
        ```
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Cluster tree file not found: {file_path}")

    # Read from file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate and convert to Pydantic model
    try:
        tree = ClusterTree.model_validate(data)
        print(f"✓ Cluster tree loaded from {file_path}")
        return tree
    except Exception as e:
        raise ValueError(f"Invalid cluster tree file: {e}")


def get_clustering_stats(file_path: Union[str, Path]) -> dict:
    """
    Get statistics about a saved clustering result without fully loading it.

    Args:
        file_path: Path to the clustering result JSON file

    Returns:
        Dictionary with statistics (num_clusters, num_chunks, tree_depth, etc.)

    Example:
        ```python
        stats = get_clustering_stats("my_clusters.json")
        print(f"Clusters: {stats['num_clusters']}")
        print(f"Tree depth: {stats['tree_depth']}")
        ```
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {
        "num_clusters": len(data.get("clusters", [])),
        "num_chunks_processed": data.get("total_chunks_processed", 0),
        "llm_calls": data.get("total_llm_calls", 0),
        "has_tree": data.get("tree") is not None,
        "tree_depth": data.get("tree", {}).get("max_depth", 0) if data.get("tree") else 0,
        "tree_total_clusters": data.get("tree", {}).get("total_clusters", 0) if data.get("tree") else 0,
    }

    return stats


def save_clustering_result_optimized(
    result: ClusteringResult,
    base_path: Union[str, Path],
    mode: str = "auto",
    compress: bool = True,
    chunk_threshold: int = 1000
) -> None:
    """
    Save clustering result with optimized storage backend.

    This function intelligently chooses the storage backend based on dataset size:
    - Small datasets (< chunk_threshold): Full JSON (simple, human-readable)
    - Large datasets (>= chunk_threshold): SQLite + compressed structure JSON

    Args:
        result: The clustering result to save
        base_path: Base path for saved files (without extension)
        mode: Storage mode ("auto", "json", or "sqlite")
        compress: Whether to compress JSON files (gzip)
        chunk_threshold: Minimum chunks to trigger SQLite mode (default: 1000)

    Creates:
        For JSON mode:
            - {base_path}.json or {base_path}.json.gz

        For SQLite mode:
            - {base_path}_structure.json.gz (tree and metadata only)
            - {base_path}_chunks.db (SQLite with full chunk text)

    Example:
        ```python
        # Auto-detect based on size
        save_clustering_result_optimized(result, "my_clusters")

        # Force SQLite mode
        save_clustering_result_optimized(result, "my_clusters", mode="sqlite")

        # Saves to: my_clusters_structure.json.gz + my_clusters_chunks.db
        ```
    """
    base_path = Path(base_path)
    num_chunks = result.total_chunks_processed

    # Determine mode
    if mode == "auto":
        mode = "sqlite" if num_chunks >= chunk_threshold else "json"

    if mode == "sqlite":
        # SQLite backend: save structure + chunks separately
        _save_with_sqlite_backend(result, base_path, compress)
    else:
        # JSON backend: save everything in JSON
        if compress:
            file_path = base_path.with_suffix('.json.gz')
            _save_json_compressed(result, file_path)
        else:
            file_path = base_path.with_suffix('.json')
            save_clustering_result(result, file_path, indent=2)


def _save_with_sqlite_backend(
    result: ClusteringResult,
    base_path: Path,
    compress: bool = True
) -> None:
    """Save clustering result using SQLite backend for chunks."""

    # Step 1: Extract all chunks from clusters
    all_chunks = []
    for cluster in result.clusters:
        all_chunks.extend(cluster.chunks)

    # Step 2: Create SQLite chunk store and save chunks
    chunks_db_path = base_path.parent / f"{base_path.name}_chunks.db"
    with SQLiteChunkStore(str(chunks_db_path)) as chunk_store:
        chunk_store.add_chunks(all_chunks)

    print(f"✓ Saved {len(all_chunks)} chunks to {chunks_db_path}")

    # Step 3: Create structure-only version (replace chunks with chunk_ids)
    structure_result = _create_structure_only_result(result)

    # Step 4: Save structure as compressed JSON
    structure_path = base_path.parent / f"{base_path.name}_structure.json"
    if compress:
        structure_path = structure_path.with_suffix('.json.gz')
        _save_json_compressed(structure_result, structure_path)
    else:
        data = structure_result.model_dump(mode='json')
        with open(structure_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved structure to {structure_path}")
    print(f"✓ Total storage: Structure ({_get_file_size(structure_path)}) + Chunks ({_get_file_size(chunks_db_path)})")


def _create_structure_only_result(result: ClusteringResult) -> ClusteringResult:
    """Create a copy of clustering result with chunk_ids instead of full chunks."""

    # Create new clusters with chunk_ids only
    new_clusters = []
    for cluster in result.clusters:
        cluster_copy = cluster.model_copy()
        # Extract chunk IDs and clear full chunks
        cluster_copy.chunk_ids = [chunk.chunk_id for chunk in cluster.chunks]
        cluster_copy.chunks = []  # Clear to save space
        new_clusters.append(cluster_copy)

    # Create new tree with updated clusters
    new_tree = None
    if result.tree:
        tree_copy = result.tree.model_copy()
        # Update clusters_by_id with structure-only clusters
        tree_copy.clusters_by_id = {c.id: c for c in new_clusters}
        new_tree = tree_copy

    # Return new result with structure only
    return ClusteringResult(
        clusters=new_clusters,
        tree=new_tree,
        chunk_to_clusters=result.chunk_to_clusters,
        clustering_config=result.clustering_config,
        tree_config=result.tree_config,
        total_chunks_processed=result.total_chunks_processed,
        total_llm_calls=result.total_llm_calls
    )


def _save_json_compressed(result: ClusteringResult, file_path: Path) -> None:
    """Save clustering result as compressed JSON (gzip)."""
    data = result.model_dump(mode='json')
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Clustering result saved to {file_path} ({_get_file_size(file_path)})")


def _get_file_size(file_path: Path) -> str:
    """Get human-readable file size."""
    if not file_path.exists():
        return "0 B"

    size_bytes = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def load_clustering_result_optimized(
    base_path: Union[str, Path],
    lazy_load: bool = True
) -> Tuple[ClusteringResult, Optional[ChunkStore]]:
    """
    Load clustering result with auto-detection of storage backend.

    Automatically detects whether the result was saved with JSON or SQLite backend
    and loads accordingly.

    Args:
        base_path: Base path used when saving (without extension)
        lazy_load: If True and SQLite backend, returns ChunkStore for lazy loading

    Returns:
        Tuple of (ClusteringResult, Optional[ChunkStore])
        - ChunkStore is provided for SQLite backend when lazy_load=True
        - ChunkStore is None for JSON backend or when lazy_load=False

    Example:
        ```python
        # Load with lazy loading (recommended for large datasets)
        result, chunk_store = load_clustering_result_optimized("my_clusters")

        # Use with retriever
        retriever = LLMRetriever(router, result.tree, chunk_store=chunk_store)

        # Don't forget to close chunk_store when done
        if chunk_store:
            chunk_store.close()
        ```
    """
    base_path = Path(base_path)

    # Check which files exist
    json_path = base_path.with_suffix('.json')
    json_gz_path = base_path.with_suffix('.json.gz')
    structure_path = base_path.parent / f"{base_path.name}_structure.json"
    structure_gz_path = base_path.parent / f"{base_path.name}_structure.json.gz"
    chunks_db_path = base_path.parent / f"{base_path.name}_chunks.db"

    # Detect backend
    if chunks_db_path.exists() and (structure_path.exists() or structure_gz_path.exists()):
        # SQLite backend
        return _load_with_sqlite_backend(base_path, lazy_load)

    elif json_gz_path.exists():
        # Compressed JSON
        result = _load_json_compressed(json_gz_path)
        return result, None

    elif json_path.exists():
        # Uncompressed JSON
        result = load_clustering_result(json_path)
        return result, None

    else:
        raise FileNotFoundError(
            f"No clustering result found at {base_path}. "
            f"Expected one of: {json_path}, {json_gz_path}, or SQLite files."
        )


def _load_with_sqlite_backend(
    base_path: Path,
    lazy_load: bool = True
) -> Tuple[ClusteringResult, Optional[ChunkStore]]:
    """Load clustering result from SQLite backend."""

    # Find structure file
    structure_path = base_path.parent / f"{base_path.name}_structure.json"
    structure_gz_path = base_path.parent / f"{base_path.name}_structure.json.gz"

    if structure_gz_path.exists():
        result = _load_json_compressed(structure_gz_path)
    elif structure_path.exists():
        with open(structure_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = ClusteringResult.model_validate(data)
    else:
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    print(f"✓ Loaded structure from {structure_path if structure_path.exists() else structure_gz_path}")

    # Load chunks
    chunks_db_path = base_path.parent / f"{base_path.name}_chunks.db"

    if lazy_load:
        # Return ChunkStore for lazy loading
        chunk_store = SQLiteChunkStore(str(chunks_db_path), read_only=True)
        print(f"✓ Opened chunks database for lazy loading: {chunks_db_path}")
        return result, chunk_store
    else:
        # Load all chunks into memory
        chunk_store = SQLiteChunkStore(str(chunks_db_path), read_only=True)
        all_chunk_ids = set()
        for cluster in result.clusters:
            all_chunk_ids.update(cluster.chunk_ids)

        chunks = chunk_store.get_chunks(list(all_chunk_ids))
        chunk_store.close()

        # Populate clusters with full chunks
        chunk_dict = {c.chunk_id: c for c in chunks}
        for cluster in result.clusters:
            cluster.chunks = [chunk_dict[cid] for cid in cluster.chunk_ids if cid in chunk_dict]

        print(f"✓ Loaded {len(chunks)} chunks into memory")
        return result, None


def _load_json_compressed(file_path: Path) -> ClusteringResult:
    """Load clustering result from compressed JSON (gzip)."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    result = ClusteringResult.model_validate(data)
    print(f"✓ Clustering result loaded from {file_path}")
    return result
