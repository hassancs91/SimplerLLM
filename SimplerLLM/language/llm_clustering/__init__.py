"""
LLM-based clustering module for SimplerLLM.

This module provides intelligent text clustering using LLM semantic understanding
rather than traditional embedding-based approaches. It supports both flat clustering
and hierarchical tree structures.

Key Features:
- Incremental cluster matching for consistency
- Multi-cluster assignment support
- Automatic hierarchical tree building
- Rich metadata generation for each cluster
- Configurable confidence thresholds and parameters

Example Usage:
    ```python
    from SimplerLLM.language.llm import LLM, LLMProvider
    from SimplerLLM.language.llm_clustering import (
        LLMClusterer,
        ChunkReference,
        ClusteringConfig,
        TreeConfig
    )

    # Initialize LLM and clusterer
    llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4")
    clusterer = LLMClusterer(llm)

    # Prepare chunks
    chunks = [
        ChunkReference(chunk_id=i, text=chunk_text)
        for i, chunk_text in enumerate(document_chunks)
    ]

    # Cluster with automatic hierarchy
    result = clusterer.cluster(chunks, build_hierarchy=True)

    # Access results
    print(f"Total clusters: {len(result.clusters)}")
    print(f"Tree depth: {result.tree.max_depth if result.tree else 0}")

    # Get clusters for a specific chunk
    chunk_clusters = result.get_clusters_for_chunk(chunk_id=5)
    ```
"""

from .models import (
    # Core data models
    Cluster,
    ClusterMetadata,
    ChunkReference,
    ClusterMatch,
    ChunkMatchingResult,
    ClusterTree,
    ClusteringResult,

    # Configuration
    ClusteringConfig,
    TreeConfig,
)

from .clusterer import LLMClusterer
from .flat_clusterer import FlatClusterer
from .tree_builder import TreeBuilder
from .persistence import (
    save_clustering_result,
    load_clustering_result,
    save_cluster_tree,
    load_cluster_tree,
    get_clustering_stats,
    save_clustering_result_optimized,
    load_clustering_result_optimized
)
from .chunk_store import (
    ChunkStore,
    InMemoryChunkStore,
    SQLiteChunkStore,
    create_chunk_store
)

__all__ = [
    # Main API
    "LLMClusterer",

    # Advanced APIs
    "FlatClusterer",
    "TreeBuilder",

    # Data models
    "Cluster",
    "ClusterMetadata",
    "ChunkReference",
    "ClusterMatch",
    "ChunkMatchingResult",
    "ClusterTree",
    "ClusteringResult",

    # Configuration
    "ClusteringConfig",
    "TreeConfig",

    # Persistence
    "save_clustering_result",
    "load_clustering_result",
    "save_cluster_tree",
    "load_cluster_tree",
    "get_clustering_stats",
    "save_clustering_result_optimized",
    "load_clustering_result_optimized",

    # Chunk Storage
    "ChunkStore",
    "InMemoryChunkStore",
    "SQLiteChunkStore",
    "create_chunk_store",
]

__version__ = "0.1.0"
