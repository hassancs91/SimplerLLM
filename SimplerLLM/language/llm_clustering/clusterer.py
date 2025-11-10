"""
Unified API for LLM-based clustering with optional hierarchical tree building.

This module provides a simple interface for clustering text chunks using
LLM-based semantic analysis.
"""

from typing import List, Optional, Dict, Any
from .models import (
    ChunkReference,
    ClusteringConfig,
    TreeConfig,
    ClusteringResult,
    ClusterTree
)
from .flat_clusterer import FlatClusterer
from .tree_builder import TreeBuilder


class LLMClusterer:
    """
    Main clustering interface that combines flat clustering and tree building.

    This class provides a unified API for:
    1. Flat clustering only
    2. Hierarchical clustering (flat + tree building)
    3. Updating existing clusters with new chunks

    Example:
        ```python
        from SimplerLLM.language.llm import LLM, LLMProvider
        from SimplerLLM.language.llm_clustering import LLMClusterer, ChunkReference

        # Initialize
        llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4")
        clusterer = LLMClusterer(llm)

        # Prepare chunks
        chunks = [
            ChunkReference(chunk_id=0, text="Introduction to AI safety..."),
            ChunkReference(chunk_id=1, text="Neural networks are..."),
            # ... more chunks
        ]

        # Cluster with hierarchy
        result = clusterer.cluster(chunks, build_hierarchy=True)

        # Access results
        print(f"Created {len(result.clusters)} clusters")
        if result.tree:
            print(f"Tree depth: {result.tree.max_depth}")
        ```
    """

    def __init__(
        self,
        llm_instance,
        clustering_config: Optional[ClusteringConfig] = None,
        tree_config: Optional[TreeConfig] = None
    ):
        """
        Initialize the clusterer.

        Args:
            llm_instance: LLM instance for generating structured outputs
            clustering_config: Configuration for flat clustering
            tree_config: Configuration for tree building
        """
        self.llm = llm_instance
        self.clustering_config = clustering_config or ClusteringConfig()
        self.tree_config = tree_config or TreeConfig()

        self.flat_clusterer = FlatClusterer(llm_instance, self.clustering_config)
        self.tree_builder = TreeBuilder(llm_instance, self.tree_config)

    def cluster(
        self,
        chunks: List[ChunkReference],
        build_hierarchy: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ClusteringResult:
        """
        Cluster chunks with optional hierarchical tree building.

        Args:
            chunks: List of text chunks to cluster
            build_hierarchy: Whether to build hierarchical tree (default: True)
            metadata: Optional metadata about the source document

        Returns:
            ClusteringResult with clusters and optional tree
        """
        # Phase 1: Flat clustering
        result = self.flat_clusterer.cluster_chunks(chunks, metadata)

        # Phase 2: Build hierarchy if requested
        if build_hierarchy and len(result.clusters) > self.tree_config.max_clusters_per_level:
            tree = self.tree_builder.build_hierarchy(result.clusters)
            result.tree = tree
            result.tree_config = self.tree_config

            # Update total LLM calls
            result.total_llm_calls += self.tree_builder.total_llm_calls

        elif build_hierarchy:
            # Create simple tree even if not strictly needed
            tree = ClusterTree(
                root_cluster_ids=[c.id for c in result.clusters],
                clusters_by_id={c.id: c for c in result.clusters},
                clusters_by_level={0: [c.id for c in result.clusters]},
                total_clusters=len(result.clusters),
                total_chunks=sum(c.chunk_count for c in result.clusters),
                max_depth=0,
                config=self.tree_config
            )
            result.tree = tree
            result.tree_config = self.tree_config

        return result

    def cluster_flat_only(
        self,
        chunks: List[ChunkReference],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ClusteringResult:
        """
        Perform flat clustering only without building hierarchy.

        Args:
            chunks: List of text chunks to cluster
            metadata: Optional metadata about the source document

        Returns:
            ClusteringResult with flat clusters only
        """
        return self.flat_clusterer.cluster_chunks(chunks, metadata)

    def build_tree_from_clusters(
        self,
        flat_clustering_result: ClusteringResult
    ) -> ClusterTree:
        """
        Build hierarchical tree from existing flat clustering result.

        Useful if you want to cluster first, inspect the flat clusters,
        and then build the tree separately.

        Args:
            flat_clustering_result: Result from cluster_flat_only()

        Returns:
            ClusterTree
        """
        return self.tree_builder.build_hierarchy(flat_clustering_result.clusters)

    def update_with_new_chunks(
        self,
        existing_result: ClusteringResult,
        new_chunks: List[ChunkReference],
        rebuild_tree: bool = True
    ) -> ClusteringResult:
        """
        Update existing clusters with new chunks.

        This method adds new chunks to the existing cluster structure,
        matching them against existing clusters or creating new ones.

        Args:
            existing_result: Previous clustering result
            new_chunks: New chunks to add
            rebuild_tree: Whether to rebuild the hierarchy tree

        Returns:
            Updated ClusteringResult

        Note:
            This creates new Cluster objects rather than mutating existing ones.
        """
        # Combine existing chunks from all clusters
        existing_chunk_references = []
        for cluster in existing_result.clusters:
            existing_chunk_references.extend(cluster.chunks)

        # Combine with new chunks
        all_chunks = existing_chunk_references + new_chunks

        # Re-cluster everything
        # Note: This is a simple approach. A more sophisticated approach
        # would incrementally add new chunks to existing clusters.
        return self.cluster(
            chunks=all_chunks,
            build_hierarchy=rebuild_tree
        )

    async def cluster_async(
        self,
        chunks: List[ChunkReference],
        build_hierarchy: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ClusteringResult:
        """
        Async version of cluster().

        Note: Currently calls sync version. Implement async LLM calls
        for true async behavior.

        Args:
            chunks: List of text chunks to cluster
            build_hierarchy: Whether to build hierarchical tree
            metadata: Optional metadata about the source document

        Returns:
            ClusteringResult
        """
        # TODO: Implement true async when LLM supports it
        return self.cluster(chunks, build_hierarchy, metadata)

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Dictionary with clustering and tree configs
        """
        return {
            "clustering_config": self.clustering_config.model_dump(),
            "tree_config": self.tree_config.model_dump()
        }

    def update_config(
        self,
        clustering_config: Optional[ClusteringConfig] = None,
        tree_config: Optional[TreeConfig] = None
    ):
        """
        Update configuration.

        Args:
            clustering_config: New clustering configuration
            tree_config: New tree configuration
        """
        if clustering_config:
            self.clustering_config = clustering_config
            self.flat_clusterer.config = clustering_config

        if tree_config:
            self.tree_config = tree_config
            self.tree_builder.config = tree_config
