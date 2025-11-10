"""
Pydantic models for LLM-based clustering system.

This module provides data structures for organizing text chunks into
hierarchical clusters using LLM-based semantic analysis.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class ClusterMetadata(BaseModel):
    """Rich metadata describing a cluster's content and characteristics."""

    canonical_name: str = Field(
        ...,
        description="The official, consistent name for this cluster"
    )
    canonical_tags: List[str] = Field(
        default_factory=list,
        description="Standardized tags for categorization"
    )
    canonical_keywords: List[str] = Field(
        default_factory=list,
        description="Key terms and phrases that characterize this cluster"
    )
    description: str = Field(
        default="",
        description="Detailed description of cluster content and themes"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Brief summary of cluster content"
    )
    topic: Optional[str] = Field(
        default=None,
        description="Primary topic or theme"
    )
    synonyms: List[str] = Field(
        default_factory=list,
        description="Alternative names or phrases for this cluster"
    )


class ChunkReference(BaseModel):
    """Reference to a text chunk with its content and metadata."""

    chunk_id: int = Field(..., description="Unique identifier for this chunk")
    text: str = Field(..., description="The actual text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (source, page, timestamp, etc.)"
    )

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v


class ClusterMatch(BaseModel):
    """Represents a match between a chunk and an existing cluster."""

    cluster_id: str = Field(..., description="ID of the matched cluster")
    cluster_name: str = Field(..., description="Canonical name of the cluster")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this match (0.0 to 1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Explanation for why this cluster was selected"
    )


class ChunkMatchingResult(BaseModel):
    """Result of matching a chunk against existing clusters."""

    matches: List[ClusterMatch] = Field(
        default_factory=list,
        description="List of matching clusters (can be multiple)"
    )
    create_new_cluster: bool = Field(
        default=False,
        description="Whether a new cluster should be created"
    )
    new_cluster_metadata: Optional[ClusterMetadata] = Field(
        default=None,
        description="Metadata for new cluster if create_new_cluster is True"
    )
    confidence_threshold_used: float = Field(
        default=0.7,
        description="The confidence threshold that was applied"
    )


class Cluster(BaseModel):
    """Represents a single cluster in the hierarchy."""

    id: str = Field(..., description="Unique cluster identifier")
    level: int = Field(
        default=0,
        ge=0,
        description="Level in hierarchy (0=leaf clusters, higher=parent clusters)"
    )
    metadata: ClusterMetadata = Field(..., description="Cluster metadata")
    chunks: List[ChunkReference] = Field(
        default_factory=list,
        description="Direct chunk members (for in-memory mode)"
    )
    chunk_ids: List[int] = Field(
        default_factory=list,
        description="Chunk IDs for lazy-loading mode (alternative to chunks)"
    )
    child_clusters: List[str] = Field(
        default_factory=list,
        description="IDs of child clusters (for parent clusters)"
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="ID of parent cluster if this is a child"
    )
    chunk_count: int = Field(
        default=0,
        ge=0,
        description="Total number of chunks (including children)"
    )

    def add_chunk(self, chunk: ChunkReference):
        """Add a chunk to this cluster."""
        self.chunks.append(chunk)
        self.chunk_count = len(self.chunks)

    def get_chunks(self, chunk_store=None):
        """
        Get chunks for this cluster.

        Supports both in-memory and lazy-loading modes:
        - If chunk_store is provided and chunk_ids exist: lazy-load from store
        - Otherwise: return in-memory chunks

        Args:
            chunk_store: Optional ChunkStore instance for lazy loading

        Returns:
            List of ChunkReferences
        """
        from .chunk_store import ChunkStore

        if chunk_store and isinstance(chunk_store, ChunkStore) and self.chunk_ids:
            # Lazy-load mode: fetch chunks from store
            return chunk_store.get_chunks(self.chunk_ids)
        else:
            # In-memory mode: return direct chunks
            return self.chunks

    def is_leaf(self) -> bool:
        """Check if this is a leaf cluster (contains chunks)."""
        return len(self.child_clusters) == 0

    def is_parent(self) -> bool:
        """Check if this is a parent cluster (contains sub-clusters)."""
        return len(self.child_clusters) > 0


class ClusteringConfig(BaseModel):
    """Configuration for the clustering process."""

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for cluster assignment"
    )
    max_clusters_per_chunk: int = Field(
        default=3,
        ge=1,
        description="Maximum number of clusters a chunk can belong to"
    )
    max_total_clusters: int = Field(
        default=30,
        ge=1,
        description="Maximum number of total clusters at leaf level"
    )
    batch_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to process per LLM call"
    )
    below_threshold_behavior: Literal["assign_and_create", "force_assign", "create_only"] = Field(
        default="assign_and_create",
        description="What to do when confidence is below threshold"
    )


class TreeConfig(BaseModel):
    """Configuration for hierarchical tree building."""

    max_children_per_parent: int = Field(
        default=10,
        ge=2,
        description="Maximum child clusters per parent cluster"
    )
    max_clusters_per_level: int = Field(
        default=10,
        ge=2,
        description="Max clusters at any level before creating parent level"
    )
    auto_depth: bool = Field(
        default=True,
        description="Automatically determine optimal tree depth"
    )
    max_depth: int = Field(
        default=4,
        ge=2,
        description="Maximum depth of the cluster tree"
    )


class ClusterTree(BaseModel):
    """Hierarchical tree structure of clusters."""

    root_cluster_ids: List[str] = Field(
        default_factory=list,
        description="IDs of top-level clusters"
    )
    clusters_by_id: Dict[str, Cluster] = Field(
        default_factory=dict,
        description="All clusters indexed by ID"
    )
    clusters_by_level: Dict[int, List[str]] = Field(
        default_factory=dict,
        description="Cluster IDs grouped by level"
    )
    total_clusters: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    max_depth: int = Field(default=0, ge=0)
    config: Optional[TreeConfig] = Field(
        default=None,
        description="Configuration used to build this tree"
    )

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get a cluster by ID."""
        return self.clusters_by_id.get(cluster_id)

    def get_clusters_at_level(self, level: int) -> List[Cluster]:
        """Get all clusters at a specific level."""
        cluster_ids = self.clusters_by_level.get(level, [])
        return [self.clusters_by_id[cid] for cid in cluster_ids if cid in self.clusters_by_id]

    def add_cluster(self, cluster: Cluster):
        """Add a cluster to the tree."""
        self.clusters_by_id[cluster.id] = cluster
        if cluster.level not in self.clusters_by_level:
            self.clusters_by_level[cluster.level] = []
        if cluster.id not in self.clusters_by_level[cluster.level]:
            self.clusters_by_level[cluster.level].append(cluster.id)
        self.total_clusters = len(self.clusters_by_id)
        self.max_depth = max(self.max_depth, cluster.level)


class ClusteringResult(BaseModel):
    """Result of the clustering operation."""

    clusters: List[Cluster] = Field(
        default_factory=list,
        description="All clusters created"
    )
    tree: Optional[ClusterTree] = Field(
        default=None,
        description="Hierarchical tree structure if built"
    )
    chunk_to_clusters: Dict[int, List[str]] = Field(
        default_factory=dict,
        description="Mapping from chunk_id to cluster_ids"
    )
    clustering_config: Optional[ClusteringConfig] = Field(default=None)
    tree_config: Optional[TreeConfig] = Field(default=None)
    total_chunks_processed: int = Field(default=0, ge=0)
    total_llm_calls: int = Field(default=0, ge=0)

    def get_clusters_for_chunk(self, chunk_id: int) -> List[Cluster]:
        """Get all clusters containing a specific chunk."""
        cluster_ids = self.chunk_to_clusters.get(chunk_id, [])
        return [c for c in self.clusters if c.id in cluster_ids]
