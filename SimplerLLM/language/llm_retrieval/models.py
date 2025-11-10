"""
Pydantic models for LLM-based hierarchical retrieval.

This module provides data structures for navigating cluster trees
and retrieving relevant chunks using LLMRouter.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NavigationStep(BaseModel):
    """Represents a single step in the tree navigation path."""

    level: int = Field(..., ge=0, description="Level in the tree (0=root)")
    cluster_name: str = Field(..., description="Name of selected cluster")
    cluster_id: str = Field(..., description="ID of selected cluster")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this selection"
    )
    reasoning: str = Field(
        default="",
        description="LLMRouter's reasoning for selecting this cluster"
    )
    alternatives_considered: int = Field(
        default=0,
        ge=0,
        description="Number of alternative clusters considered at this level"
    )


class RetrievalResult(BaseModel):
    """A single retrieved chunk with full context."""

    chunk_id: int = Field(..., description="ID of the retrieved chunk")
    chunk_text: str = Field(..., description="The actual chunk content")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this chunk's relevance"
    )
    reasoning: str = Field(
        default="",
        description="Why this chunk was selected"
    )
    cluster_path: List[str] = Field(
        default_factory=list,
        description="Path of cluster names from root to leaf"
    )
    cluster_id_path: List[str] = Field(
        default_factory=list,
        description="Path of cluster IDs from root to leaf"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original chunk metadata"
    )
    rank: int = Field(
        default=0,
        ge=0,
        description="Rank among retrieved results (1=best)"
    )


class HierarchicalRetrievalResponse(BaseModel):
    """Complete response from hierarchical retrieval."""

    query: str = Field(..., description="The original query")
    results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieved chunks ranked by relevance"
    )
    navigation_path: List[NavigationStep] = Field(
        default_factory=list,
        description="Full navigation path through the cluster tree"
    )
    total_llm_calls: int = Field(
        default=0,
        ge=0,
        description="Total number of LLM/LLMRouter calls made"
    )
    total_time_ms: Optional[float] = Field(
        default=None,
        description="Total retrieval time in milliseconds"
    )
    explored_clusters: int = Field(
        default=0,
        ge=0,
        description="Total number of clusters explored"
    )
    total_chunks_evaluated: int = Field(
        default=0,
        ge=0,
        description="Total number of chunks evaluated in final selection"
    )

    def get_top_result(self) -> Optional[RetrievalResult]:
        """Get the top-ranked result."""
        return self.results[0] if self.results else None

    def get_top_k(self, k: int) -> List[RetrievalResult]:
        """Get top k results."""
        return self.results[:k]

    def format_navigation_path(self) -> str:
        """Format the navigation path as a readable string."""
        if not self.navigation_path:
            return "No navigation path"

        path_str = "Navigation Path:\n"
        for step in self.navigation_path:
            indent = "  " * step.level
            path_str += f"{indent}L{step.level}: {step.cluster_name} (confidence: {step.confidence:.2f})\n"
            if step.reasoning:
                path_str += f"{indent}     Reasoning: {step.reasoning[:100]}...\n"

        return path_str


class RetrievalConfig(BaseModel):
    """Configuration for hierarchical retrieval."""

    top_k: int = Field(
        default=3,
        ge=1,
        description="Number of top chunks to retrieve"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for selecting a path"
    )
    explore_multiple_paths: bool = Field(
        default=False,
        description="Whether to explore multiple cluster paths in parallel"
    )
    max_parallel_paths: int = Field(
        default=2,
        ge=1,
        description="Maximum number of parallel paths to explore if enabled"
    )
    include_reasoning: bool = Field(
        default=True,
        description="Whether to include reasoning in results"
    )
    fallback_to_siblings: bool = Field(
        default=False,
        description="If selected cluster has no results, try sibling clusters"
    )


class RetrievalStats(BaseModel):
    """Statistics about retrieval performance."""

    total_retrievals: int = Field(default=0, ge=0)
    total_llm_calls: int = Field(default=0, ge=0)
    average_time_ms: float = Field(default=0.0, ge=0.0)
    average_navigation_depth: float = Field(default=0.0, ge=0.0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
