"""
Hierarchical retrieval using LLMRouter to navigate cluster trees.

This module implements intelligent retrieval by using LLMRouter to navigate
from root clusters down to specific chunks, providing explainable and accurate
retrieval without relying on embeddings.
"""

from typing import List, Optional, Tuple
import time
from ..llm_router import LLMRouter
from ..llm_clustering.models import ClusterTree, Cluster
from .models import (
    NavigationStep,
    RetrievalResult,
    HierarchicalRetrievalResponse,
    RetrievalConfig,
    RetrievalStats
)


class LLMRetriever:
    """
    Hierarchical retrieval using LLMRouter for cluster tree navigation.

    This retriever navigates a cluster tree using LLMRouter at each level,
    providing transparent, explainable retrieval with full reasoning chains.

    Example:
        ```python
        from SimplerLLM.language.llm import LLM, LLMProvider
        from SimplerLLM.language.llm_router import LLMRouter
        from SimplerLLM.language.llm_retrieval import LLMRetriever, RetrievalConfig

        # Setup
        llm = LLM.create(provider=LLMProvider.ANTHROPIC)
        router = LLMRouter(llm)
        retriever = LLMRetriever(router, cluster_tree)

        # Retrieve
        response = retriever.retrieve(
            query="What are technical approaches to AI alignment?",
            top_k=3
        )

        # Access results
        for result in response.results:
            print(f"Chunk {result.chunk_id}: {result.chunk_text[:100]}...")
            print(f"Confidence: {result.confidence}")
            print(f"Path: {' -> '.join(result.cluster_path)}")
        ```
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        cluster_tree: ClusterTree,
        config: Optional[RetrievalConfig] = None,
        chunk_store=None
    ):
        """
        Initialize the retriever.

        Args:
            llm_router: LLMRouter instance for navigation decisions
            cluster_tree: Hierarchical cluster tree to navigate
            config: Retrieval configuration
            chunk_store: Optional ChunkStore for lazy-loading chunks (for large datasets)
        """
        self.router = llm_router
        self.tree = cluster_tree
        self.config = config or RetrievalConfig()
        self.chunk_store = chunk_store
        self.stats = RetrievalStats()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> HierarchicalRetrievalResponse:
        """
        Retrieve relevant chunks by navigating the cluster tree.

        Args:
            query: User's query
            top_k: Number of chunks to retrieve (overrides config)

        Returns:
            HierarchicalRetrievalResponse with results and navigation path
        """
        start_time = time.time()
        k = top_k or self.config.top_k

        # Start navigation from root
        navigation_path: List[NavigationStep] = []
        llm_calls = 0

        # Navigate through tree levels
        current_clusters = [
            self.tree.get_cluster(cid)
            for cid in self.tree.root_cluster_ids
        ]
        current_clusters = [c for c in current_clusters if c is not None]

        current_level = self.tree.max_depth
        cluster_path: List[str] = []
        cluster_id_path: List[str] = []

        # Navigate down the tree
        while current_clusters:
            # Select best cluster at this level
            selected_cluster, step = self._route_to_cluster(
                query,
                current_clusters,
                current_level
            )
            llm_calls += 1

            if selected_cluster is None:
                # No good match found
                break

            # Record navigation step
            navigation_path.append(step)
            cluster_path.append(selected_cluster.metadata.canonical_name)
            cluster_id_path.append(selected_cluster.id)

            # Check if we're at leaf level
            if selected_cluster.is_leaf():
                # We've reached chunks, now select best chunks
                results = self._route_to_chunks(
                    query,
                    selected_cluster,
                    k,
                    cluster_path,
                    cluster_id_path
                )
                llm_calls += 1
                break

            # Navigate to children
            child_ids = selected_cluster.child_clusters
            current_clusters = [
                self.tree.get_cluster(cid)
                for cid in child_ids
            ]
            current_clusters = [c for c in current_clusters if c is not None]
            current_level -= 1
        else:
            # No clusters to navigate
            results = []

        # Build response
        total_time = (time.time() - start_time) * 1000  # Convert to ms

        response = HierarchicalRetrievalResponse(
            query=query,
            results=results,
            navigation_path=navigation_path,
            total_llm_calls=llm_calls,
            total_time_ms=total_time,
            explored_clusters=len(navigation_path),
            total_chunks_evaluated=len(results) if results else 0
        )

        # Update stats
        self._update_stats(response)

        return response

    def _route_to_cluster(
        self,
        query: str,
        clusters: List[Cluster],
        level: int
    ) -> Tuple[Optional[Cluster], NavigationStep]:
        """
        Use LLMRouter to select best cluster at current level.

        Args:
            query: User query
            clusters: Clusters to choose from
            level: Current level in tree

        Returns:
            Tuple of (selected_cluster, navigation_step)
        """
        # Prepare choices for LLMRouter
        self.router.remove_all_choices()

        for cluster in clusters:
            # Format cluster information for router
            choice_text = self._format_cluster_for_router(cluster)
            self.router.add_choice(
                content=choice_text,
                metadata={"cluster_id": cluster.id}
            )

        # Route using LLMRouter
        try:
            route_result = self.router.route(query)

            if route_result and route_result.confidence_score >= self.config.confidence_threshold:
                # Get selected cluster
                selected_cluster = clusters[route_result.selected_index]

                step = NavigationStep(
                    level=level,
                    cluster_name=selected_cluster.metadata.canonical_name,
                    cluster_id=selected_cluster.id,
                    confidence=route_result.confidence_score,
                    reasoning=route_result.reasoning if self.config.include_reasoning else "",
                    alternatives_considered=len(clusters)
                )

                return selected_cluster, step

            else:
                # Low confidence, create step anyway
                step = NavigationStep(
                    level=level,
                    cluster_name="No match",
                    cluster_id="",
                    confidence=0.0,
                    reasoning="No cluster met confidence threshold",
                    alternatives_considered=len(clusters)
                )
                return None, step

        except Exception as e:
            print(f"Warning: Routing failed at level {level}: {e}")
            step = NavigationStep(
                level=level,
                cluster_name="Error",
                cluster_id="",
                confidence=0.0,
                reasoning=f"Routing error: {str(e)}",
                alternatives_considered=len(clusters)
            )
            return None, step

    def _route_to_chunks(
        self,
        query: str,
        leaf_cluster: Cluster,
        top_k: int,
        cluster_path: List[str],
        cluster_id_path: List[str]
    ) -> List[RetrievalResult]:
        """
        Use LLMRouter to select best chunks from leaf cluster.

        Args:
            query: User query
            leaf_cluster: Leaf cluster containing chunks
            top_k: Number of chunks to retrieve
            cluster_path: Path of cluster names from root
            cluster_id_path: Path of cluster IDs from root

        Returns:
            List of RetrievalResults
        """
        # Get chunks using lazy loading if chunk_store is provided
        chunks = leaf_cluster.get_chunks(self.chunk_store)

        if not chunks:
            return []

        # Prepare choices for LLMRouter
        self.router.remove_all_choices()

        for chunk in chunks:
            self.router.add_choice(
                content=chunk.text,
                metadata={"chunk_id": chunk.chunk_id}
            )

        # Get top-k results
        try:
            route_results = self.router.route_top_k(query, k=min(top_k, len(chunks)))

            if not route_results:
                return []

            # Build RetrievalResult objects
            results = []
            for rank, route_result in enumerate(route_results, 1):
                chunk = chunks[route_result.selected_index]

                result = RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.text,
                    confidence=route_result.confidence_score,
                    reasoning=route_result.reasoning if self.config.include_reasoning else "",
                    cluster_path=cluster_path.copy(),
                    cluster_id_path=cluster_id_path.copy(),
                    metadata=chunk.metadata,
                    rank=rank
                )
                results.append(result)

            return results

        except Exception as e:
            print(f"Warning: Chunk routing failed: {e}")
            return []

    def _format_cluster_for_router(self, cluster: Cluster) -> str:
        """
        Format cluster information for LLMRouter choice.

        Args:
            cluster: Cluster to format

        Returns:
            Formatted string for router
        """
        metadata = cluster.metadata

        formatted = f"{metadata.canonical_name}\n"
        formatted += f"Tags: {', '.join(metadata.canonical_tags[:5])}\n"
        formatted += f"Keywords: {', '.join(metadata.canonical_keywords[:8])}\n"
        formatted += f"Description: {metadata.description[:300]}"

        if cluster.is_parent():
            formatted += f"\n({len(cluster.child_clusters)} sub-clusters, {cluster.chunk_count} chunks)"

        return formatted

    def update_tree(self, new_cluster_tree: ClusterTree):
        """
        Update the cluster tree.

        Useful for swapping in a new/updated tree structure.

        Args:
            new_cluster_tree: New cluster tree
        """
        self.tree = new_cluster_tree

    def get_retrieval_stats(self) -> RetrievalStats:
        """
        Get retrieval performance statistics.

        Returns:
            RetrievalStats object
        """
        return self.stats

    def _update_stats(self, response: HierarchicalRetrievalResponse):
        """Update internal statistics."""
        self.stats.total_retrievals += 1
        self.stats.total_llm_calls += response.total_llm_calls

        # Update average time
        if response.total_time_ms:
            total_time = self.stats.average_time_ms * (self.stats.total_retrievals - 1)
            total_time += response.total_time_ms
            self.stats.average_time_ms = total_time / self.stats.total_retrievals

        # Update average depth
        depth = len(response.navigation_path)
        total_depth = self.stats.average_navigation_depth * (self.stats.total_retrievals - 1)
        total_depth += depth
        self.stats.average_navigation_depth = total_depth / self.stats.total_retrievals

    async def retrieve_async(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> HierarchicalRetrievalResponse:
        """
        Async version of retrieve().

        Note: Currently calls sync version. Implement async LLMRouter
        for true async behavior.

        Args:
            query: User's query
            top_k: Number of chunks to retrieve

        Returns:
            HierarchicalRetrievalResponse
        """
        # TODO: Implement true async when LLMRouter supports it
        return self.retrieve(query, top_k)
