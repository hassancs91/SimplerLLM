"""
Hierarchical tree builder that organizes flat clusters into a tree structure.

This module applies the same incremental matching strategy to group clusters
into parent clusters, creating a hierarchical navigation structure.
"""

from typing import List, Optional, Dict
import uuid
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from .models import (
    Cluster,
    ClusterMetadata,
    ClusterTree,
    TreeConfig,
    ClusterMatch,
    ChunkMatchingResult
)


class TreeBuilder:
    """
    Builds hierarchical cluster trees using incremental LLM-based grouping.

    Takes flat clusters and organizes them into a multi-level tree structure
    by matching clusters to parent clusters, similar to how chunks are matched
    to clusters in flat clustering.
    """

    def __init__(self, llm_instance, config: Optional[TreeConfig] = None):
        """
        Initialize the tree builder.

        Args:
            llm_instance: LLM instance for generating structured outputs
            config: Tree building configuration
        """
        self.llm = llm_instance
        self.config = config or TreeConfig()
        self.total_llm_calls = 0

    def build_hierarchy(self, flat_clusters: List[Cluster]) -> ClusterTree:
        """
        Build a hierarchical tree from flat clusters.

        Args:
            flat_clusters: List of leaf-level clusters to organize

        Returns:
            ClusterTree with hierarchical structure
        """
        if not flat_clusters:
            return ClusterTree(config=self.config)

        # Start with leaf clusters at level 0
        current_level_clusters = flat_clusters
        current_level = 0
        tree = ClusterTree(config=self.config)

        # Add all leaf clusters to tree
        for cluster in flat_clusters:
            cluster.level = 0
            tree.add_cluster(cluster)

        # Build hierarchy bottom-up
        while True:
            # Check if we need another level
            if len(current_level_clusters) <= self.config.max_clusters_per_level:
                # Top level reached, these are root clusters
                tree.root_cluster_ids = [c.id for c in current_level_clusters]
                break

            if current_level + 1 >= self.config.max_depth:
                # Max depth reached, stop here
                tree.root_cluster_ids = [c.id for c in current_level_clusters]
                break

            # Create parent level
            parent_clusters = self._group_clusters_into_parents(
                current_level_clusters,
                current_level + 1
            )

            # Add parent clusters to tree
            for parent in parent_clusters:
                tree.add_cluster(parent)

                # Update children to point to parent
                for child_id in parent.child_clusters:
                    child = tree.get_cluster(child_id)
                    if child:
                        child.parent_id = parent.id

            # Move up to next level
            current_level_clusters = parent_clusters
            current_level += 1

        # Calculate total chunks
        tree.total_chunks = sum(c.chunk_count for c in flat_clusters)

        return tree

    def _group_clusters_into_parents(
        self,
        clusters: List[Cluster],
        parent_level: int
    ) -> List[Cluster]:
        """
        Group clusters into parent clusters using incremental matching.

        Args:
            clusters: Clusters to group
            parent_level: Level number for parent clusters

        Returns:
            List of parent clusters
        """
        parent_clusters: List[Cluster] = []

        # Process each cluster and match to parent
        for cluster in clusters:
            matching_result = self._match_cluster_to_parents(cluster, parent_clusters)

            if matching_result.matches and matching_result.matches[0].confidence >= 0.7:
                # Assign to existing parent
                best_match = matching_result.matches[0]
                parent = next(p for p in parent_clusters if p.id == best_match.cluster_id)
                parent.child_clusters.append(cluster.id)
                parent.chunk_count += cluster.chunk_count

            else:
                # Create new parent cluster
                new_parent = self._create_parent_cluster([cluster], parent_level)
                parent_clusters.append(new_parent)

        return parent_clusters

    def _match_cluster_to_parents(
        self,
        cluster: Cluster,
        parent_clusters: List[Cluster]
    ) -> ChunkMatchingResult:
        """
        Match a cluster against existing parent clusters.

        Args:
            cluster: Cluster to match
            parent_clusters: Existing parent clusters

        Returns:
            Matching result
        """
        if not parent_clusters:
            # No parents exist, need to create first one
            metadata = self._generate_parent_metadata([cluster])
            return ChunkMatchingResult(
                matches=[],
                create_new_cluster=True,
                new_cluster_metadata=metadata,
                confidence_threshold_used=0.7
            )

        # Build prompt for matching
        parent_summaries = self._format_parent_summaries(parent_clusters)
        cluster_info = self._format_cluster_info(cluster)

        prompt = f"""You are organizing clusters into a hierarchical structure by grouping similar clusters under parent clusters.

EXISTING PARENT CLUSTERS:
{parent_summaries}

---

CLUSTER TO ASSIGN:
{cluster_info}

---

TASK:
Determine if this cluster should be grouped under any of the existing parent clusters.
Consider:
- Thematic similarity
- Topic overlap
- Conceptual relationships

Respond with:
- matches: List of matching parent clusters with confidence scores (0.0 to 1.0)
- create_new_cluster: true if this cluster doesn't fit any existing parent, false otherwise
- new_cluster_metadata: If creating a new parent, provide metadata
"""

        try:
            response = generate_pydantic_json_model(
                model_class=ChunkMatchingResult,
                prompt=prompt,
                llm_instance=self.llm,
                max_tokens=800,
                system_prompt="You are an expert at organizing information into hierarchical structures."
            )
            self.total_llm_calls += 1

            # Check if response is an error string
            if isinstance(response, str):
                raise ValueError(f"LLM generation failed: {response}")

            return response

        except Exception as e:
            print(f"Warning: Failed to match cluster to parents: {e}")
            metadata = self._generate_parent_metadata([cluster])
            return ChunkMatchingResult(
                matches=[],
                create_new_cluster=True,
                new_cluster_metadata=metadata,
                confidence_threshold_used=0.7
            )

    def _create_parent_cluster(
        self,
        child_clusters: List[Cluster],
        level: int
    ) -> Cluster:
        """
        Create a parent cluster from child clusters.

        Args:
            child_clusters: Child clusters to group
            level: Level for the parent cluster

        Returns:
            New parent Cluster
        """
        metadata = self._generate_parent_metadata(child_clusters)
        cluster_id = f"parent_{uuid.uuid4().hex[:8]}"

        total_chunks = sum(c.chunk_count for c in child_clusters)

        parent = Cluster(
            id=cluster_id,
            level=level,
            metadata=metadata,
            chunks=[],  # Parents don't directly contain chunks
            child_clusters=[c.id for c in child_clusters],
            chunk_count=total_chunks
        )

        return parent

    def _generate_parent_metadata(
        self,
        child_clusters: List[Cluster]
    ) -> ClusterMetadata:
        """
        Generate metadata for a parent cluster based on its children.

        Args:
            child_clusters: Child clusters to summarize

        Returns:
            ClusterMetadata for parent
        """
        # Format child cluster information
        child_info = []
        for child in child_clusters:
            info = f"""- {child.metadata.canonical_name}
  Tags: {', '.join(child.metadata.canonical_tags)}
  Description: {child.metadata.description[:150]}"""
            child_info.append(info)

        child_summary = "\n".join(child_info)

        prompt = f"""You are creating a parent cluster that groups together related sub-clusters.

SUB-CLUSTERS TO GROUP:
{child_summary}

---

Generate metadata for a parent cluster that encompasses these sub-clusters:
- canonical_name: A broader, encompassing name that covers all sub-clusters
- canonical_tags: Common tags that apply across sub-clusters
- canonical_keywords: Key terms from all sub-clusters
- description: Description of the overall theme/topic
- topic: The overarching topic

The parent cluster should be at a higher level of abstraction than the sub-clusters.
"""

        try:
            response = generate_pydantic_json_model(
                model_class=ClusterMetadata,
                prompt=prompt,
                llm_instance=self.llm,
                max_tokens=500,
                system_prompt="You are an expert at creating hierarchical taxonomies and metadata."
            )
            self.total_llm_calls += 1

            # Check if response is an error string
            if isinstance(response, str):
                raise ValueError(f"LLM generation failed: {response}")

            return response

        except Exception as e:
            print(f"Warning: Failed to generate parent metadata: {e}")
            # Fallback: merge tags and create unique generic name
            all_tags = set()
            for child in child_clusters:
                all_tags.update(child.metadata.canonical_tags)

            unique_id = uuid.uuid4().hex[:8]
            return ClusterMetadata(
                canonical_name=f"General Topics_{unique_id}",
                canonical_tags=list(all_tags)[:5],
                canonical_keywords=[],
                description="Parent cluster grouping related topics"
            )

    def _format_parent_summaries(self, parents: List[Cluster]) -> str:
        """Format parent cluster summaries for LLM prompt."""
        summaries = []
        for i, parent in enumerate(parents, 1):
            summary = f"""Parent {i}: "{parent.metadata.canonical_name}"
  ID: {parent.id}
  Tags: {', '.join(parent.metadata.canonical_tags)}
  Children: {len(parent.child_clusters)} clusters
  Total chunks: {parent.chunk_count}
  Description: {parent.metadata.description[:200]}"""
            summaries.append(summary)

        return "\n\n".join(summaries)

    def _format_cluster_info(self, cluster: Cluster) -> str:
        """Format cluster information for LLM prompt."""
        return f"""Cluster: "{cluster.metadata.canonical_name}"
  ID: {cluster.id}
  Tags: {', '.join(cluster.metadata.canonical_tags)}
  Keywords: {', '.join(cluster.metadata.canonical_keywords[:8])}
  Chunks: {cluster.chunk_count}
  Description: {cluster.metadata.description}"""

    def _determine_optimal_depth(self, cluster_count: int) -> int:
        """
        Determine optimal tree depth based on cluster count.

        Args:
            cluster_count: Number of leaf clusters

        Returns:
            Recommended tree depth
        """
        if cluster_count <= self.config.max_clusters_per_level:
            return 1  # Flat structure is fine
        elif cluster_count <= 50:
            return 2
        elif cluster_count <= 100:
            return 3
        else:
            return min(4, self.config.max_depth)
