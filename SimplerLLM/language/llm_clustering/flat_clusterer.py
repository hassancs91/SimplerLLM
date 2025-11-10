"""
Flat clustering implementation using incremental LLM-based matching.

This module provides the core clustering algorithm that processes chunks
in batches and incrementally builds clusters with consistent metadata.
"""

from typing import List, Optional, Dict, Any
import uuid
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from .models import (
    Cluster,
    ClusterMetadata,
    ChunkReference,
    ClusterMatch,
    ChunkMatchingResult,
    ClusteringConfig,
    ClusteringResult
)


class FlatClusterer:
    """
    Implements incremental flat clustering using LLM-based semantic matching.

    This clusterer processes chunks in batches, matching them against existing
    clusters or creating new ones. It maintains consistency by reusing exact
    cluster metadata throughout the process.
    """

    def __init__(self, llm_instance, config: Optional[ClusteringConfig] = None):
        """
        Initialize the flat clusterer.

        Args:
            llm_instance: LLM instance for generating structured outputs
            config: Clustering configuration parameters
        """
        self.llm = llm_instance
        self.config = config or ClusteringConfig()
        self.total_llm_calls = 0

    def cluster_chunks(
        self,
        chunks: List[ChunkReference],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ClusteringResult:
        """
        Cluster chunks using incremental batch matching.

        Args:
            chunks: List of chunks to cluster
            metadata: Optional metadata about the source document

        Returns:
            ClusteringResult with flat clusters
        """
        clusters: List[Cluster] = []
        chunk_to_clusters: Dict[int, List[str]] = {}

        # Process chunks in batches sequentially
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]

            # Process this batch against existing clusters
            batch_results = self._process_batch(batch, clusters)

            # Update clusters and assignments based on results
            for chunk, matching_result in zip(batch, batch_results):
                assigned_cluster_ids = []

                # Handle existing cluster matches
                if matching_result.matches:
                    # Sort by confidence and take top max_clusters_per_chunk
                    sorted_matches = sorted(
                        matching_result.matches,
                        key=lambda m: m.confidence,
                        reverse=True
                    )[:self.config.max_clusters_per_chunk]

                    for match in sorted_matches:
                        if match.confidence >= self.config.confidence_threshold:
                            # Assign to existing cluster
                            cluster = next(c for c in clusters if c.id == match.cluster_id)
                            cluster.add_chunk(chunk)
                            assigned_cluster_ids.append(match.cluster_id)
                        elif self.config.below_threshold_behavior in ["assign_and_create", "force_assign"]:
                            # Below threshold but still assign best match
                            if sorted_matches[0].cluster_id == match.cluster_id:
                                cluster = next(c for c in clusters if c.id == match.cluster_id)
                                cluster.add_chunk(chunk)
                                assigned_cluster_ids.append(match.cluster_id)

                # Handle new cluster creation
                should_create_new = (
                    matching_result.create_new_cluster or
                    (not assigned_cluster_ids and self.config.below_threshold_behavior != "force_assign") or
                    (self.config.below_threshold_behavior == "assign_and_create" and
                     matching_result.matches and
                     matching_result.matches[0].confidence < self.config.confidence_threshold)
                )

                if should_create_new and len(clusters) < self.config.max_total_clusters:
                    new_cluster = self._create_new_cluster(
                        chunk,
                        matching_result.new_cluster_metadata
                    )
                    clusters.append(new_cluster)
                    assigned_cluster_ids.append(new_cluster.id)

                # Record chunk-to-cluster mapping
                if assigned_cluster_ids:
                    chunk_to_clusters[chunk.chunk_id] = assigned_cluster_ids

        return ClusteringResult(
            clusters=clusters,
            chunk_to_clusters=chunk_to_clusters,
            clustering_config=self.config,
            total_chunks_processed=len(chunks),
            total_llm_calls=self.total_llm_calls
        )

    def _process_batch(
        self,
        batch: List[ChunkReference],
        existing_clusters: List[Cluster]
    ) -> List[ChunkMatchingResult]:
        """
        Process a batch of chunks against existing clusters.

        Args:
            batch: Chunks to process
            existing_clusters: Current clusters to match against

        Returns:
            List of matching results for each chunk in batch
        """
        if not existing_clusters:
            # No existing clusters, create new ones for each chunk
            return [
                self._match_chunk_to_clusters(chunk, [])
                for chunk in batch
            ]

        # Build prompt with existing clusters and batch
        results = []
        for chunk in batch:
            result = self._match_chunk_to_clusters(chunk, existing_clusters)
            results.append(result)

        return results

    def _match_chunk_to_clusters(
        self,
        chunk: ChunkReference,
        clusters: List[Cluster]
    ) -> ChunkMatchingResult:
        """
        Match a single chunk against existing clusters using LLM.

        Args:
            chunk: The chunk to match
            clusters: Existing clusters to match against

        Returns:
            ChunkMatchingResult with matches and/or new cluster suggestion
        """
        if not clusters:
            # No clusters exist yet, need to create first one
            metadata = self._generate_cluster_metadata(chunk)
            return ChunkMatchingResult(
                matches=[],
                create_new_cluster=True,
                new_cluster_metadata=metadata,
                confidence_threshold_used=self.config.confidence_threshold
            )

        # Build prompt for LLM to match chunk against clusters
        cluster_summaries = self._format_cluster_summaries(clusters)

        prompt = f"""You are analyzing a text chunk to determine which existing cluster(s) it belongs to.

EXISTING CLUSTERS:
{cluster_summaries}

---

CHUNK TO ANALYZE:
"{chunk.text}"

---

TASK:
1. Determine if this chunk matches ANY of the existing clusters
2. A chunk can match MULTIPLE clusters if it's relevant to multiple topics
3. For each match, provide a confidence score (0.0 to 1.0)
4. If the chunk doesn't match any existing cluster well, indicate that a NEW cluster should be created
5. If creating a new cluster, provide the metadata for it

Respond with:
- matches: List of matching clusters with confidence scores and reasoning
- create_new_cluster: true if a new cluster is needed, false otherwise
- new_cluster_metadata: If creating new cluster, provide canonical_name, canonical_tags, canonical_keywords, description
"""

        # Call LLM with structured output
        try:
            response = generate_pydantic_json_model(
                model_class=ChunkMatchingResult,
                prompt=prompt,
                llm_instance=self.llm,
                max_tokens=1000,
                system_prompt="You are an expert at semantic text analysis and clustering."
            )
            self.total_llm_calls += 1

            # Check if response is an error string
            if isinstance(response, str):
                raise ValueError(f"LLM generation failed: {response}")

            # Ensure confidence threshold is set
            response.confidence_threshold_used = self.config.confidence_threshold

            return response

        except Exception as e:
            # Fallback: create new cluster if LLM fails
            print(f"Warning: LLM call failed: {e}. Creating new cluster as fallback.")
            # Use UUID to ensure unique cluster names
            unique_id = uuid.uuid4().hex[:8]
            metadata = ClusterMetadata(
                canonical_name=f"Uncategorized_{unique_id}",
                canonical_tags=["uncategorized"],
                canonical_keywords=[],
                description="Auto-generated cluster due to processing error"
            )
            return ChunkMatchingResult(
                matches=[],
                create_new_cluster=True,
                new_cluster_metadata=metadata,
                confidence_threshold_used=self.config.confidence_threshold
            )

    def _create_new_cluster(
        self,
        chunk: ChunkReference,
        metadata: Optional[ClusterMetadata] = None
    ) -> Cluster:
        """
        Create a new cluster with the given chunk.

        Args:
            chunk: Initial chunk for the cluster
            metadata: Cluster metadata (generated if not provided)

        Returns:
            New Cluster object
        """
        if metadata is None:
            metadata = self._generate_cluster_metadata(chunk)

        cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"

        cluster = Cluster(
            id=cluster_id,
            level=0,  # Leaf level
            metadata=metadata,
            chunks=[chunk],
            chunk_count=1
        )

        return cluster

    def _generate_cluster_metadata(self, chunk: ChunkReference) -> ClusterMetadata:
        """
        Generate metadata for a new cluster based on a chunk.

        Args:
            chunk: Representative chunk for the new cluster

        Returns:
            ClusterMetadata
        """
        prompt = f"""Analyze this text chunk and generate metadata for a new cluster.

CHUNK:
"{chunk.text}"

---

Generate:
- canonical_name: A clear, descriptive name for this cluster (e.g., "AI Safety Discussion")
- canonical_tags: 3-5 tags for categorization (lowercase, hyphenated)
- canonical_keywords: 5-10 key terms and phrases
- description: A detailed description of the cluster's theme
- topic: The primary topic in 2-4 words

Make the name and tags consistent and professional. Use clear, searchable terminology.
"""

        try:
            response = generate_pydantic_json_model(
                model_class=ClusterMetadata,
                prompt=prompt,
                llm_instance=self.llm,
                max_tokens=500,
                system_prompt="You are an expert at semantic text analysis and creating descriptive metadata."
            )
            self.total_llm_calls += 1

            # Check if response is an error string
            if isinstance(response, str):
                raise ValueError(f"LLM generation failed: {response}")

            return response

        except Exception as e:
            # Fallback metadata with unique name
            print(f"Warning: Failed to generate metadata: {e}. Using fallback.")
            unique_id = uuid.uuid4().hex[:8]
            return ClusterMetadata(
                canonical_name=f"Uncategorized_{unique_id}",
                canonical_tags=["general"],
                canonical_keywords=[],
                description="Auto-generated cluster"
            )

    def _format_cluster_summaries(self, clusters: List[Cluster]) -> str:
        """
        Format cluster metadata into compact summaries for LLM prompt.

        Args:
            clusters: Clusters to summarize

        Returns:
            Formatted string of cluster summaries
        """
        summaries = []
        for i, cluster in enumerate(clusters, 1):
            summary = f"""Cluster {i}: "{cluster.metadata.canonical_name}"
  ID: {cluster.id}
  Tags: {', '.join(cluster.metadata.canonical_tags)}
  Keywords: {', '.join(cluster.metadata.canonical_keywords[:5])}
  Description: {cluster.metadata.description[:200]}"""
            summaries.append(summary)

        return "\n\n".join(summaries)

    async def cluster_chunks_async(
        self,
        chunks: List[ChunkReference],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ClusteringResult:
        """
        Async version of cluster_chunks.

        Note: Currently calls sync version. Implement async LLM calls
        for true async behavior.
        """
        # TODO: Implement true async when LLM supports it
        return self.cluster_chunks(chunks, metadata)
