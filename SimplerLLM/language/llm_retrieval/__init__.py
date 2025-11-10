"""
LLM-based hierarchical retrieval module for SimplerLLM.

This module provides intelligent retrieval through cluster trees using
LLMRouter for navigation decisions, offering explainable and accurate
retrieval without relying on embeddings for the retrieval step.

Key Features:
- Hierarchical tree navigation using LLMRouter
- Full explainability with reasoning chains
- Confidence scores at every decision point
- Multi-level cluster traversal
- Performance statistics tracking

Example Usage:
    ```python
    from SimplerLLM.language.llm import LLM, LLMProvider
    from SimplerLLM.language.llm_router import LLMRouter
    from SimplerLLM.language.llm_clustering import LLMClusterer, ChunkReference
    from SimplerLLM.language.llm_retrieval import LLMRetriever, RetrievalConfig

    # Setup
    llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4")
    router = LLMRouter(llm)

    # Cluster documents
    clusterer = LLMClusterer(llm)
    chunks = [ChunkReference(chunk_id=i, text=text) for i, text in enumerate(texts)]
    clustering_result = clusterer.cluster(chunks, build_hierarchy=True)

    # Setup retriever
    retriever = LLMRetriever(router, clustering_result.tree)

    # Retrieve relevant chunks
    response = retriever.retrieve(
        query="What are the main AI safety challenges?",
        top_k=3
    )

    # Access results
    for result in response.results:
        print(f"Rank {result.rank}: {result.chunk_text[:100]}...")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Path: {' -> '.join(result.cluster_path)}")
        print(f"Reasoning: {result.reasoning}")

    # View navigation path
    print(response.format_navigation_path())
    ```
"""

from .models import (
    # Retrieval models
    NavigationStep,
    RetrievalResult,
    HierarchicalRetrievalResponse,

    # Configuration
    RetrievalConfig,
    RetrievalStats,
)

from .retriever import LLMRetriever

__all__ = [
    # Main API
    "LLMRetriever",

    # Data models
    "NavigationStep",
    "RetrievalResult",
    "HierarchicalRetrievalResponse",

    # Configuration
    "RetrievalConfig",
    "RetrievalStats",
]

__version__ = "0.1.0"
