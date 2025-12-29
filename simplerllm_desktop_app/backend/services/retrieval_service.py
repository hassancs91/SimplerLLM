"""
Retrieval Service - Wrapper for SimplerLLM LLMClusterer and LLMRetriever with streaming support.
"""
import threading
import queue
import time
from typing import Dict, Any, Optional, Generator, List
from services.llm_service import LLMService

# Import SimplerLLM components
try:
    from SimplerLLM.language.llm_clustering import LLMClusterer, ChunkReference
    from SimplerLLM.language.llm_retrieval import LLMRetriever, RetrievalConfig
    from SimplerLLM.language.llm_router import LLMRouter
except ImportError:
    from simplerllm.language.llm_clustering import LLMClusterer, ChunkReference
    from simplerllm.language.llm_retrieval import LLMRetriever, RetrievalConfig
    from simplerllm.language.llm_router import LLMRouter


# Sample datasets for demonstration
SAMPLE_DATASETS = {
    'ai_news': {
        'id': 'ai_news',
        'name': 'AI News Articles',
        'description': '10 short articles about recent AI developments and breakthroughs',
        'texts': [
            "OpenAI announced GPT-5, their most capable language model yet. The new model shows significant improvements in reasoning and can solve complex mathematical problems that previous versions struggled with. Researchers claim it achieves near-human performance on several academic benchmarks.",

            "Google DeepMind's AlphaFold 3 can now predict protein-drug interactions with unprecedented accuracy. This breakthrough could accelerate drug discovery by identifying promising compounds before expensive clinical trials. The model has already been used to identify potential treatments for rare diseases.",

            "Anthropic released Claude 3, featuring improved safety measures and reduced hallucinations. The company emphasized their commitment to AI safety research and constitutional AI approaches. Claude 3 demonstrates better refusal of harmful requests while remaining helpful for legitimate use cases.",

            "Meta open-sourced Llama 3, their largest language model to date. The 400 billion parameter model is freely available for commercial use, democratizing access to frontier AI capabilities. Researchers worldwide can now fine-tune and study the model architecture.",

            "AI safety researchers warn about the risks of artificial general intelligence. Leading scientists published an open letter calling for more research into AI alignment and interpretability. They argue that current safety measures may be insufficient for more capable future systems.",

            "Nvidia unveiled their next-generation H200 GPU optimized for AI training. The new chip offers 2x performance improvement over the H100 while maintaining similar power consumption. Major cloud providers have already placed orders for millions of units.",

            "Microsoft integrated AI copilots across their entire Office suite. Users can now generate documents, analyze spreadsheets, and create presentations using natural language commands. Early reports suggest significant productivity improvements for knowledge workers.",

            "Tesla's Full Self-Driving system achieved Level 4 autonomy certification in several states. The system can now operate without human supervision in specific geographic areas. Regulatory approval came after extensive safety testing and data collection.",

            "Researchers developed AI that can generate realistic 3D scenes from text descriptions. The model combines diffusion techniques with neural radiance fields to create immersive environments. Applications range from video game development to architectural visualization.",

            "China unveiled their own frontier AI model rivaling GPT-4 capabilities. The government-backed project demonstrates significant progress in large language model research. Experts debate whether this signals an intensifying AI race between major powers."
        ]
    },
    'tech_docs': {
        'id': 'tech_docs',
        'name': 'Tech Documentation',
        'description': 'Programming tutorials and documentation excerpts',
        'texts': [
            "Python decorators are a powerful feature that allows you to modify the behavior of functions. A decorator is a function that takes another function as an argument and returns a modified version. Common uses include logging, authentication checks, and caching results.",

            "React hooks revolutionized how developers write functional components. The useState hook manages local state, while useEffect handles side effects like API calls. Custom hooks allow you to extract and reuse stateful logic across components.",

            "Docker containers provide lightweight, portable environments for applications. Unlike virtual machines, containers share the host OS kernel, making them faster and more efficient. Docker Compose allows you to define multi-container applications in a single YAML file.",

            "Git branching strategies help teams collaborate on code effectively. The most common approaches are GitFlow, trunk-based development, and GitHub Flow. Each has trade-offs between complexity and flexibility for different team sizes.",

            "SQL query optimization starts with understanding your database's execution plan. Adding appropriate indexes can dramatically improve query performance. However, too many indexes can slow down write operations, so balance is important.",

            "REST API design follows principles like statelessness and resource-based URLs. Use HTTP methods semantically: GET for reading, POST for creating, PUT for updating, DELETE for removing. Proper status codes help clients understand the response type.",

            "TypeScript adds static typing to JavaScript, catching errors at compile time. Interfaces define object shapes, while generics enable type-safe reusable code. The compiler can infer types in many cases, reducing boilerplate.",

            "Kubernetes orchestrates containerized applications across clusters. Pods are the smallest deployable units, containing one or more containers. Services provide stable network endpoints, while Ingress manages external access.",

            "Machine learning pipelines consist of data preprocessing, model training, and deployment stages. Feature engineering often has the biggest impact on model performance. Regular retraining prevents model drift as data distributions change.",

            "Microservices architecture breaks monolithic applications into independent services. Each service can be developed, deployed, and scaled independently. Communication happens through APIs or message queues like RabbitMQ or Kafka."
        ]
    },
    'research': {
        'id': 'research',
        'name': 'ML Research Abstracts',
        'description': 'Abstracts from machine learning research papers',
        'texts': [
            "We introduce a novel attention mechanism that reduces computational complexity from O(n^2) to O(n log n). Our method maintains competitive performance on standard benchmarks while enabling processing of significantly longer sequences. Experiments on language modeling and machine translation demonstrate consistent improvements.",

            "This paper proposes a self-supervised pretraining objective for vision transformers. By predicting masked patches, the model learns rich visual representations without labeled data. Transfer learning experiments show state-of-the-art results on ImageNet classification and COCO object detection.",

            "We present a theoretical framework for understanding neural network generalization. Our analysis reveals that implicit regularization from optimization algorithms plays a crucial role. These insights explain why overparameterized networks generalize despite having more parameters than training examples.",

            "This work explores the scaling laws of language models across compute, data, and parameters. We find that performance follows predictable power laws, enabling efficient resource allocation. Our findings suggest optimal model sizes given fixed compute budgets.",

            "We introduce a method for aligning language models with human preferences using reinforcement learning. Our approach combines reward modeling with policy optimization to produce more helpful and harmless responses. Human evaluators prefer our aligned model over baselines in 78% of comparisons.",

            "This paper presents a unified framework for few-shot learning across multiple modalities. By conditioning on task demonstrations, our model adapts to new tasks without fine-tuning. Experiments span text classification, image recognition, and speech processing tasks.",

            "We propose a differentiable neural architecture search algorithm that discovers efficient network designs. Unlike previous methods requiring thousands of GPU hours, our approach completes search in several hours. Discovered architectures achieve competitive accuracy with significantly fewer parameters.",

            "This work investigates emergent capabilities in large language models. We identify phase transitions where new abilities appear at specific scale thresholds. Understanding these emergent behaviors is crucial for predicting and controlling future AI systems."
        ]
    }
}


class RetrievalService:
    """Service for building retrieval indexes and querying with real-time streaming updates."""

    def __init__(self):
        self.llm_service = LLMService()
        # Store the current index state
        self._cluster_tree = None
        self._chunks = None
        self._retriever = None
        self._index_provider = None
        self._index_model = None

    def get_samples(self) -> List[Dict[str, Any]]:
        """Get available sample datasets."""
        return [
            {
                'id': sample['id'],
                'name': sample['name'],
                'description': sample['description'],
                'chunk_count': len(sample['texts'])
            }
            for sample in SAMPLE_DATASETS.values()
        ]

    def get_sample_texts(self, sample_id: str) -> Optional[List[str]]:
        """Get texts for a sample dataset."""
        sample = SAMPLE_DATASETS.get(sample_id)
        return sample['texts'] if sample else None

    def chunk_text(self, text: str) -> List[ChunkReference]:
        """
        Split text into chunks for indexing.
        Uses paragraph-based chunking (double newlines).
        """
        # Split on double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # If no paragraphs, split on single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        # If still too few, split on sentences (period followed by space)
        if len(paragraphs) <= 2 and len(text) > 500:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            # Group sentences into chunks of ~200 words
            paragraphs = []
            current_chunk = []
            current_word_count = 0
            for sentence in sentences:
                word_count = len(sentence.split())
                if current_word_count + word_count > 200 and current_chunk:
                    paragraphs.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_word_count = word_count
                else:
                    current_chunk.append(sentence)
                    current_word_count += word_count
            if current_chunk:
                paragraphs.append(' '.join(current_chunk))

        # Create ChunkReference objects
        chunks = []
        for i, text in enumerate(paragraphs):
            if text and len(text) > 20:  # Skip very short chunks
                chunks.append(ChunkReference(chunk_id=i, text=text))

        return chunks

    def build_index_streaming(
        self,
        source: str,
        text: Optional[str],
        sample_id: Optional[str],
        provider: str,
        model: str
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Build a retrieval index from text or sample with streaming updates.

        Args:
            source: 'text' or 'sample'
            text: Raw text to index (if source='text')
            sample_id: Sample dataset ID (if source='sample')
            provider: LLM provider
            model: Model name

        Yields:
            Dict events with types: 'start', 'chunking_complete', 'clustering_progress',
            'complete', 'error'
        """
        event_queue = queue.Queue()

        def build_index():
            try:
                # Get texts to index
                if source == 'sample':
                    texts = self.get_sample_texts(sample_id)
                    if not texts:
                        event_queue.put({
                            'type': 'error',
                            'error': f'Sample dataset "{sample_id}" not found'
                        })
                        return
                    # For samples, each text is already a chunk
                    chunks = [ChunkReference(chunk_id=i, text=t) for i, t in enumerate(texts)]
                else:
                    # Chunk the provided text
                    if not text or len(text.strip()) < 50:
                        event_queue.put({
                            'type': 'error',
                            'error': 'Text must be at least 50 characters'
                        })
                        return
                    chunks = self.chunk_text(text)

                if len(chunks) < 3:
                    event_queue.put({
                        'type': 'error',
                        'error': 'Need at least 3 chunks to build an index'
                    })
                    return

                # Emit chunking complete event
                event_queue.put({
                    'type': 'chunking_complete',
                    'chunks_count': len(chunks)
                })

                # Get LLM instance
                llm = self.llm_service.get_llm(provider, model)
                if not llm:
                    event_queue.put({
                        'type': 'error',
                        'error': f'Failed to initialize LLM for {provider}/{model}'
                    })
                    return

                # Emit clustering progress
                event_queue.put({
                    'type': 'clustering_progress',
                    'stage': 'flat_clustering',
                    'message': 'Grouping chunks into semantic clusters...'
                })

                # Create clusterer and build tree
                clusterer = LLMClusterer(llm)
                result = clusterer.cluster(chunks, build_hierarchy=True)

                # Store the tree and chunks
                self._cluster_tree = result.tree
                self._chunks = chunks
                self._index_provider = provider
                self._index_model = model

                # Create retriever for later queries
                router = LLMRouter(llm)
                self._retriever = LLMRetriever(
                    llm_router=router,
                    cluster_tree=self._cluster_tree,
                    config=RetrievalConfig(top_k=3, confidence_threshold=0.5)
                )

                # Serialize tree structure for frontend visualization
                tree_data = self._serialize_tree(result.tree)

                # Emit complete event
                event_queue.put({
                    'type': 'complete',
                    'tree': tree_data,
                    'stats': {
                        'total_chunks': len(chunks),
                        'total_clusters': result.tree.total_clusters if result.tree else 0,
                        'max_depth': result.tree.max_depth if result.tree else 0,
                        'llm_calls': result.total_llm_calls
                    }
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                event_queue.put({
                    'type': 'error',
                    'error': str(e)
                })

        # Start build in background thread
        thread = threading.Thread(target=build_index)
        thread.start()

        # Yield start event
        yield {'type': 'start', 'timestamp': time.time()}

        # Yield events from queue until complete or error
        while True:
            try:
                event = event_queue.get(timeout=300)  # 5 minute timeout for clustering
                yield event

                if event['type'] in ('complete', 'error'):
                    break

            except queue.Empty:
                yield {
                    'type': 'error',
                    'error': 'Index building timed out'
                }
                break

        thread.join(timeout=5)

    def query_streaming(
        self,
        query: str,
        top_k: int,
        provider: str,
        model: str
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Query the built index with streaming updates.

        Args:
            query: Search query
            top_k: Number of results to return
            provider: LLM provider
            model: Model name

        Yields:
            Dict events with types: 'start', 'navigation_step', 'result', 'complete', 'error'
        """
        event_queue = queue.Queue()

        def run_query():
            try:
                if not self._retriever or not self._cluster_tree:
                    event_queue.put({
                        'type': 'error',
                        'error': 'No index built. Please build an index first.'
                    })
                    return

                # Check if we need to recreate retriever with different provider/model
                if provider != self._index_provider or model != self._index_model:
                    llm = self.llm_service.get_llm(provider, model)
                    if not llm:
                        event_queue.put({
                            'type': 'error',
                            'error': f'Failed to initialize LLM for {provider}/{model}'
                        })
                        return
                    router = LLMRouter(llm)
                    self._retriever = LLMRetriever(
                        llm_router=router,
                        cluster_tree=self._cluster_tree,
                        config=RetrievalConfig(top_k=top_k, confidence_threshold=0.5)
                    )

                # Run retrieval
                response = self._retriever.retrieve(query, top_k=top_k)

                # Emit navigation steps
                for step in response.navigation_path:
                    event_queue.put({
                        'type': 'navigation_step',
                        'level': step.level,
                        'cluster_name': step.cluster_name,
                        'cluster_id': step.cluster_id,
                        'confidence': step.confidence,
                        'reasoning': step.reasoning,
                        'alternatives_considered': step.alternatives_considered
                    })

                # Emit results
                for result in response.results:
                    event_queue.put({
                        'type': 'result',
                        'rank': result.rank,
                        'chunk_id': result.chunk_id,
                        'chunk_text': result.chunk_text,
                        'confidence': result.confidence,
                        'reasoning': result.reasoning,
                        'cluster_path': result.cluster_path
                    })

                # Emit complete event
                event_queue.put({
                    'type': 'complete',
                    'results': [
                        {
                            'rank': r.rank,
                            'chunk_id': r.chunk_id,
                            'chunk_text': r.chunk_text,
                            'confidence': r.confidence,
                            'reasoning': r.reasoning,
                            'cluster_path': r.cluster_path
                        }
                        for r in response.results
                    ],
                    'navigation_path': [
                        {
                            'level': s.level,
                            'cluster_name': s.cluster_name,
                            'cluster_id': s.cluster_id,
                            'confidence': s.confidence,
                            'reasoning': s.reasoning
                        }
                        for s in response.navigation_path
                    ],
                    'stats': {
                        'total_llm_calls': response.total_llm_calls,
                        'total_time_ms': response.total_time_ms,
                        'explored_clusters': response.explored_clusters,
                        'total_chunks_evaluated': response.total_chunks_evaluated
                    }
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                event_queue.put({
                    'type': 'error',
                    'error': str(e)
                })

        # Start query in background thread
        thread = threading.Thread(target=run_query)
        thread.start()

        # Yield start event
        yield {'type': 'start', 'timestamp': time.time()}

        # Yield events from queue until complete or error
        while True:
            try:
                event = event_queue.get(timeout=120)  # 2 minute timeout
                yield event

                if event['type'] in ('complete', 'error'):
                    break

            except queue.Empty:
                yield {
                    'type': 'error',
                    'error': 'Query timed out'
                }
                break

        thread.join(timeout=5)

    def _serialize_tree(self, tree) -> Dict[str, Any]:
        """Serialize ClusterTree to JSON-compatible dict for visualization."""
        if not tree:
            return {}

        nodes = []
        edges = []

        # Build nodes and edges from tree
        for cluster_id, cluster in tree.clusters_by_id.items():
            is_root = cluster_id in tree.root_cluster_ids

            # Get level from clusters_by_level
            level = 0
            for lvl, cluster_ids in tree.clusters_by_level.items():
                if cluster_id in cluster_ids:
                    level = lvl
                    break

            node = {
                'id': cluster_id,
                'name': cluster.metadata.canonical_name if cluster.metadata else f'Cluster {cluster_id}',
                'description': cluster.metadata.description[:200] if cluster.metadata and cluster.metadata.description else '',
                'keywords': cluster.metadata.canonical_keywords[:5] if cluster.metadata else [],
                'tags': cluster.metadata.canonical_tags[:3] if cluster.metadata else [],
                'chunk_count': cluster.chunk_count,
                'is_root': is_root,
                'is_leaf': cluster.is_leaf(),
                'level': level
            }
            nodes.append(node)

            # Add edges to children
            if hasattr(cluster, 'child_clusters') and cluster.child_clusters:
                for child_id in cluster.child_clusters:
                    edges.append({
                        'from': cluster_id,
                        'to': child_id
                    })

        return {
            'nodes': nodes,
            'edges': edges,
            'root_ids': tree.root_cluster_ids,
            'max_depth': tree.max_depth,
            'total_clusters': tree.total_clusters,
            'total_chunks': tree.total_chunks
        }

    def clear_index(self):
        """Clear the current index."""
        self._cluster_tree = None
        self._chunks = None
        self._retriever = None
        self._index_provider = None
        self._index_model = None

    def has_index(self) -> bool:
        """Check if an index is built."""
        return self._retriever is not None


# Singleton instance
retrieval_service = RetrievalService()
