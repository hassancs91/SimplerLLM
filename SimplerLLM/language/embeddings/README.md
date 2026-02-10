# Embeddings - Multi-Provider Text Embeddings

SimplerLLM Embeddings provides a unified interface for generating text embeddings using multiple providers (OpenAI, Voyage AI, Cohere). Perfect for semantic search, RAG systems, clustering, and similarity detection.

## Features

- **Three Providers**: OpenAI, Voyage AI, and Cohere support
- **Sync & Async**: Both synchronous and asynchronous methods
- **Factory Pattern**: Simple `EmbeddingsLLM.create()` interface
- **Provider-Specific Features**: input_type optimization, custom dimensions, truncation
- **Structured Response**: Optional metadata with `LLMEmbeddingsResponse`
- **Retry Logic**: Built-in retry with exponential backoff
- **Batch Support**: Embed multiple texts in a single call

## Quick Start

```python
from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider

# Create embeddings instance
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-small"
)

# Generate embedding for a single text
vector = embeddings.generate_embeddings("Machine learning is fascinating")
print(f"Embedding dimension: {len(vector)}")  # 1536

# Generate embeddings for multiple texts
texts = ["First document", "Second document", "Third document"]
vectors = embeddings.generate_embeddings(texts)
print(f"Generated {len(vectors)} embeddings")
```

## Supported Providers

### OpenAI

```python
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-small"  # or "text-embedding-3-large"
)

vector = embeddings.generate_embeddings("Your text here")
```

**Models:**

| Model | Dimensions | Notes |
|-------|------------|-------|
| text-embedding-3-small | 1536 | Fast, cost-effective (default) |
| text-embedding-3-large | 3072 | Highest quality |
| text-embedding-ada-002 | 1536 | Legacy model |

### Voyage AI

```python
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.VOYAGE,
    model_name="voyage-3"
)

# Optimized for retrieval
query_emb = embeddings.generate_embeddings(
    "search query",
    input_type="query"
)
doc_emb = embeddings.generate_embeddings(
    "document to index",
    input_type="document"
)

# Custom dimension
small_emb = embeddings.generate_embeddings(
    "text",
    output_dimension=512  # 256, 512, 1024, or 2048
)
```

**Models:**

| Model | Use Case |
|-------|----------|
| voyage-3 | General purpose (default) |
| voyage-3-lite | Fast, cost-effective |
| voyage-code-3 | Code embeddings |
| voyage-finance-2 | Financial domain |

**Parameters:**

- `input_type`: "query" or "document" for retrieval optimization
- `output_dimension`: 256, 512, 1024, or 2048
- `output_dtype`: "float", "int8", "uint8", "binary", "ubinary"

### Cohere

```python
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.COHERE,
    model_name="embed-english-v3.0"
)

# For documents to be searched
doc_emb = embeddings.generate_embeddings(
    "Document content here",
    input_type="search_document"
)

# For search queries
query_emb = embeddings.generate_embeddings(
    "What is AI?",
    input_type="search_query"
)

# Handle long texts
long_emb = embeddings.generate_embeddings(
    very_long_text,
    truncate="END"  # "START", "END", or "NONE"
)
```

**Models:**

| Model | Use Case |
|-------|----------|
| embed-english-v3.0 | English text (default) |
| embed-multilingual-v3.0 | 100+ languages |
| embed-v4.0 | Latest model |

**Parameters:**

- `input_type`: "search_document", "search_query", "classification", "clustering"
- `truncate`: "START", "END", or "NONE"
- `embedding_types`: List of specific embedding types

## Full Response with Metadata

```python
response = embeddings.generate_embeddings(
    "Your text here",
    full_response=True
)

print(f"Embedding: {response.generated_embedding[:5]}...")
print(f"Model: {response.model}")
print(f"Time: {response.process_time:.3f}s")
print(f"Raw response: {response.llm_provider_response}")
```

## Async Usage

```python
import asyncio

async def embed_texts():
    embeddings = EmbeddingsLLM.create(
        provider=EmbeddingsProvider.OPENAI,
        model_name="text-embedding-3-small"
    )

    # Single async call
    vector = await embeddings.generate_embeddings_async("Text to embed")

    # Parallel async calls
    texts = ["Text 1", "Text 2", "Text 3"]
    tasks = [embeddings.generate_embeddings_async(t) for t in texts]
    vectors = await asyncio.gather(*tasks)

    return vectors

vectors = asyncio.run(embed_texts())
```

## Use Cases

### 1. Semantic Search

```python
from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider
import numpy as np

embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.VOYAGE,
    model_name="voyage-3"
)

# Index documents
documents = ["Doc about AI", "Doc about cooking", "Doc about travel"]
doc_vectors = [
    embeddings.generate_embeddings(doc, input_type="document")
    for doc in documents
]

# Search
query = "machine learning"
query_vector = embeddings.generate_embeddings(query, input_type="query")

# Find most similar (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(query_vector, doc) for doc in doc_vectors]
best_match_idx = np.argmax(similarities)
print(f"Best match: {documents[best_match_idx]}")
```

### 2. RAG System Integration

```python
from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider
from SimplerLLM.vectors import VectorDB, VectorProvider

# Create embeddings
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-small"
)

# Create vector database
db = VectorDB.create(provider=VectorProvider.QDRANT, collection_name="my_docs")

# Store documents
for doc in documents:
    vector = embeddings.generate_embeddings(doc)
    db.insert(vector=vector, metadata={"text": doc})

# Query
query_vector = embeddings.generate_embeddings(user_query)
results = db.search(query_vector, limit=5)
```

### 3. Text Clustering

```python
from sklearn.cluster import KMeans

texts = ["...", "...", "..."]  # Your texts
vectors = embeddings.generate_embeddings(texts)

# Cluster into 3 groups
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(vectors)
```

### 4. Similarity Detection

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

text1 = "The cat sat on the mat."
text2 = "A feline rested on the rug."
text3 = "The stock market crashed today."

vec1 = embeddings.generate_embeddings(text1)
vec2 = embeddings.generate_embeddings(text2)
vec3 = embeddings.generate_embeddings(text3)

print(f"Similarity (1-2): {cosine_similarity(vec1, vec2):.4f}")  # High
print(f"Similarity (1-3): {cosine_similarity(vec1, vec3):.4f}")  # Low
```

## Environment Variables

```bash
# Required for respective providers
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...
COHERE_API_KEY=...
```

## API Reference

### EmbeddingsLLM.create()

```python
@staticmethod
def create(
    provider: EmbeddingsProvider,  # OPENAI, VOYAGE, or COHERE (required)
    model_name: str = None,        # Provider-specific model (optional)
    api_key: str = None,           # API key (optional, falls back to env var)
    user_id: str = None            # Optional tracking ID
) -> BaseEmbeddings
```

### generate_embeddings()

```python
def generate_embeddings(
    user_input: Union[str, List[str]],  # Text(s) to embed
    model_name: str = None,              # Model override
    full_response: bool = False,         # Return metadata
    # Provider-specific (Voyage):
    input_type: str = None,              # "query" or "document"
    output_dimension: int = None,        # 256, 512, 1024, 2048
    output_dtype: str = "float",         # Data type
    # Provider-specific (Cohere):
    input_type: str = "search_document", # Semantic context
    embedding_types: List = None,        # Specific types
    truncate: str = "END"                # Truncation mode
) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]
```

### generate_embeddings_async()

Same parameters as `generate_embeddings()`, returns awaitable.

## Best Practices

1. **Batch for efficiency**: Embed multiple texts in one call when possible
2. **Use input_type**: For Voyage/Cohere, use "query" for searches and "document" for indexing
3. **Cache embeddings**: Store computed embeddings to avoid re-computation
4. **Choose dimensions wisely**: Smaller dimensions = faster search, lower storage
5. **Handle rate limits**: The built-in retry logic handles transient errors

## Module Structure

```
SimplerLLM/language/embeddings/
├── __init__.py      # Module exports and documentation
├── base.py          # EmbeddingsLLM factory class
├── providers.py     # Provider implementations
├── models.py        # EmbeddingsProvider enum
└── README.md        # This documentation
```

## Troubleshooting

### Import Error

```python
# If you get: cannot import 'EmbeddingsLLM'
# Make sure you're using the correct import:
from SimplerLLM.language import EmbeddingsLLM, EmbeddingsProvider
```

### API Key Missing

```python
# Set via environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# Or pass directly
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    api_key="your-key"
)
```

### Voyage Not Available

```bash
# Install voyageai package
pip install voyageai
```

### Cohere v5 Required

```bash
# Cohere requires version 5.0+
pip install cohere>=5.0
```
