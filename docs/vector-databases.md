# Vector Databases

Store and search vectors with Local or Qdrant backends.

## Basic Usage

```python
from SimplerLLM.vectors import VectorDB, VectorProvider

db = VectorDB.create(provider=VectorProvider.LOCAL, db_folder="./vectors")

# Add a vector with metadata
vector_id = db.add_vector(
    vector=[0.1, 0.2, 0.3, 0.4],
    meta={"text": "What is Python?", "source": "faq"}
)

# Search for similar vectors
results = db.top_cosine_similarity(
    target_vector=[0.1, 0.2, 0.3, 0.4],
    top_n=3
)

for vid, metadata, score in results:
    print(f"Score: {score:.3f} — {metadata['text']}")
```

## Providers

| Provider | Enum Value | Description |
|----------|-----------|-------------|
| Local | `VectorProvider.LOCAL` | In-memory with disk persistence (.svdb files) |
| Qdrant | `VectorProvider.QDRANT` | Self-hosted or Qdrant Cloud |

## Adding Vectors

### Single Vector

```python
vector_id = db.add_vector(
    vector=[0.1, 0.2, 0.3],
    meta={"text": "Example document", "category": "science"}
)
```

### Batch

```python
vectors_with_meta = [
    ([0.1, 0.2, 0.3], {"text": "First document"}),
    ([0.4, 0.5, 0.6], {"text": "Second document"}),
    ([0.7, 0.8, 0.9], {"text": "Third document"}),
]

db.add_vectors_batch(vectors_with_meta)
```

### Text with Embedding

Store the original text alongside its embedding for RAG workflows:

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)
vector = embeddings.generate_embeddings("What is machine learning?")

db.add_text_with_embedding(
    text="What is machine learning?",
    embedding=vector,
    metadata={"source": "faq"}
)
```

## Searching

### Cosine Similarity

```python
results = db.top_cosine_similarity(
    target_vector=query_vector,
    top_n=5
)

for vector_id, metadata, score in results:
    print(f"Score: {score:.3f} — {metadata['text']}")
```

### Text-Based Search

Search by text directly — embeddings are generated automatically:

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)

results = db.search_by_text(
    query_text="How does machine learning work?",
    embeddings_llm_instance=embeddings,
    top_n=5
)

for vector_id, metadata, score in results:
    print(f"Score: {score:.3f} — {metadata['text']}")
```

## Metadata Filtering

### Filter Function on Search

Pass a filter function that receives `(vector_id, metadata)` and returns `True` to include:

```python
results = db.top_cosine_similarity(
    target_vector=query_vector,
    top_n=5,
    filter_func=lambda vid, meta: meta.get("category") == "science"
)
```

### Query by Metadata

Find vectors matching specific metadata fields:

```python
results = db.query_by_metadata(category="science", source="faq")
```

## Persistence

Save and load the local database to disk:

```python
# Save
db.save_to_disk("my_collection")

# Load
db.load_from_disk("my_collection")
```

> **Note:** Files are saved as `.svdb` in the `db_folder` directory.

## Qdrant

### Self-Hosted

```python
db = VectorDB.create(
    provider=VectorProvider.QDRANT,
    url="localhost",
    port=6333,
    collection_name="my_collection",
    dimension=1536
)
```

### Qdrant Cloud

```python
db = VectorDB.create(
    provider=VectorProvider.QDRANT,
    url="your-cluster.qdrant.io",
    port=6333,
    collection_name="my_collection",
    dimension=1536,
    api_key="your-qdrant-api-key"
)
```

All methods (`add_vector`, `top_cosine_similarity`, `search_by_text`, etc.) work the same across both providers.

## Management

```python
# Get total vector count
count = db.get_vector_count()

# Get database statistics
stats = db.get_stats()

# Get a vector by ID
vector_id, vector, metadata = db.get_vector_by_id("some-id")

# List all IDs
ids = db.list_all_ids()

# Delete a vector
db.delete_vector("some-id")

# Clear all vectors
db.clear_database()
```
