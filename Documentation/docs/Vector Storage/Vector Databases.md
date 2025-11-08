---
sidebar_position: 2
---

# Vector Databases

Store, index, and query embeddings at scale using SimplerLLM's unified vector database interface.

## What Are Vector Databases?

Vector databases are specialized databases designed to store and efficiently search high-dimensional vectors (embeddings). They enable:

- **Similarity Search:** Find semantically similar content in milliseconds
- **Scalability:** Handle millions of vectors efficiently
- **Metadata Filtering:** Combine vector similarity with traditional filters
- **Real-time Updates:** Add, update, or delete vectors on the fly
- **RAG Applications:** Power retrieval-augmented generation systems

## Supported Vector Databases

### Local (In-Memory)

Simple in-memory vector storage:
- No external dependencies
- Perfect for prototyping and small datasets
- Fast for development
- Supports persistence to disk

### Qdrant

Production-ready vector database:
- Scalable and performant
- Advanced filtering capabilities
- Cloud or self-hosted
- Built-in persistence

## Quick Start: Local Vector Database

The local vector database is perfect for development and testing:

```python
from SimplerLLM.vectors.vector_db import VectorDB
from SimplerLLM.vectors.vector_providers import VectorProvider
from SimplerLLM.language.embeddings import EmbeddingsOpenAI
import numpy as np

# Create embeddings instance
embeddings = EmbeddingsOpenAI()

# Create local vector database with factory pattern
vector_db = VectorDB.create(
    provider=VectorProvider.LOCAL,
    db_folder='./my_vectors',
    dimension=1536  # OpenAI embedding dimension
)

# Add documents
documents = [
    "SimplerLLM makes AI development easy",
    "Python is a popular programming language",
    "Vector databases enable semantic search"
]

# Store documents with embeddings
for doc in documents:
    embedding = embeddings.generate_embeddings(doc)
    vector_db.add_text_with_embedding(
        text=doc,
        embedding=embedding,
        normalize=True
    )

# Search for similar documents using text
query = "How to build AI applications?"
results = vector_db.search_by_text(query, embeddings, top_n=2)

for vector_id, metadata, similarity in results:
    print(f"Text: {metadata['text']}")
    print(f"Similarity: {similarity:.4f}\n")
```

## Qdrant Vector Database

Qdrant is a production-ready vector database with advanced features.

### Setup

The Qdrant client (`qdrant-client`) is automatically installed with SimplerLLM, so no additional installation is needed.

To run Qdrant locally with Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Basic Usage

```python
from SimplerLLM.vectors.vector_db import VectorDB
from SimplerLLM.vectors.vector_providers import VectorProvider
from SimplerLLM.language.embeddings import EmbeddingsOpenAI

# Create embeddings instance
embeddings = EmbeddingsOpenAI()

# Create Qdrant vector database with factory pattern
vector_db = VectorDB.create(
    provider=VectorProvider.QDRANT,
    collection_name='my_documents',
    url='localhost',
    port=6333,
    dimension=1536  # OpenAI embedding dimension
)

# Add documents
documents = [
    {"text": "Machine learning is a subset of AI", "category": "AI"},
    {"text": "Python is great for data science", "category": "Programming"},
    {"text": "Vector search enables semantic similarity", "category": "Database"}
]

for doc in documents:
    embedding = embeddings.generate_embeddings(doc['text'])
    vector_db.add_text_with_embedding(
        text=doc['text'],
        embedding=embedding,
        metadata={"category": doc['category']},
        normalize=True
    )

# Search with metadata filtering
query = "artificial intelligence applications"

# Use filter function for category filtering
def filter_by_category(vector_id, metadata):
    return metadata.get('category') == 'AI'

results = vector_db.search_by_text(
    query,
    embeddings,
    top_n=3,
    filter_func=filter_by_category
)

for vector_id, metadata, similarity in results:
    print(f"Text: {metadata['text']}")
    print(f"Category: {metadata['category']}")
    print(f"Similarity: {similarity:.4f}\n")
```

## Advanced Features

### Batch Operations

Efficiently add multiple vectors at once:

```python
from SimplerLLM.vectors.vector_db import VectorDB
from SimplerLLM.vectors.vector_providers import VectorProvider
from SimplerLLM.language.embeddings import EmbeddingsOpenAI

embeddings = EmbeddingsOpenAI()
vector_db = VectorDB.create(
    provider=VectorProvider.QDRANT,
    collection_name='batch_documents',
    url='localhost',
    port=6333,
    dimension=1536
)

# Prepare batch data
documents = [
    "Document 1 about machine learning",
    "Document 2 about data science",
    "Document 3 about artificial intelligence"
]

# Generate embeddings in batch
vectors = [embeddings.generate_embeddings(doc) for doc in documents]

# Add all vectors at once using batch method
vectors_with_meta = [
    (vector, {'id': i, 'text': doc})
    for i, (vector, doc) in enumerate(zip(vectors, documents))
]

added_ids = vector_db.add_vectors_batch(vectors_with_meta, normalize=True)
print(f"Added {len(added_ids)} vectors")
```

### Metadata Filtering

Combine vector similarity with metadata filters:

```python
# Add documents with rich metadata
docs_with_metadata = [
    {
        "text": "Introduction to machine learning",
        "author": "John Doe",
        "date": "2024-01-15",
        "level": "beginner"
    },
    {
        "text": "Advanced deep learning techniques",
        "author": "Jane Smith",
        "date": "2024-02-20",
        "level": "advanced"
    }
]

for doc in docs_with_metadata:
    embedding = embeddings.generate_embeddings(doc['text'])
    vector_db.add_text_with_embedding(
        text=doc['text'],
        embedding=embedding,
        metadata={k: v for k, v in doc.items() if k != 'text'}
    )

# Search with filter function for author and level
def filter_by_author_and_level(vector_id, metadata):
    return (metadata.get('author') == 'John Doe' and
            metadata.get('level') == 'beginner')

results = vector_db.search_by_text(
    "learning algorithms",
    embeddings,
    top_n=5,
    filter_func=filter_by_author_and_level
)

# Or use query_by_metadata for exact matching
matching_vectors = vector_db.query_by_metadata(author='John Doe', level='beginner')
for vector_id, vector, metadata in matching_vectors:
    print(f"Found: {metadata['text']}")
```

### Updating and Deleting Vectors

```python
# Update existing vector by ID
success = vector_db.update_vector(
    vector_id="doc_123",
    new_vector=new_embedding,
    new_metadata={'text': 'Updated document text', 'updated_at': '2024-03-01'},
    normalize=True
)

# Update only metadata
success = vector_db.update_vector(
    vector_id="doc_123",
    new_metadata={'status': 'reviewed'}
)

# Delete vector by ID
success = vector_db.delete_vector(vector_id="doc_456")
if success:
    print("Vector deleted successfully")

# Get information about a specific vector
result = vector_db.get_vector_by_id("doc_123")
if result:
    vector, metadata = result
    print(f"Vector dimension: {len(vector)}")
    print(f"Metadata: {metadata}")
```

## VectorProvider Enum

SimplerLLM uses an enum-based system for selecting vector database providers:

```python
from SimplerLLM.vectors.vector_providers import VectorProvider

# Available providers
print(VectorProvider.LOCAL)    # VectorProvider.LOCAL
print(VectorProvider.QDRANT)   # VectorProvider.QDRANT

# Use in factory method
from SimplerLLM.vectors.vector_db import VectorDB

# Local provider
local_db = VectorDB.create(
    provider=VectorProvider.LOCAL,
    db_folder='./vectors',
    dimension=1536
)

# Qdrant provider
qdrant_db = VectorDB.create(
    provider=VectorProvider.QDRANT,
    collection_name='my_collection',
    url='localhost',
    port=6333,
    dimension=1536
)
```

## Database Statistics

Get standardized statistics across all vector database providers:

```python
from SimplerLLM.vectors.vector_db import VectorDB
from SimplerLLM.vectors.vector_providers import VectorProvider

# Create any vector database
vector_db = VectorDB.create(
    provider=VectorProvider.LOCAL,
    db_folder='./vectors',
    dimension=1536
)

# Add some vectors...
# (vector addition code here)

# Get statistics
stats = vector_db.get_stats()

# All providers return these required fields:
print(f"Total vectors: {stats['total_vectors']}")
print(f"Dimension: {stats['dimension']}")
print(f"Provider: {stats['provider']}")

# Providers may also return additional fields:
# LOCAL provider includes:
#   - size_in_memory_mb: Memory usage in MB
#   - metadata_keys: List of unique metadata keys

# QDRANT provider includes:
#   - collection_name: Name of the collection
#   - vectors_count: Detailed vector count
#   - status: Collection status
```

## Provider Switching

Easily switch between providers thanks to the unified interface:

```python
from SimplerLLM.vectors.vector_db import VectorDB
from SimplerLLM.vectors.vector_providers import VectorProvider
from SimplerLLM.language.embeddings import EmbeddingsOpenAI
import os

embeddings = EmbeddingsOpenAI()

# Determine provider based on environment
USE_PRODUCTION = os.getenv('USE_PRODUCTION', 'false').lower() == 'true'

if USE_PRODUCTION:
    # Use Qdrant for production
    vector_db = VectorDB.create(
        provider=VectorProvider.QDRANT,
        collection_name='my_app',
        url='localhost',
        port=6333,
        dimension=1536
    )
    print("Using Qdrant (production)")
else:
    # Use local for development
    vector_db = VectorDB.create(
        provider=VectorProvider.LOCAL,
        db_folder='./dev_vectors',
        dimension=1536
    )
    print("Using local storage (development)")

# Same code works with both providers!
documents = ["Document 1", "Document 2", "Document 3"]
for doc in documents:
    embedding = embeddings.generate_embeddings(doc)
    vector_db.add_text_with_embedding(doc, embedding)

# Search works the same way
results = vector_db.search_by_text("query", embeddings, top_n=5)

# Save persistence (LOCAL only, gracefully skipped for QDRANT)
try:
    vector_db.save_to_disk('my_collection')
except NotImplementedError:
    print("Persistence not needed for this provider")
```

## Optional Features

Some features are provider-specific and may not be available on all databases:

```python
# LOCAL provider supports:
local_db = VectorDB.create(
    provider=VectorProvider.LOCAL,
    db_folder='./vectors',
    dimension=1536
)

# Save to disk (LOCAL only)
local_db.save_to_disk('my_collection')
local_db.load_from_disk('my_collection')

# Compress vectors to save memory (LOCAL only)
compression_ratio = local_db.compress_vectors(bits=16)
print(f"Compressed by {compression_ratio:.2f}x")

# QDRANT provider has built-in persistence
# No need for save/load operations

# Attempting unsupported operations raises NotImplementedError:
try:
    qdrant_db.compress_vectors(bits=16)
except NotImplementedError as e:
    print(f"Feature not supported: {e}")
```

## Error Handling

SimplerLLM provides custom exceptions for proper error handling:

```python
from SimplerLLM.vectors.vector_db import (
    VectorDB,
    VectorDBError,
    DimensionMismatchError,
    VectorDBConnectionError,
    VectorDBOperationError,
    VectorNotFoundError
)
from SimplerLLM.vectors.vector_providers import VectorProvider
from SimplerLLM.language.embeddings import EmbeddingsOpenAI
import numpy as np

embeddings = EmbeddingsOpenAI()

try:
    # Create vector database
    vector_db = VectorDB.create(
        provider=VectorProvider.QDRANT,
        collection_name='test_collection',
        url='localhost',
        port=6333,
        dimension=1536
    )

    # Add vector
    embedding = embeddings.generate_embeddings("Sample text")
    vector_id = vector_db.add_vector(
        vector=embedding,
        meta={'text': 'Sample text'},
        normalize=True
    )
    print(f"Added vector with ID: {vector_id}")

    # Search - dimension mismatch example
    wrong_vector = np.array([1.0, 0.0], dtype=np.float32)  # Wrong dimension
    results = vector_db.top_cosine_similarity(wrong_vector, top_n=5)

except VectorDBConnectionError as e:
    print(f"Failed to connect to vector database: {e}")
    print("Make sure Qdrant is running on localhost:6333")

except DimensionMismatchError as e:
    print(f"Dimension mismatch: {e}")
    print("Ensure all vectors have the same dimension")

except VectorDBOperationError as e:
    print(f"Operation failed: {e}")

except VectorNotFoundError as e:
    print(f"Vector not found: {e}")

except VectorDBError as e:
    print(f"Vector database error: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Real-World Example: RAG System

Build a Retrieval-Augmented Generation (RAG) system:

```python
from SimplerLLM.vectors.vector_db import VectorDB
from SimplerLLM.vectors.vector_providers import VectorProvider
from SimplerLLM.language.embeddings import EmbeddingsOpenAI
from SimplerLLM.language.llm import LLM, LLMProvider

class RAGSystem:
    def __init__(self):
        # Initialize components
        self.embeddings = EmbeddingsOpenAI()
        self.vector_db = VectorDB.create(
            provider=VectorProvider.QDRANT,
            collection_name='knowledge_base',
            url='localhost',
            port=6333,
            dimension=1536
        )
        self.llm = LLM.create(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o"
        )

    def add_knowledge(self, documents):
        """Add documents to knowledge base"""
        for doc in documents:
            embedding = self.embeddings.generate_embeddings(doc)
            self.vector_db.add_text_with_embedding(
                text=doc,
                embedding=embedding,
                normalize=True
            )

    def query(self, question, top_n=3):
        """Answer question using RAG"""
        # 1. Find relevant documents using search_by_text
        results = self.vector_db.search_by_text(
            question,
            self.embeddings,
            top_n=top_n
        )

        # 2. Build context from results
        context = "\n\n".join([
            f"Document {i+1}: {metadata['text']}"
            for i, (vector_id, metadata, similarity) in enumerate(results)
        ])

        # 3. Generate answer with context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

        answer = self.llm.generate_response(prompt=prompt)
        return answer, results

# Usage
rag = RAGSystem()

# Add knowledge to the system
knowledge = [
    "SimplerLLM is a Python library for working with LLMs",
    "SimplerLLM supports multiple providers including OpenAI and Anthropic",
    "Vector databases enable semantic search in RAG systems"
]
rag.add_knowledge(knowledge)

# Query the system
question = "What is SimplerLLM?"
answer, sources = rag.query(question)

print(f"Answer: {answer}\n")
print("Sources used:")
for i, (vector_id, metadata, similarity) in enumerate(sources, 1):
    print(f"{i}. {metadata['text']} (similarity: {similarity:.4f})")
```

## Choosing a Vector Database

### Use Local when:
- Prototyping or development
- Working with small datasets (<10k vectors)
- No external dependencies desired
- Need disk persistence for development

### Use Qdrant when:
- Building production applications
- Need to scale beyond memory limits
- Require advanced filtering and search
- Need persistence and reliability
- Working with large datasets (>10k vectors)

## Performance Optimization

1. **Choose the Right Index:** Different index types (HNSW, IVF, etc.) offer tradeoffs between speed and accuracy. Configure based on your needs.

2. **Batch Operations:** Use batch operations when adding or updating multiple vectors to reduce overhead.

3. **Filter Early:** Apply metadata filters before vector search when possible to reduce the search space.

4. **Monitor Memory Usage:** Vector databases can be memory-intensive. Monitor usage and scale appropriately for production.

## Best Practices

### Development vs Production
- **Development:** Use local vector database for rapid prototyping and testing
- **Production:** Use Qdrant or other production databases for scalability and reliability
- **Consistency:** Use the same embedding model throughout your application
- **Indexing:** Create appropriate indexes on metadata fields used for filtering
- **Backups:** Regularly backup your vector database in production environments

### Vector Dimensions
- Always specify the correct dimension when creating a database
- All vectors in a collection must have the same dimension
- OpenAI text-embedding-ada-002: 1536 dimensions
- OpenAI text-embedding-3-small: 1536 dimensions
- OpenAI text-embedding-3-large: 3072 dimensions

### Normalization
- Cosine similarity requires normalized vectors
- SimplerLLM automatically normalizes vectors when `normalize=True`
- Always normalize for semantic search applications

## API Reference

### Core Methods (All Providers)

#### `add_vector(vector, meta, normalize=True, id=None)`
Add a single vector with metadata.

#### `add_vectors_batch(vectors_with_meta, normalize=False)`
Add multiple vectors in batch.

#### `add_text_with_embedding(text, embedding, metadata=None, normalize=True, id=None)`
Add text content along with its embedding.

#### `delete_vector(vector_id)`
Delete a vector by its ID.

#### `update_vector(vector_id, new_vector=None, new_metadata=None, normalize=True)`
Update a vector or its metadata.

#### `top_cosine_similarity(target_vector, top_n=3, filter_func=None)`
Find most similar vectors using cosine similarity.

#### `search_by_text(query_text, embeddings_llm_instance, top_n=3, filter_func=None)`
Search using text query (converts to embedding internally).

#### `query_by_metadata(**kwargs)`
Query vectors by metadata fields.

#### `get_vector_by_id(vector_id)`
Retrieve a specific vector and its metadata.

#### `list_all_ids()`
Get all vector IDs in the database.

#### `get_vector_count()`
Get the total number of vectors.

#### `clear_database()`
Remove all vectors from the database.

#### `get_stats()`
Get statistics about the database.

### Optional Methods (Provider-Specific)

#### `save_to_disk(collection_name)` (LOCAL only)
Save the database to disk.

#### `load_from_disk(collection_name)` (LOCAL only)
Load the database from disk.

#### `compress_vectors(bits=16)` (LOCAL only)
Compress vectors to lower precision.

## Next Steps

- Learn about [Vector Embeddings](./Vector%20Embeddings.md)
- Explore [Text Chunking](../Tools/Text%20Chunking.md) for RAG systems
- Check out [Content Loading](../Tools/Content%20Loading.md) for multi-source RAG
