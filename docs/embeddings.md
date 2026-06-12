# Embeddings

Generate text embeddings with OpenAI, Voyage AI, Cohere, or any model through OpenRouter or CometAPI.

## Basic Usage

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)

vector = embeddings.generate_embeddings("What is machine learning?")
print(len(vector))  # 1536
```

## Providers

| Provider | Enum Value | Default Model | Dimensions |
|----------|-----------|---------------|------------|
| OpenAI | `EmbeddingsProvider.OPENAI` | `text-embedding-3-small` | 1536 |
| Voyage AI | `EmbeddingsProvider.VOYAGE` | `voyage-3` | 1024 |
| Cohere | `EmbeddingsProvider.COHERE` | `embed-english-v3.0` | 1024 |
| OpenRouter | `EmbeddingsProvider.OPENROUTER` | `openai/text-embedding-3-small` | 1536 |
| CometAPI | `EmbeddingsProvider.COMETAPI` | `text-embedding-3-small` | 1536 |

## Batch Embeddings

Pass a list of strings to embed multiple texts in a single API call:

```python
texts = [
    "What is Python?",
    "How does machine learning work?",
    "Explain neural networks"
]

vectors = embeddings.generate_embeddings(texts)
print(len(vectors))     # 3
print(len(vectors[0]))  # 1536
```

## Full Response with Metadata

Set `full_response=True` to get timing and model info:

```python
response = embeddings.generate_embeddings(
    "What is deep learning?",
    full_response=True
)

print(len(response.generated_embedding))  # 1536
print(f"Model: {response.model}")
print(f"Time: {response.process_time:.2f}s")
```

## Switching Providers

All providers use the same `generate_embeddings()` method:

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

# OpenAI
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)

# Voyage AI
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.VOYAGE)

# Cohere
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.COHERE)

# OpenRouter
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENROUTER)

# CometAPI
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.COMETAPI)
```

To use a specific model:

```python
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-large"  # 3072 dimensions
)
```

## Voyage AI

Voyage AI supports input type optimization and custom dimensions:

```python
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.VOYAGE)

# Optimize for search queries
query_vector = embeddings.generate_embeddings(
    "How does Python async work?",
    input_type="query"
)

# Optimize for documents being indexed
doc_vector = embeddings.generate_embeddings(
    "Python's asyncio module provides infrastructure for writing...",
    input_type="document"
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_type` | `str` | `"query"` or `"document"` — optimizes for retrieval |
| `output_dimension` | `int` | `256`, `512`, `1024`, or `2048` |
| `output_dtype` | `str` | `"float"`, `"int8"`, `"uint8"`, `"binary"`, `"ubinary"` |

Available models: `voyage-3`, `voyage-3-lite`, `voyage-code-3`, `voyage-finance-2`.

## Cohere

Cohere supports multiple input types for different use cases:

```python
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.COHERE)

# For documents being indexed
doc_vector = embeddings.generate_embeddings(
    "Python is a programming language...",
    input_type="search_document"
)

# For search queries
query_vector = embeddings.generate_embeddings(
    "What is Python?",
    input_type="search_query"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_type` | `str` | `"search_document"` | `"search_document"`, `"search_query"`, `"classification"`, `"clustering"` |
| `truncate` | `str` | `"END"` | `"START"`, `"END"`, or `"NONE"` |

Available models: `embed-english-v3.0`, `embed-multilingual-v3.0`, `embed-v4.0`.

## OpenRouter

OpenRouter gives access to embedding models from multiple providers through a single API key (`OPENROUTER_API_KEY`). Model names use the `provider/model` format:

```python
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENROUTER,
    model_name="openai/text-embedding-3-small"
)

vector = embeddings.generate_embeddings("What is machine learning?")
print(len(vector))  # 1536

# Switch models without changing API keys
vector = embeddings.generate_embeddings(
    "What is machine learning?",
    model_name="qwen/qwen3-embedding-8b"
)
```

Example models: `openai/text-embedding-3-small`, `openai/text-embedding-3-large`, `qwen/qwen3-embedding-8b`. See the [OpenRouter embedding models list](https://openrouter.ai/collections/embedding-models) for all available models.

## CometAPI

CometAPI gives access to embedding models through a single API key (`COMETAPI_API_KEY` or `COMETAPI_KEY`). Model names use their native format without a prefix:

```python
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.COMETAPI,
    model_name="text-embedding-3-small"
)

vector = embeddings.generate_embeddings("What is machine learning?")
print(len(vector))  # 1536

# Larger model through the same key
vector = embeddings.generate_embeddings(
    "What is machine learning?",
    model_name="text-embedding-3-large"
)
```

Example models: `text-embedding-3-small`, `text-embedding-3-large`. See [CometAPI models](https://www.cometapi.com/models) for all available models.

## Async Usage

```python
import asyncio
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)

async def main():
    vector = await embeddings.generate_embeddings_async("What is AI?")
    print(len(vector))

asyncio.run(main())
```
