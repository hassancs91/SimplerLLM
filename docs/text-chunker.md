# Text Chunker

Split text into chunks using size limits, sentences, paragraphs, or semantic similarity.

## Basic Usage

```python
from SimplerLLM.tools.text_chunker import chunk_by_max_chunk_size

chunks = chunk_by_max_chunk_size(text="Your long text here...", max_chunk_size=500)

print(chunks.num_chunks)
print(chunks.chunk_list[0].text)
```

## Chunking Strategies

| Strategy | Function | Speed | API Calls | Best For |
|----------|----------|-------|-----------|----------|
| Max Size | `chunk_by_max_chunk_size()` | Very fast | None | Consistent chunk sizes, token limits |
| Sentences | `chunk_by_sentences()` | Fast | None | Grammatically complete chunks |
| Paragraphs | `chunk_by_paragraphs()` | Fast | None | Structured documents with clear paragraphs |
| Semantics | `chunk_by_semantics()` | Slow | Yes | Topic-based chunking, RAG systems |

## By Max Chunk Size

Split text into chunks of a maximum character count:

```python
from SimplerLLM.tools.text_chunker import chunk_by_max_chunk_size

# Fixed-size chunks
chunks = chunk_by_max_chunk_size(text="Your long text...", max_chunk_size=500)

# Preserve sentence boundaries
chunks = chunk_by_max_chunk_size(
    text="Your long text...",
    max_chunk_size=500,
    preserve_sentence_structure=True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | The input text to split |
| `max_chunk_size` | `int` | — | Maximum characters per chunk |
| `preserve_sentence_structure` | `bool` | `False` | Respect sentence endings when splitting |

> **Note:** When `preserve_sentence_structure=True`, a single sentence longer than `max_chunk_size` is kept as one chunk.

## By Sentences

Split text at sentence boundaries:

```python
from SimplerLLM.tools.text_chunker import chunk_by_sentences

chunks = chunk_by_sentences(text="First sentence. Second sentence! Third?")

for chunk in chunks.chunk_list:
    print(chunk.text)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The input text to split |

## By Paragraphs

Split text at paragraph boundaries:

```python
from SimplerLLM.tools.text_chunker import chunk_by_paragraphs

chunks = chunk_by_paragraphs(text="First paragraph.\nSecond paragraph.\nThird paragraph.")

print(chunks.num_chunks)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The input text to split |

## By Semantics

Split text based on semantic similarity using embeddings. Groups related sentences together:

```python
from SimplerLLM.tools.text_chunker import chunk_by_semantics
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI)

chunks = chunk_by_semantics(
    text="Your long text...",
    llm_embeddings_instance=embeddings,
    threshold_percentage=90
)

for chunk in chunks.chunk_list:
    print(chunk.text[:100])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | The input text to split |
| `llm_embeddings_instance` | `EmbeddingsLLM` | — | An embeddings instance for computing similarity |
| `threshold_percentage` | `int` | `90` | Percentile threshold for breakpoints (higher = more chunks) |

> **Note:** This method makes API calls to generate embeddings, which adds cost and latency.

## Response Format

All functions return a `TextChunks` object:

```python
chunks = chunk_by_sentences(text="First sentence. Second sentence.")

# TextChunks
print(chunks.num_chunks)           # 2

# ChunkInfo
chunk = chunks.chunk_list[0]
print(chunk.text)                  # "First sentence."
print(chunk.num_characters)        # 15
print(chunk.num_words)             # 2
```

| Field | Type | Description |
|-------|------|-------------|
| `TextChunks.num_chunks` | `int` | Total number of chunks |
| `TextChunks.chunk_list` | `List[ChunkInfo]` | List of individual chunks |
| `ChunkInfo.text` | `str` | The chunk text |
| `ChunkInfo.num_characters` | `int` | Number of characters in the chunk |
| `ChunkInfo.num_words` | `int` | Number of words in the chunk |
