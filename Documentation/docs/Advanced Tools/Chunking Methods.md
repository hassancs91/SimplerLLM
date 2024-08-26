---
sidebar_position: 4
--- 

# Chunking Methods

This section provides detailed information on the text chunking capabilities of the SimplerLLM library. These functions allow users to split text into pieces based on sentence, paragraph, size, or semantic similarity.

Each method is designed to accommodate different analytical needs, enhancing text processing tasks in various applications such as data preprocessing, content analysis, and information retrieval.

The Data by all these functions is returned in form of a `Text Chunks` object that includes the following parameters:
- `chunk_list` (List): This is a list of `ChunkInfo` objects that includes:
    - `text` (string): The text of the chunk itself.
    - `num_characters` (string): The number of characters in the chunk.
    - `num_words` (string): The number of words in the chunk.
- `num_chunks` (int): The total number of chunks returned.

## chunk_by_sentences Function

Breaks down the provided text into sentences using punctuation marks as delimiters. It takes 1 parameter which is:
- `text` (str): Text you want to chunk into sentences.

It then returns a `Text Chunks` object. Here's a sample usage:

```python
from SimplerLLM.tools.text_chunker import chunk_by_sentences

text = "First sentence. Second sentence? Third sentence!"

sentences = chunk_by_sentences(text)

print(sentences)
```

## chunk_by_paragraphs Function

Segments the provided text into paragraphs based on newline characters. It takes 1 parameter:
- `text` (str): Text you want to chunk into paragraphs.

It then returns a `Text Chunks` object. Here's a sample usage:

```python
from SimplerLLM.tools.text_chunker import chunk_by_paragraphs

text = "First paragraph, still going.\n\nSecond paragraph starts."

paragraphs = chunk_by_paragraphs(text)

print(paragraphs)
```

## chunk_by_max_chunk_size Function

Splits the input text into chunks that do not exceed a specified size. Additionally, it can preserve the meaning of sentences by ensuring that chunks do not split sentences in the middle. It takes 3 parameters:
- `text` (str): The text you want to chunk.
- `max_chunk_size` (int): The maximum size of each chunk in characters.
- `preserve_sentence_structure` (bool, optional): Whether you want to preserve sentence meaning. Set to False by default.

It returns a `Text Chunks` object. Here's how you can use it:

```python
from SimplerLLM.tools.text_chunker import chunk_by_max_chunk_size

text = "Hello world! This is an example of text chunking. Enjoy using SimplerLLM."

chunks = chunk_by_max_chunk_size(text, 50, True)

print(chunks)
```

## chunk_by_semantics Function

Uses semantic similarity to divide text into chunks. It takes 2 parameters:
- `text` (str): Text to be segmented based on semantic content.
- `llm_embeddings_instance` (EmbeddingsLLM): An instance of a language model used to generate text embeddings for semantic analysis.
- `threshold_percentage` (int, Optional): The percentile threshold you want to use to chunk the text. It is set by default to 90.

It returns a list of `ChunkInfo` objects, each representing a semantically coherent segment of the original text. Here's an example of how to use it:

```python
from SimplerLLM.tools.text_chunker import chunk_by_semantics
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

text = "Discussing AI. Artificial intelligence has many applications. However, Dogs like bones"
embeddings_model = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI,
                                        model_name="text-embedding-3-small"
                                        threshold_percentage=80) 

semantic_chunks = chunk_by_semantics(text, embeddings_model)

print(semantic_chunks)
```
That's how you can benefit from SimplerLLM to make Text Chunking Simpler!