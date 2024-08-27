---
sidebar_position: 1
--- 

# Vector Embeddings

This component of the SimplerLLM Library facilitates the generation of text embeddings using OpenAI's embedding model for now. This functionality helps developers working on natural language processing applications, enabling them to easily generate embeddings for a variety of use cases such as [semantic chunking](https://docs.simplerllm.com/Advanced%20Tools/Chunking%20Methods#chunk_by_semantics-function), clustering in machine learning, finding semantic similarity, etc...

Before using this function make sure that your environment is set up with the necessary API keys. Place your OpenAI API key in the `.env` file as shown below:

```
OPENAI_API_KEY="your_openai_api_key"
```

## Using the `EmbeddingsLLM` Class

The `EmbeddingsLLM` class is designed to handle the generation of text embeddings, supporting both synchronous and asynchronous operations.

Start by creating an instance of the `EmbeddingsLLM` class entering the provider and the model you wish to use, like this:

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

embeddings_llm_instance = EmbeddingsLLM.create(EmbeddingsProvider.OPENAI, "text-embedding-3-small") 
```

### Generating Embeddings Synchronously

The `generate_embeddings` method allows you to generate embeddings synchronously. It takes 2 parameters:
- `user_input`: String or list of strings for which embeddings are required.
- `full_response` (Optional): Boolean indicating whether to return the full API response. It's set by default to false where it returns only the text embeddings, not the full API response.

Here's an example of generating embeddings for a list of strings:

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

texts = ["Hello World", "Discussing AI.", "Artificial intelligence has many applications."]
embeddings_llm_instance = EmbeddingsLLM.create(EmbeddingsProvider.OPENAI, "text-embedding-3-small") 

embeddings = embeddings_llm_instance.generate_embeddings(texts)

print(embeddings)
```

### Generating Embeddings Asynchronously

For applications that benefit from non-blocking operations, use the `generate_embeddings_async` method to perform asynchronous embeddings generation. It also takes the same 2 parameters:
- `user_input`: String or list of strings for which embeddings are required.
- `full_response` (Optional): Boolean indicating whether to return the full API response. It's set by default to false where it returns only the text embeddings, not the full API response.

Generating embeddings asynchronously for a list of strings:

```python
import asyncio
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

async def generate_async_embeddings():
    texts = ["Hello World", "Discussing AI.", "Artificial intelligence has many applications."]
    embeddings_llm_instance = EmbeddingsLLM.create(EmbeddingsProvider.OPENAI, "text-embedding-3-small")
    
    tasks = [embeddings_llm_instance.generate_embeddings_async(text) for text in texts]
    
    embeddings = await asyncio.gather(*tasks)

    print(embeddings)

asyncio.run(generate_async_embeddings())
```

This method allows your application to remain responsive while processing multiple embedding requests simultaneously.

---

That's how you can benefit from SimplerLLM to make Vector Embeddings Generation Simpler!
