# SimplerLLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Join the Discord chat!](https://img.shields.io/badge/Join-Discord-7289DA.svg)](https://discord.gg/HUrtZXyp3j)

**Your Easy Pass to Advanced AI** - A comprehensive Python library for simplified Large Language Model interactions.

## Overview

SimplerLLM is an open-source Python library designed to simplify interactions with Large Language Models (LLMs) for researchers, developers, and AI enthusiasts. It provides a unified interface for multiple LLM providers, robust tools for content processing, and advanced features like reliable failover systems and intelligent routing.

[📚 Full Documentation](https://docs.simplerllm.com/)

## Installation

```bash
pip install simplerllm
```

### Optional Dependencies

For voice/audio features (AudioPlayer file playback):
```bash
pip install simplerllm[voice]
# Or install pygame directly:
pip install pygame>=2.5.0
```

## Key Features

### 🔗 Unified LLM Interface
- **8 LLM Providers**: OpenAI, Anthropic, Google Gemini, Cohere, OpenRouter, DeepSeek, and Ollama
- **Consistent API**: Same interface across all providers
- **100+ Models**: Access to diverse models through OpenRouter integration
- **Async Support**: Full asynchronous capabilities

### 🛡️ Reliability & Failover
- **Reliable LLM**: Automatic failover between primary and secondary providers
- **Retry Logic**: Built-in exponential backoff for failed requests
- **Validation**: Automatic provider validation during initialization

### 🎯 Structured Output
- **Pydantic Integration**: Generate validated JSON responses
- **Type Safety**: Automatic validation and parsing
- **Retry Logic**: Automatic retry on validation failures

### 🔍 Vector Operations
- **Multiple Providers**: OpenAI, Voyage AI, and Cohere embeddings
- **Local & Cloud Storage**: Local vector database and Qdrant integration
- **Semantic Search**: Advanced similarity search capabilities

### 🧠 Intelligent Routing
- **LLM Router**: AI-powered content routing and selection
- **Metadata Filtering**: Route based on content metadata
- **Confidence Scoring**: Intelligent selection with confidence metrics

### 🛠️ Advanced Tools
- **Content Loading**: PDF, DOCX, web pages, and more
- **Text Chunking**: Semantic, sentence, and paragraph-based chunking
- **Search Integration**: Serper and Value Serp APIs
- **Prompt Templates**: Dynamic prompt generation and management

## Quick Start

### Basic LLM Usage

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# Create LLM instance
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Generate response
response = llm.generate_response(prompt="Explain quantum computing in simple terms")
print(response)
```

### All Supported Providers

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# OpenAI
openai_llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Anthropic Claude
anthropic_llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022")

# Google Gemini
gemini_llm = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-1.5-pro")

# Cohere
cohere_llm = LLM.create(provider=LLMProvider.COHERE, model_name="command-r-plus")

# OpenRouter (Access to 100+ models)
openrouter_llm = LLM.create(provider=LLMProvider.OPENROUTER, model_name="openai/gpt-4o")

# DeepSeek
deepseek_llm = LLM.create(provider=LLMProvider.DEEPSEEK, model_name="deepseek-chat")

# Ollama (Local models)
ollama_llm = LLM.create(provider=LLMProvider.OLLAMA, model_name="llama2")
```

### Reliable LLM with Failover

```python
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM

# Create primary and secondary LLMs
primary_llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
secondary_llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022")

# Create reliable LLM with automatic failover
reliable_llm = ReliableLLM(primary_llm, secondary_llm)

# If primary fails, automatically uses secondary
response = reliable_llm.generate_response(prompt="Explain machine learning")
print(response)
```

### Structured JSON Output with Pydantic

```python
from pydantic import BaseModel, Field
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

class MovieRecommendation(BaseModel):
    title: str = Field(description="Movie title")
    genre: str = Field(description="Movie genre")
    year: int = Field(description="Release year")
    rating: float = Field(description="IMDb rating")
    reason: str = Field(description="Why this movie is recommended")

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
prompt = "Recommend a great science fiction movie from the 2020s"

recommendation = generate_pydantic_json_model(
    llm_instance=llm,
    prompt=prompt,
    model_class=MovieRecommendation
)

print(f"Title: {recommendation.title}")
print(f"Genre: {recommendation.genre}")
print(f"Year: {recommendation.year}")
print(f"Rating: {recommendation.rating}")
print(f"Reason: {recommendation.reason}")
```

### Reliable JSON Generation

```python
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_reliable

# Use reliable LLM with JSON generation
reliable_llm = ReliableLLM(primary_llm, secondary_llm)

recommendation, provider, model_name = generate_pydantic_json_model_reliable(
    reliable_llm=reliable_llm,
    prompt=prompt,
    model_class=MovieRecommendation
)

print(f"Generated by: {provider.name} using {model_name}")
print(f"Title: {recommendation.title}")
```

## Embeddings

### All Embedding Providers

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

# OpenAI Embeddings
openai_embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-large"
)

# Voyage AI Embeddings
voyage_embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.VOYAGE,
    model_name="voyage-3-large"
)

# Cohere Embeddings
cohere_embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.COHERE,
    model_name="embed-english-v3.0"
)

# Generate embeddings
text = "SimplerLLM makes AI development easier"
embeddings = openai_embeddings.generate_embeddings(text)
print(f"Embedding dimensions: {len(embeddings)}")
```

### Advanced Embedding Features

```python
# Voyage AI with advanced options
voyage_embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.VOYAGE,
    model_name="voyage-3-large"
)

# Optimize for search queries vs documents
query_embeddings = voyage_embeddings.generate_embeddings(
    user_input="What is machine learning?",
    input_type="query",
    output_dimension=1024
)

document_embeddings = voyage_embeddings.generate_embeddings(
    user_input="Machine learning is a subset of artificial intelligence...",
    input_type="document",
    output_dimension=1024
)

# Cohere with different input types
cohere_embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.COHERE,
    model_name="embed-english-v3.0"
)

classification_embeddings = cohere_embeddings.generate_embeddings(
    user_input="This is a positive review",
    input_type="classification"
)
```

## Vector Databases

### Local Vector Database

```python
from SimplerLLM.vectors.vector_db import VectorDB, VectorProvider
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

# Create embeddings model
embeddings_model = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-small"
)

# Create local vector database
vector_db = VectorDB.create(provider=VectorProvider.LOCAL)

# Add documents
documents = [
    "SimplerLLM is a Python library for LLM interactions",
    "Vector databases store high-dimensional embeddings",
    "Semantic search finds similar content based on meaning"
]

for i, doc in enumerate(documents):
    embedding = embeddings_model.generate_embeddings(doc)
    vector_db.add_vector(id=f"doc_{i}", vector=embedding, metadata={"text": doc})

# Search for similar documents
query = "Python library for AI"
query_embedding = embeddings_model.generate_embeddings(query)
results = vector_db.search(query_embedding, top_k=2)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['metadata']['text']}")
```

### Qdrant Vector Database

```python
from SimplerLLM.vectors.vector_db import VectorDB, VectorProvider

# Create Qdrant vector database
qdrant_db = VectorDB.create(
    provider=VectorProvider.QDRANT,
    url="http://localhost:6333",
    collection_name="my_collection"
)

# Same interface as local vector database
embedding = embeddings_model.generate_embeddings("Sample document")
qdrant_db.add_vector(id="doc_1", vector=embedding, metadata={"text": "Sample document"})
```

## Intelligent Routing

```python
from SimplerLLM.language.llm_router.router import LLMRouter
from SimplerLLM.language.llm import LLM, LLMProvider

# Create router with LLM
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
router = LLMRouter(llm)

# Add choices
choices = [
    ("Machine Learning Tutorial", {"category": "education", "difficulty": "beginner"}),
    ("Advanced Deep Learning", {"category": "education", "difficulty": "advanced"}),
    ("Python Programming Guide", {"category": "programming", "difficulty": "intermediate"})
]

router.add_choices(choices)

# Route based on input
result = router.route("I want to learn the basics of AI")
if result:
    print(f"Selected: {result.selected_index}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Reasoning: {result.reasoning}")

# Get top 3 matches
top_results = router.route_top_k("programming tutorial", k=3)
for result in top_results:
    choice_content, metadata = router.get_choice(result.selected_index)
    print(f"Choice: {choice_content}")
    print(f"Confidence: {result.confidence_score}")
```

## Advanced Tools

### Content Loading

```python
from SimplerLLM.tools.generic_loader import load_content

# Load from various sources
pdf_content = load_content("document.pdf")
web_content = load_content("https://example.com/article")
docx_content = load_content("document.docx")

print(f"PDF content: {pdf_content.content[:100]}...")
print(f"Web content: {web_content.content[:100]}...")
```

### Text Chunking

```python
from SimplerLLM.tools.text_chunker import (
    chunk_by_sentences,
    chunk_by_paragraphs,
    chunk_by_semantics,
    chunk_by_max_chunk_size
)
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

text = "Your long document text here..."

# Sentence-based chunking
sentence_chunks = chunk_by_sentences(text, max_sentences=3)

# Paragraph-based chunking
paragraph_chunks = chunk_by_paragraphs(text, max_paragraphs=2)

# Semantic chunking
embeddings_model = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-small"
)
semantic_chunks = chunk_by_semantics(text, embeddings_model, threshold_percentage=80)

# Size-based chunking
size_chunks = chunk_by_max_chunk_size(text, max_chunk_size=1000)
```

### Search Integration

```python
from SimplerLLM.tools.serp import search_with_serper_api

# Search the web
results = search_with_serper_api("latest AI developments", num_results=5)
for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['link']}")
    print(f"Snippet: {result['snippet']}")
```

### Prompt Templates

```python
from SimplerLLM.prompts.prompt_builder import create_prompt_template, create_multi_value_prompts

# Single prompt template
template = create_prompt_template("Write a {style} article about {topic}")
template.assign_params(style="technical", topic="machine learning")
print(template.content)

# Multi-value prompts
multi_template = create_multi_value_prompts(
    "Hello {name}, your meeting is on {date} about {topic}"
)

params_list = [
    {"name": "Alice", "date": "Monday", "topic": "AI"},
    {"name": "Bob", "date": "Tuesday", "topic": "ML"},
]

prompts = multi_template.generate_prompts(params_list)
for prompt in prompts:
    print(prompt)
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# LLM Providers
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_SITE_URL=your_site_url  # Optional
OPENROUTER_SITE_NAME=your_site_name  # Optional

# Embeddings
VOYAGE_API_KEY=your_voyage_api_key

# Tools
RAPIDAPI_API_KEY=your_rapidapi_key
SERPER_API_KEY=your_serper_api_key
VALUE_SERP_API_KEY=your_value_serp_api_key
STABILITY_API_KEY=your_stability_api_key
```

### Async Support

Most functions support async operations:

```python
import asyncio
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_async

async def main():
    llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    
    # Async response generation
    response = await llm.generate_response_async(prompt="What is async programming?")
    print(response)
    
    # Async JSON generation
    result = await generate_pydantic_json_model_async(
        llm_instance=llm,
        prompt="Generate a product review",
        model_class=MovieRecommendation
    )
    print(result)

asyncio.run(main())
```

## Error Handling and Best Practices

### Robust Error Handling

```python
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

try:
    llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    
    result = generate_pydantic_json_model(
        llm_instance=llm,
        prompt="Generate a summary",
        model_class=MovieRecommendation,
        max_retries=3
    )
    
    if isinstance(result, str):  # Error case
        print(f"Error: {result}")
    else:
        print(f"Success: {result}")
        
except Exception as e:
    print(f"Exception: {e}")
```

### Cost Calculation

```python
from SimplerLLM.language.llm_addons import calculate_text_generation_costs

input_text = "Your input prompt here"
output_text = "Generated response here"

cost_info = calculate_text_generation_costs(
    input=input_text,
    response=output_text,
    cost_per_million_input_tokens=2.50,  # Example: GPT-4 pricing
    cost_per_million_output_tokens=10.00,
    approximate=True  # Use approximate token counting
)

print(f"Input tokens: {cost_info['input_tokens']}")
print(f"Output tokens: {cost_info['output_tokens']}")
print(f"Total cost: ${cost_info['total_cost']:.6f}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- [📖 Documentation](https://docs.simplerllm.com/)
- [💬 Discord Community](https://discord.gg/HUrtZXyp3j)

---

**SimplerLLM** - Making AI development simpler, one line of code at a time.