# SimplerLLM Complete Reference for LLMs

**Version:** Current (January 2025)
**Purpose:** Comprehensive API reference for coding with SimplerLLM - optimized for LLM consumption

---

## Table of Contents

1. [Quick Start & Installation](#1-quick-start--installation)
2. [Core Concepts](#2-core-concepts)
3. [LLM Interface](#3-llm-interface)
4. [Structured JSON Output](#4-structured-json-output)
5. [Embeddings](#5-embeddings)
6. [Vector Databases](#6-vector-databases)
7. [LLM Router](#7-llm-router)
8. [MiniAgents & Flows](#8-miniagents--flows)
9. [Prompt Management](#9-prompt-management)
10. [Tools Module](#10-tools-module)
11. [Advanced Features](#11-advanced-features)
12. [Common Patterns](#12-common-patterns)
13. [API Quick Reference](#13-api-quick-reference)

---

## 1. Quick Start & Installation

### Installation

```bash
pip install simplerllm
```

### Environment Setup

Create a `.env` file in your project root:

```env
# LLM Providers
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
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

### Basic Usage

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# Create an LLM instance
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Generate a response
response = llm.generate_response(prompt="Explain quantum computing")
print(response)
```

---

## 2. Core Concepts

### Architecture Overview

SimplerLLM provides a **unified interface** across multiple LLM providers with consistent API patterns:

- **LLM Module**: Core text generation with 8+ providers
- **Embeddings Module**: Vector generation from text
- **Vector DB Module**: Storage and retrieval of embeddings
- **Tools Module**: Content loading, chunking, search integration
- **Router Module**: Intelligent content routing
- **Flow Module**: Workflow automation (MiniAgents)
- **Prompts Module**: Dynamic template management

### Key Design Patterns

1. **Factory Pattern**: Use `.create()` to instantiate providers
2. **Unified Interface**: Same methods across all providers
3. **Async Support**: `_async` suffix for async methods
4. **Pydantic Models**: Type-safe structured outputs
5. **Failover**: ReliableLLM for automatic provider fallback

---

## 3. LLM Interface

### Supported Providers

SimplerLLM supports **8 LLM providers** through the `LLMProvider` enum:

1. `LLMProvider.OPENAI` - OpenAI (GPT-4, GPT-3.5, etc.)
2. `LLMProvider.ANTHROPIC` - Anthropic Claude
3. `LLMProvider.GEMINI` - Google Gemini
4. `LLMProvider.COHERE` - Cohere
5. `LLMProvider.DEEPSEEK` - DeepSeek
6. `LLMProvider.OPENROUTER` - OpenRouter (100+ models)
7. `LLMProvider.OLLAMA` - Ollama (local models)
8. `LLMProvider.LWH` - Learn With Hasan (custom provider)

### Creating LLM Instances

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# OpenAI
llm = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o",
    temperature=0.7,
    top_p=1.0,
    api_key=None,  # Uses env var if not provided
    verbose=False
)

# Anthropic Claude
llm = LLM.create(
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-3-5-sonnet-20241022"
)

# Google Gemini
llm = LLM.create(
    provider=LLMProvider.GEMINI,
    model_name="gemini-1.5-pro"
)

# Cohere
llm = LLM.create(
    provider=LLMProvider.COHERE,
    model_name="command-r-plus"
)

# OpenRouter (access 100+ models)
llm = LLM.create(
    provider=LLMProvider.OPENROUTER,
    model_name="openai/gpt-4o"
)

# DeepSeek
llm = LLM.create(
    provider=LLMProvider.DEEPSEEK,
    model_name="deepseek-chat"
)

# Ollama (local models)
llm = LLM.create(
    provider=LLMProvider.OLLAMA,
    model_name="llama2"
)
```

### LLM Parameters

**Constructor Parameters:**
- `provider` (LLMProvider): The LLM provider enum
- `model_name` (str): Model identifier (default: "gpt-4o-mini")
- `temperature` (float): Randomness control, 0.0-2.0 (default: 0.7)
- `top_p` (float): Nucleus sampling, 0.0-1.0 (default: 1.0)
- `api_key` (str, optional): API key override (uses env var if None)
- `user_id` (str, optional): User identifier for tracking
- `verbose` (bool): Enable detailed logging (default: False)

### Text Generation Methods

#### Basic Generation

```python
response = llm.generate_response(
    prompt="What is machine learning?",
    system_prompt="You are a helpful AI assistant",  # Optional
    max_tokens=300,  # Default: 300
    temperature=0.7,  # Optional override
    top_p=1.0,  # Optional override
    full_response=False,  # If True, returns LLMFullResponse object
    json_mode=False  # Enable JSON output mode
)
```

**Returns:** `str` (text response) or `LLMFullResponse` object if `full_response=True`

#### Async Generation

```python
import asyncio

async def main():
    response = await llm.generate_response_async(
        prompt="Explain async programming",
        system_prompt="You are a helpful AI assistant",
        max_tokens=500
    )
    print(response)

asyncio.run(main())
```

#### Chat-Based Generation (Messages)

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What are its key features?"}
]

response = llm.generate_response(
    messages=messages,
    max_tokens=500
)
```

### ReliableLLM (Failover System)

Automatically falls back to a secondary provider if the primary fails.

```python
from SimplerLLM.language.llm.reliable import ReliableLLM

# Create primary and secondary LLMs
primary = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
secondary = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022")

# Create ReliableLLM
reliable_llm = ReliableLLM(
    primary_llm=primary,
    secondary_llm=secondary,
    verbose=False
)

# If primary fails, automatically uses secondary
response = reliable_llm.generate_response(
    prompt="Explain machine learning",
    max_tokens=500
)

# Get which provider was used
response, provider, model_name = reliable_llm.generate_response(
    prompt="Explain AI",
    return_provider=True
)
print(f"Used: {provider.name} with model {model_name}")
```

**ReliableLLM Methods:**
- `generate_response()` - Same signature as LLM.generate_response()
- `generate_response_async()` - Async version
- Additional parameter: `return_provider=True` returns tuple `(response, provider, model_name)`

**Validation:**
- Both providers are validated during initialization
- Raises `ValueError` if both providers are invalid
- Sets internal flags: `primary_valid`, `secondary_valid`

---

## 4. Structured JSON Output

Generate validated JSON responses using Pydantic models.

### Basic JSON Generation

```python
from pydantic import BaseModel, Field
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

# Define a Pydantic model
class MovieRecommendation(BaseModel):
    title: str = Field(description="Movie title")
    genre: str = Field(description="Movie genre")
    year: int = Field(description="Release year")
    rating: float = Field(description="IMDb rating out of 10")
    reason: str = Field(description="Why this movie is recommended")

# Generate structured JSON
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

recommendation = generate_pydantic_json_model(
    model_class=MovieRecommendation,
    prompt="Recommend a great science fiction movie from the 2020s",
    llm_instance=llm,
    max_retries=3,  # Retry on validation failure (default: 3)
    max_tokens=4096,
    temperature=0.7,
    top_p=1.0,
    system_prompt="The Output is a VALID Structured JSON",
    full_response=False  # If True, returns LLMFullResponse with model_object attribute
)

# Access validated fields
print(f"Title: {recommendation.title}")
print(f"Genre: {recommendation.genre}")
print(f"Year: {recommendation.year}")
print(f"Rating: {recommendation.rating}")
```

### JSON Generation with ReliableLLM

```python
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_reliable

recommendation, provider, model_name = generate_pydantic_json_model_reliable(
    model_class=MovieRecommendation,
    prompt="Recommend a thriller movie",
    reliable_llm=reliable_llm,
    max_retries=3
)

print(f"Generated by: {provider.name} using {model_name}")
print(f"Title: {recommendation.title}")
```

### Async JSON Generation

```python
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_async

async def get_recommendation():
    result = await generate_pydantic_json_model_async(
        model_class=MovieRecommendation,
        prompt="Recommend a comedy movie",
        llm_instance=llm,
        max_retries=3
    )
    return result

recommendation = asyncio.run(get_recommendation())
```

### JSON Generation Parameters

**Function Signature:**
```python
generate_pydantic_json_model(
    model_class: Type[BaseModel],  # Required
    prompt: str,  # Required
    llm_instance: LLM,  # Required
    max_retries: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,  # Exponential backoff starting delay
    custom_prompt_suffix: str = None,  # Override default JSON instruction
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False
) -> Union[BaseModel, LLMFullResponse, str]
```

**Returns:**
- `BaseModel` instance if successful and `full_response=False`
- `LLMFullResponse` object with `.model_object` attribute if `full_response=True`
- Error message `str` if validation fails after retries

**Error Handling:**
- Automatic retry on validation failures (exponential backoff)
- Returns error string if max retries exceeded
- Check if result `isinstance(result, str)` to detect errors

---

## 5. Embeddings

Generate vector embeddings from text using multiple providers.

### Embedding Providers

SimplerLLM supports **3 embedding providers**:

1. `EmbeddingsProvider.OPENAI` - OpenAI embeddings
2. `EmbeddingsProvider.VOYAGE` - Voyage AI embeddings
3. `EmbeddingsProvider.COHERE` - Cohere embeddings

### Creating Embedding Instances

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

# OpenAI
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-large",  # or "text-embedding-3-small"
    api_key=None  # Uses env var if not provided
)

# Voyage AI
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.VOYAGE,
    model_name="voyage-3-large",  # or "voyage-3", "voyage-3-lite"
    api_key=None
)

# Cohere
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.COHERE,
    model_name="embed-english-v3.0",  # or "embed-multilingual-v3.0"
    api_key=None
)
```

### Generating Embeddings

#### OpenAI Embeddings

```python
# Single text
text = "SimplerLLM makes AI development easier"
embedding = embeddings.generate_embeddings(text)
print(f"Embedding dimensions: {len(embedding)}")

# Multiple texts (batch)
texts = ["Text 1", "Text 2", "Text 3"]
embedding_list = embeddings.generate_embeddings(texts)
```

**Parameters:**
- `user_input` (str or list): Text(s) to embed
- `model_name` (str, optional): Override instance model
- `full_response` (bool): Return full API response (default: False)

#### Voyage AI Embeddings (Advanced)

```python
# Optimized for search queries
query_embeddings = embeddings.generate_embeddings(
    user_input="What is machine learning?",
    input_type="query",  # "query" or "document"
    output_dimension=1024,  # 256, 512, 1024, or 2048
    output_dtype="float"  # "float", "int8", "uint8", "binary", "ubinary"
)

# Optimized for documents
doc_embeddings = embeddings.generate_embeddings(
    user_input="Machine learning is a subset of AI...",
    input_type="document",
    output_dimension=1024
)
```

**Voyage-Specific Parameters:**
- `input_type` (str, optional): "query" or "document" for retrieval optimization
- `output_dimension` (int, optional): Embedding size (256, 512, 1024, 2048)
- `output_dtype` (str, optional): Data type for efficiency

#### Cohere Embeddings

```python
# Classification task
classification_emb = embeddings.generate_embeddings(
    user_input="This is a positive review",
    input_type="classification"  # "search_document", "search_query", "classification", "clustering"
)

# Search documents
doc_emb = embeddings.generate_embeddings(
    user_input="Long document text...",
    input_type="search_document",
    embedding_types=None,  # List of embedding types to return
    truncate="END"  # "START", "END", or "NONE"
)
```

**Cohere-Specific Parameters:**
- `input_type` (str): Task type - "search_document", "search_query", "classification", "clustering"
- `embedding_types` (list, optional): List of embedding types to return
- `truncate` (str): Truncation strategy - "START", "END", or "NONE"

### Async Embeddings

All embedding providers support async generation:

```python
async def get_embeddings():
    embedding = await embeddings.generate_embeddings_async(
        user_input="Async embedding generation"
    )
    return embedding

result = asyncio.run(get_embeddings())
```

---

## 6. Vector Databases

Store and search vector embeddings for semantic search.

### Vector Database Providers

1. `VectorProvider.LOCAL` - In-memory local vector database
2. `VectorProvider.QDRANT` - Qdrant vector database (cloud or self-hosted)

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
    vector_db.add_vector(
        id=f"doc_{i}",
        vector=embedding,
        metadata={"text": doc, "index": i}
    )

# Search for similar documents
query = "Python library for AI"
query_embedding = embeddings_model.generate_embeddings(query)
results = vector_db.search(query_embedding, top_k=2)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['metadata']['text']}")
    print(f"ID: {result['id']}")
```

### Qdrant Vector Database

```python
# Create Qdrant vector database
qdrant_db = VectorDB.create(
    provider=VectorProvider.QDRANT,
    url="http://localhost:6333",  # Qdrant server URL
    collection_name="my_collection",
    vector_size=1536,  # Must match embedding dimensions
    distance="Cosine"  # "Cosine", "Euclid", or "Dot"
)

# Same interface as local DB
embedding = embeddings_model.generate_embeddings("Sample document")
qdrant_db.add_vector(
    id="doc_1",
    vector=embedding,
    metadata={"text": "Sample document", "category": "example"}
)

# Search
results = qdrant_db.search(query_embedding, top_k=5)
```

### Vector DB Methods

**Common Methods Across All Providers:**

```python
# Add a vector
vector_db.add_vector(
    id: str,  # Unique identifier
    vector: List[float],  # Embedding vector
    metadata: Dict = None  # Optional metadata
)

# Add multiple vectors (batch)
vector_db.add_vectors(
    ids: List[str],
    vectors: List[List[float]],
    metadatas: List[Dict] = None
)

# Search for similar vectors
results = vector_db.search(
    query_vector: List[float],
    top_k: int = 5,  # Number of results
    filter: Dict = None  # Metadata filter (Qdrant only)
)

# Delete a vector
vector_db.delete_vector(id: str)

# Get vector by ID
vector = vector_db.get_vector(id: str)

# Clear all vectors
vector_db.clear()
```

**Search Results Format:**
```python
[
    {
        "id": "doc_1",
        "score": 0.95,  # Similarity score
        "metadata": {"text": "...", "custom_field": "..."}
    },
    ...
]
```

---

## 7. LLM Router

AI-powered content routing and selection system.

### Creating a Router

```python
from SimplerLLM.language.llm_router.router import LLMRouter

# Create router with an LLM instance
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
router = LLMRouter(
    llm_instance=llm,
    confidence_threshold=0.5,  # Minimum confidence score (default: 0.5)
    max_choices_per_batch=100  # Max choices per LLM call (default: 100)
)
```

### Adding Choices

```python
# Add choices with metadata
choices = [
    ("Machine Learning Tutorial", {"category": "education", "difficulty": "beginner"}),
    ("Advanced Deep Learning", {"category": "education", "difficulty": "advanced"}),
    ("Python Programming Guide", {"category": "programming", "difficulty": "intermediate"}),
    ("Data Science Basics", {"category": "education", "difficulty": "beginner"})
]

indices = router.add_choices(choices)

# Add single choice
index = router.add_choice(
    content="Neural Networks Explained",
    metadata={"category": "education", "difficulty": "intermediate"}
)
```

### Routing

#### Single Best Match

```python
# Route to best matching choice
result = router.route("I want to learn the basics of AI")

if result:
    print(f"Selected Index: {result.selected_index}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Reasoning: {result.reasoning}")

    # Get the actual choice content
    choice_content, metadata = router.get_choice(result.selected_index)
    print(f"Choice: {choice_content}")
    print(f"Metadata: {metadata}")
```

#### Top K Matches

```python
# Get top 3 matches
top_results = router.route_top_k("programming tutorial", k=3)

for result in top_results:
    choice_content, metadata = router.get_choice(result.selected_index)
    print(f"Choice: {choice_content}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Reasoning: {result.reasoning}")
```

### Metadata Filtering

```python
# Route through choices matching metadata filter
result = router.route_with_metadata(
    input_text="beginner tutorial",
    metadata_filter={"difficulty": "beginner", "category": "education"}
)

# Top K with metadata filter
top_results = router.route_top_k_with_metadata(
    input_text="learn programming",
    metadata_filter={"category": "programming"},
    k=2
)
```

### Router Management

```python
# Get all choices
all_choices = router.get_choices()  # Returns list of (content, metadata) tuples

# Get specific choice
choice = router.get_choice(index=0)

# Update a choice
router.update_choice(
    index=1,
    content="Updated content",
    metadata={"new_key": "value"}
)

# Remove a choice
router.remove_choice(index=2)

# Custom prompt template
router.set_prompt_template(
    "Given the input: {input}\n\nSelect from:\n{choices}\n\nReturn the best match."
)
```

### Async Routing

```python
# Async versions of all routing methods
result = await router.route_async("input text")
top_results = await router.route_top_k_async("input text", k=3)
result = await router.route_with_metadata_async("input", {"key": "value"})
```

### Router Response Model

```python
class RouterResponse:
    selected_index: int  # Index of selected choice
    confidence_score: float  # 0.0 to 1.0
    reasoning: str  # LLM's explanation for selection
```

---

## 8. MiniAgents & Flows

Linear workflow automation system inspired by n8n/Make/Zapier.

### Creating a MiniAgent

```python
from SimplerLLM.language.flow.flow import MiniAgent

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

agent = MiniAgent(
    name="YouTube Summarizer",
    llm_instance=llm,
    system_prompt="You are a helpful AI assistant",
    max_steps=3,  # Maximum number of steps allowed
    verbose=True  # Enable detailed logging
)
```

### Adding Steps

#### LLM Steps

```python
# Basic LLM step
agent.add_step(
    step_type="llm",
    prompt="Summarize the following text: {previous_output}",
    params={"max_tokens": 500, "temperature": 0.7}
)

# LLM step with JSON output
from pydantic import BaseModel, Field

class Summary(BaseModel):
    title: str = Field(description="Summary title")
    key_points: List[str] = Field(description="Main points")
    word_count: int = Field(description="Original word count")

agent.add_step(
    step_type="llm",
    prompt="Extract key information from: {previous_output}",
    output_model=Summary,  # Returns Pydantic model
    max_retries=3,  # Retries for JSON validation
    params={"max_tokens": 1000}
)
```

#### Tool Steps

```python
# Tool step (uses registered tools)
agent.add_step(
    step_type="tool",
    tool_name="youtube_transcript",  # Built-in tool
    params={}  # Optional tool parameters
)

# Tool with custom parameters
agent.add_step(
    step_type="tool",
    tool_name="chunk_by_sentences",
    params={"max_sentences": 3}
)
```

### Running Flows

#### Synchronous Execution

```python
# Run the flow
result = agent.run(user_input="https://youtube.com/watch?v=xyz")

# Check result
if result.success:
    print(f"Final Output: {result.final_output}")
    print(f"Total Duration: {result.total_duration_seconds:.2f}s")

    # Inspect each step
    for step in result.steps:
        print(f"Step {step.step_number}: {step.step_type}")
        print(f"Duration: {step.duration_seconds:.2f}s")
        print(f"Output: {step.output_data}")
else:
    print(f"Flow failed: {result.error}")
```

#### Async Execution

```python
async def run_flow():
    result = await agent.run_async(user_input="input data")
    return result

result = asyncio.run(run_flow())
```

#### Concurrent Flows

```python
import asyncio

async def run_multiple_agents():
    # Run multiple agents concurrently
    results = await asyncio.gather(
        agent1.run_async("input 1"),
        agent2.run_async("input 2"),
        agent3.run_async("input 3")
    )
    return results

all_results = asyncio.run(run_multiple_agents())
```

### Variable Interpolation

Flows support placeholder replacement in prompts:

```python
# {previous_output} - replaced with previous step's output
agent.add_step(
    step_type="llm",
    prompt="Summarize: {previous_output}"
)

# {input} - replaced with the flow's initial input
agent.add_step(
    step_type="llm",
    prompt="Process this URL: {input}"
)

# If no placeholder, input is appended
agent.add_step(
    step_type="llm",
    prompt="Create a summary"  # Input appended as "\n\nInput: <data>"
)
```

### Built-in Tools for Flows

The Flow system has access to 15+ registered tools:

- `youtube_transcript` - Extract YouTube video transcript
- `load_content` - Load from URL, PDF, DOCX, TXT
- `chunk_by_sentences` - Split text into sentences
- `chunk_by_paragraphs` - Split by paragraphs
- `chunk_by_semantics` - Semantic chunking
- `chunk_by_size` - Size-based chunking
- `web_search_serper` - Search with Serper API
- And more...

### Custom Tool Registration

```python
from SimplerLLM.language.flow.tool_registry import ToolRegistry

# Register a custom function as a tool
def my_custom_tool(input_text: str) -> str:
    """Process text in a custom way"""
    return input_text.upper()

ToolRegistry.register_tool("my_tool", my_custom_tool)

# Use in flow
agent.add_step(step_type="tool", tool_name="my_tool")
```

### Flow Management

```python
# Get number of steps
count = agent.get_step_count()

# Clear all steps
agent.clear_steps()
```

### Flow Result Model

```python
class FlowResult:
    agent_name: str
    total_steps: int
    steps: List[StepResult]  # Individual step results
    total_duration_seconds: float
    final_output: Any  # Final step output (None if failed)
    success: bool
    error: Optional[str]  # Error message if failed

class StepResult:
    step_number: int
    step_type: str  # "llm" or "tool"
    input_data: Any
    output_data: Any
    duration_seconds: float
    tool_used: Optional[str]  # Tool name if tool step
    prompt_used: Optional[str]  # Prompt if LLM step
    output_model_class: Optional[str]  # Pydantic model class name
    error: Optional[str]  # Error if step failed
```

### Complete Flow Example

```python
from pydantic import BaseModel, Field
from SimplerLLM.language.flow.flow import MiniAgent

# Define output model
class VideoSummary(BaseModel):
    title: str
    key_points: List[str]
    duration_estimate: str

# Create agent
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
agent = MiniAgent(
    name="Video Summarizer",
    llm_instance=llm,
    max_steps=3,
    verbose=True
)

# Step 1: Get transcript
agent.add_step(
    step_type="tool",
    tool_name="youtube_transcript"
)

# Step 2: Generate structured summary
agent.add_step(
    step_type="llm",
    prompt="Analyze this video transcript: {previous_output}",
    output_model=VideoSummary,
    max_retries=3
)

# Step 3: Create final report
agent.add_step(
    step_type="llm",
    prompt="Write a blog post based on: {previous_output}"
)

# Run
result = agent.run("https://youtube.com/watch?v=xyz")
if result.success:
    print(result.final_output)
```

---

## 9. Prompt Management

Dynamic prompt template system for reusable prompts.

### Simple Prompt Templates

```python
from SimplerLLM.prompts.prompt_builder import create_prompt_template

# Create template with placeholders
template = create_prompt_template(
    "Write a {style} article about {topic} for {audience}"
)

# Assign parameters
filled_prompt = template.assign_parms(
    style="technical",
    topic="machine learning",
    audience="developers"
)

print(filled_prompt)
# Output: "Write a technical article about machine learning for developers"

# Access the content
print(template.content)  # Same as filled_prompt

# Update template
template.update_template("New template: {param1} and {param2}")
filled = template.assign_parms(param1="value1", param2="value2")
```

### Multi-Value Prompts (Batch Generation)

```python
from SimplerLLM.prompts.prompt_builder import create_multi_value_prompts

# Create template for multiple parameter sets
multi_template = create_multi_value_prompts(
    "Hello {name}, your meeting is on {date} about {topic}"
)

# Generate multiple prompts
params_list = [
    {"name": "Alice", "date": "Monday", "topic": "AI"},
    {"name": "Bob", "date": "Tuesday", "topic": "ML"},
    {"name": "Carol", "date": "Wednesday", "topic": "Data Science"}
]

prompts = multi_template.generate_prompts(params_list)

for prompt in prompts:
    print(prompt)

# Output:
# Hello Alice, your meeting is on Monday about AI
# Hello Bob, your meeting is on Tuesday about ML
# Hello Carol, your meeting is on Wednesday about Data Science

# Access all generated prompts
print(multi_template.generated_prompts)
```

### Prompt Template Classes

**SimplePrompt:**
```python
class SimplePrompt:
    template: str  # Template with {placeholders}
    content: str  # Latest filled template

    def assign_parms(**kwargs) -> str  # Fill template
    def update_template(new_template: str)  # Change template
```

**MultiValuePrompt:**
```python
class MultiValuePrompt:
    template: str
    generated_prompts: List[str]  # All generated prompts

    def generate_prompts(params_list: List[Dict]) -> List[str]
```

### Error Handling

```python
try:
    template = create_prompt_template("Hello {name}")
    filled = template.assign_parms(name="Alice")  # OK
    filled = template.assign_parms(wrong_key="value")  # Raises KeyError
except KeyError as e:
    print(f"Missing parameter: {e}")
```

---

## 10. Tools Module

Content loading, processing, and integration utilities.

### Content Loading

Load content from various sources with a unified interface.

```python
from SimplerLLM.tools.generic_loader import load_content

# Load from URL (blog article, webpage)
doc = load_content("https://example.com/article")

# Load from PDF file
doc = load_content("document.pdf")

# Load from DOCX file
doc = load_content("document.docx")

# Load from text file (TXT, CSV)
doc = load_content("file.txt")

# Access document properties
print(f"Title: {doc.title}")
print(f"Content: {doc.content[:500]}...")
print(f"Word count: {doc.word_count}")
print(f"Character count: {doc.character_count}")
print(f"File size: {doc.file_size} bytes")
print(f"Source: {doc.url_or_path}")
```

**TextDocument Model:**
```python
class TextDocument:
    content: str  # Full text content
    word_count: int
    character_count: int
    file_size: Optional[int]  # In bytes (None for URLs)
    title: Optional[str]  # Extracted title (URLs only)
    url_or_path: Optional[str]  # Source location
```

### Text Chunking

Split text into manageable chunks using different strategies.

#### Chunk by Max Size

```python
from SimplerLLM.tools.text_chunker import chunk_by_max_chunk_size

text = "Your long document text here..."

# Basic size-based chunking
chunks = chunk_by_max_chunk_size(
    text=text,
    max_chunk_size=1000,  # Characters per chunk
    preserve_sentence_structure=False
)

# Size-based with sentence preservation
chunks = chunk_by_max_chunk_size(
    text=text,
    max_chunk_size=1000,
    preserve_sentence_structure=True  # Don't split mid-sentence
)

# Access chunk information
print(f"Number of chunks: {chunks.num_chunks}")
for chunk_info in chunks.chunk_list:
    print(f"Text: {chunk_info.text[:100]}...")
    print(f"Characters: {chunk_info.num_characters}")
    print(f"Words: {chunk_info.num_words}")
```

#### Chunk by Sentences

```python
from SimplerLLM.tools.text_chunker import chunk_by_sentences

chunks = chunk_by_sentences(text)

for chunk_info in chunks.chunk_list:
    print(chunk_info.text)
```

#### Chunk by Paragraphs

```python
from SimplerLLM.tools.text_chunker import chunk_by_paragraphs

chunks = chunk_by_paragraphs(text)
```

#### Semantic Chunking (Advanced)

Groups related sentences based on meaning similarity.

```python
from SimplerLLM.tools.text_chunker import chunk_by_semantics
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

# Create embeddings model
embeddings_model = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-small"
)

# Semantic chunking
chunks = chunk_by_semantics(
    text=text,
    llm_embeddings_instance=embeddings_model,
    threshold_percentage=90  # Higher = more chunks (default: 90)
)

# threshold_percentage:
# - 90 (default): More, smaller chunks (strict similarity)
# - 80: Moderate chunking
# - 70: Fewer, larger chunks (looser similarity)
```

**TextChunks Model:**
```python
class TextChunks:
    num_chunks: int
    chunk_list: List[ChunkInfo]

class ChunkInfo:
    text: str
    num_characters: int
    num_words: int
```

### Search Engine Integration

#### Serper API (Google Search)

```python
from SimplerLLM.tools.serp import search_with_serper_api

# Search the web
results = search_with_serper_api(
    query="latest AI developments",
    num_results=5,  # Number of results to return
    location="United States",  # Optional location
    language="en"  # Optional language
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['link']}")
    print(f"Snippet: {result['snippet']}")
    print(f"Position: {result.get('position', 'N/A')}")
```

### YouTube Data Extraction

```python
from SimplerLLM.tools.youtube import get_youtube_transcript, get_youtube_metadata

# Get video transcript
video_url = "https://youtube.com/watch?v=VIDEO_ID"
transcript = get_youtube_transcript(video_url)
print(transcript)  # Full transcript text

# Get video metadata
metadata = get_youtube_metadata(video_url)
print(f"Title: {metadata['title']}")
print(f"Duration: {metadata['duration']}")
print(f"Views: {metadata['views']}")
```

### File Operations

```python
from SimplerLLM.tools.file_functions import write_to_file, read_from_file

# Write to file
write_to_file(
    file_path="output.txt",
    content="Text to write",
    mode="w"  # "w" (overwrite) or "a" (append)
)

# Read from file
content = read_from_file("output.txt")
```

### RapidAPI Integration

```python
from SimplerLLM.tools.rapid_api import call_rapid_api

# Generic RapidAPI caller
response = call_rapid_api(
    api_url="https://api-endpoint.rapidapi.com/...",
    method="GET",  # or "POST"
    headers={"custom-header": "value"},
    params={"param1": "value1"}
)
```

---

## 11. Advanced Features

### Cost Calculation

Calculate token usage and costs for API calls.

```python
from SimplerLLM.language.llm_addons import calculate_text_generation_costs

input_text = "Your input prompt here"
output_text = "Generated response here"

cost_info = calculate_text_generation_costs(
    input=input_text,
    response=output_text,
    cost_per_million_input_tokens=2.50,  # Example: GPT-4 pricing
    cost_per_million_output_tokens=10.00,
    approximate=True  # Fast approximation (chars/4) vs tiktoken
)

print(f"Input tokens: {cost_info['input_tokens']}")
print(f"Output tokens: {cost_info['output_tokens']}")
print(f"Input cost: ${cost_info['input_cost']:.6f}")
print(f"Output cost: ${cost_info['output_cost']:.6f}")
print(f"Total cost: ${cost_info['total_cost']:.6f}")
```

**approximate=True:** Uses `len(text) // 4` (fast)
**approximate=False:** Uses `tiktoken` library (accurate but slower)

### Verbose Logging

Enable detailed logging for debugging.

```python
from SimplerLLM.utils.custom_verbose import verbose_print

# Use in any component
llm = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o",
    verbose=True  # Enable verbose logging
)

agent = MiniAgent(
    name="Agent",
    llm_instance=llm,
    verbose=True
)

# Custom verbose prints
verbose_print("This is an info message", "info")
verbose_print("This is a debug message", "debug")
verbose_print("This is a warning", "warning")
verbose_print("This is an error", "error")
```

### Full Response Objects

Get complete API response including token counts.

```python
# LLM with full_response
response_obj = llm.generate_response(
    prompt="Explain AI",
    full_response=True
)

print(f"Text: {response_obj.generated_text}")
print(f"Input tokens: {response_obj.input_token_count}")
print(f"Output tokens: {response_obj.output_token_count}")
print(f"Model: {response_obj.model_name}")

# JSON generation with full_response
result = generate_pydantic_json_model(
    model_class=MyModel,
    prompt="Generate data",
    llm_instance=llm,
    full_response=True
)

print(f"Model object: {result.model_object}")
print(f"Tokens used: {result.input_token_count + result.output_token_count}")
```

**LLMFullResponse Model:**
```python
class LLMFullResponse:
    generated_text: str
    input_token_count: int
    output_token_count: int
    model_name: str
    model_object: Optional[BaseModel]  # Only for JSON generation
    provider: Optional[LLMProvider]  # Only for ReliableLLM
```

---

## 12. Common Patterns

### Pattern 1: RAG (Retrieval-Augmented Generation)

```python
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider
from SimplerLLM.vectors.vector_db import VectorDB, VectorProvider
from SimplerLLM.tools.generic_loader import load_content
from SimplerLLM.tools.text_chunker import chunk_by_max_chunk_size

# 1. Load documents
doc1 = load_content("document1.pdf")
doc2 = load_content("https://example.com/article")

# 2. Chunk documents
chunks1 = chunk_by_max_chunk_size(doc1.content, max_chunk_size=500, preserve_sentence_structure=True)
chunks2 = chunk_by_max_chunk_size(doc2.content, max_chunk_size=500, preserve_sentence_structure=True)

all_chunks = chunks1.chunk_list + chunks2.chunk_list

# 3. Create embeddings and vector DB
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small")
vector_db = VectorDB.create(provider=VectorProvider.LOCAL)

# 4. Store chunks
for i, chunk_info in enumerate(all_chunks):
    embedding = embeddings.generate_embeddings(chunk_info.text)
    vector_db.add_vector(
        id=f"chunk_{i}",
        vector=embedding,
        metadata={"text": chunk_info.text, "index": i}
    )

# 5. Query
query = "What is machine learning?"
query_embedding = embeddings.generate_embeddings(query)
results = vector_db.search(query_embedding, top_k=3)

# 6. Generate answer with context
context = "\n\n".join([r['metadata']['text'] for r in results])
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

answer = llm.generate_response(
    prompt=f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}",
    max_tokens=500
)

print(answer)
```

### Pattern 2: Async Batch Processing

```python
import asyncio
from SimplerLLM.language.llm import LLM, LLMProvider

async def process_item(llm, item):
    return await llm.generate_response_async(
        prompt=f"Summarize: {item}",
        max_tokens=100
    )

async def batch_process(items):
    llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    tasks = [process_item(llm, item) for item in items]
    results = await asyncio.gather(*tasks)

    return results

items = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
results = asyncio.run(batch_process(items))

for i, result in enumerate(results):
    print(f"Item {i+1}: {result}")
```

### Pattern 3: Multi-Provider Fallback Chain

```python
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM

# Create multiple provider instances
openai_llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
anthropic_llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022")
gemini_llm = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-1.5-pro")

# Two-tier fallback
reliable_llm_1 = ReliableLLM(primary_llm=openai_llm, secondary_llm=anthropic_llm)
reliable_llm_2 = ReliableLLM(primary_llm=reliable_llm_1, secondary_llm=gemini_llm)

# If OpenAI fails → Anthropic, if both fail → Gemini
response = reliable_llm_2.generate_response(prompt="Explain quantum computing")
```

### Pattern 4: Structured Data Extraction Pipeline

```python
from pydantic import BaseModel, Field
from SimplerLLM.language.flow.flow import MiniAgent
from SimplerLLM.language.llm import LLM, LLMProvider

class Article(BaseModel):
    title: str
    author: str
    publish_date: str
    summary: str
    key_topics: List[str]

class SEOMetadata(BaseModel):
    meta_title: str = Field(max_length=60)
    meta_description: str = Field(max_length=160)
    keywords: List[str]

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
agent = MiniAgent("Data Extractor", llm, max_steps=3)

# Step 1: Load content
agent.add_step(step_type="tool", tool_name="load_content")

# Step 2: Extract article data
agent.add_step(
    step_type="llm",
    prompt="Extract article metadata from: {previous_output}",
    output_model=Article
)

# Step 3: Generate SEO metadata
agent.add_step(
    step_type="llm",
    prompt="Create SEO metadata for: {previous_output}",
    output_model=SEOMetadata
)

result = agent.run("https://example.com/article")
seo_data = result.final_output
```

### Pattern 5: Intelligent Content Router

```python
from SimplerLLM.language.llm_router.router import LLMRouter
from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
router = LLMRouter(llm, confidence_threshold=0.7)

# Add content categories
categories = [
    ("Python Programming", {"type": "code", "language": "python"}),
    ("Machine Learning Basics", {"type": "education", "domain": "AI"}),
    ("Web Development", {"type": "code", "domain": "web"}),
    ("Data Science", {"type": "education", "domain": "data"}),
    ("DevOps Practices", {"type": "ops", "domain": "infrastructure"})
]

router.add_choices(categories)

# Route user queries to appropriate content
user_queries = [
    "How do I use pandas for data analysis?",
    "What is a neural network?",
    "How to deploy a React app?"
]

for query in user_queries:
    result = router.route(query)
    if result:
        content, metadata = router.get_choice(result.selected_index)
        print(f"Query: {query}")
        print(f"Matched: {content} (confidence: {result.confidence_score:.2f})")
        print(f"Reasoning: {result.reasoning}\n")
```

---

## 13. API Quick Reference

### LLM Interface

```python
# Create LLM
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o", temperature=0.7, verbose=False)

# Generate text
response = llm.generate_response(prompt="...", system_prompt="...", max_tokens=300)
response = await llm.generate_response_async(prompt="...")

# ReliableLLM
reliable = ReliableLLM(primary_llm, secondary_llm, verbose=False)
response = reliable.generate_response(prompt="...", return_provider=False)
```

### Structured JSON

```python
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

result = generate_pydantic_json_model(
    model_class=MyModel,
    prompt="...",
    llm_instance=llm,
    max_retries=3,
    full_response=False
)
```

### Embeddings

```python
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small")
embedding = embeddings.generate_embeddings("text")
embedding = await embeddings.generate_embeddings_async("text")
```

### Vector Databases

```python
vector_db = VectorDB.create(provider=VectorProvider.LOCAL)
vector_db.add_vector(id="1", vector=[...], metadata={})
results = vector_db.search(query_vector=[...], top_k=5)
```

### Router

```python
router = LLMRouter(llm, confidence_threshold=0.5)
router.add_choice(content="...", metadata={})
result = router.route("input text")
results = router.route_top_k("input text", k=3)
```

### MiniAgents

```python
agent = MiniAgent(name="Agent", llm_instance=llm, max_steps=3, verbose=False)
agent.add_step(step_type="llm", prompt="...", output_model=None)
agent.add_step(step_type="tool", tool_name="...")
result = agent.run(user_input)
result = await agent.run_async(user_input)
```

### Prompts

```python
from SimplerLLM.prompts.prompt_builder import create_prompt_template, create_multi_value_prompts

template = create_prompt_template("Text {placeholder}")
filled = template.assign_parms(placeholder="value")

multi = create_multi_value_prompts("Template {key}")
prompts = multi.generate_prompts([{"key": "val1"}, {"key": "val2"}])
```

### Tools

```python
from SimplerLLM.tools.generic_loader import load_content
from SimplerLLM.tools.text_chunker import chunk_by_sentences, chunk_by_max_chunk_size, chunk_by_semantics

doc = load_content("file.pdf")  # or URL
chunks = chunk_by_max_chunk_size(text, max_chunk_size=1000, preserve_sentence_structure=True)
chunks = chunk_by_semantics(text, embeddings_model, threshold_percentage=90)
```

### Cost Calculation

```python
from SimplerLLM.language.llm_addons import calculate_text_generation_costs

cost = calculate_text_generation_costs(
    input="...",
    response="...",
    cost_per_million_input_tokens=2.5,
    cost_per_million_output_tokens=10.0,
    approximate=True
)
```

---

## Appendix: All Provider Enums

### LLMProvider

```python
from SimplerLLM.language.llm import LLMProvider

LLMProvider.OPENAI
LLMProvider.ANTHROPIC
LLMProvider.GEMINI
LLMProvider.COHERE
LLMProvider.DEEPSEEK
LLMProvider.OPENROUTER
LLMProvider.OLLAMA
LLMProvider.LWH
```

### EmbeddingsProvider

```python
from SimplerLLM.language.embeddings import EmbeddingsProvider

EmbeddingsProvider.OPENAI
EmbeddingsProvider.VOYAGE
EmbeddingsProvider.COHERE
```

### VectorProvider

```python
from SimplerLLM.vectors.vector_db import VectorProvider

VectorProvider.LOCAL
VectorProvider.QDRANT
```

---

## Appendix: Common Model Names

### OpenAI Models

- `gpt-4o` - GPT-4 Omni (latest)
- `gpt-4o-mini` - GPT-4 Omni Mini
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-3.5-turbo` - GPT-3.5 Turbo

### Anthropic Models

- `claude-3-5-sonnet-20241022` - Claude 3.5 Sonnet (latest)
- `claude-3-opus-20240229` - Claude 3 Opus
- `claude-3-sonnet-20240229` - Claude 3 Sonnet
- `claude-3-haiku-20240307` - Claude 3 Haiku

### Google Gemini Models

- `gemini-1.5-pro` - Gemini 1.5 Pro
- `gemini-1.5-flash` - Gemini 1.5 Flash
- `gemini-pro` - Gemini Pro

### Cohere Models

- `command-r-plus` - Command R+
- `command-r` - Command R
- `command` - Command

### DeepSeek Models

- `deepseek-chat` - DeepSeek Chat
- `deepseek-coder` - DeepSeek Coder

### Embedding Models

**OpenAI:**
- `text-embedding-3-large` (3072 dimensions)
- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-ada-002` (1536 dimensions)

**Voyage AI:**
- `voyage-3-large` (2048 dimensions)
- `voyage-3` (1024 dimensions)
- `voyage-3-lite` (512 dimensions)

**Cohere:**
- `embed-english-v3.0` (1024 dimensions)
- `embed-multilingual-v3.0` (1024 dimensions)

---

**End of SimplerLLM Reference**

For the latest documentation and updates, visit: https://docs.simplerllm.com/
