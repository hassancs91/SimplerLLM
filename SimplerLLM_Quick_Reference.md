# SimplerLLM Quick Reference

**Version:** January 2025
**Purpose:** Condensed API reference for fast lookup while coding
**For comprehensive details, see:** [SimplerLLM_Complete_Reference.md](SimplerLLM_Complete_Reference.md)

---

## Installation & Setup

```bash
pip install simplerllm
```

Create `.env` file:
```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key
COHERE_API_KEY=your_key
VOYAGE_API_KEY=your_key
SERPER_API_KEY=your_key
```

---

## LLM Interface

### Supported Providers
8 providers via `LLMProvider` enum: OPENAI, ANTHROPIC, GEMINI, COHERE, DEEPSEEK, OPENROUTER, OLLAMA, LWH

### Basic Usage

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# Create LLM
llm = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o",  # default: "gpt-4o-mini"
    temperature=0.7,  # default: 0.7
    top_p=1.0,  # default: 1.0
    api_key=None,  # uses env var
    verbose=False
)

# Generate response
response = llm.generate_response(
    prompt="What is machine learning?",
    system_prompt="You are a helpful AI assistant",
    max_tokens=300,
    temperature=0.7,
    top_p=1.0
)

# Async
response = await llm.generate_response_async(prompt="...")
```

### Other Providers

```python
LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022")
LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-1.5-pro")
LLM.create(provider=LLMProvider.COHERE, model_name="command-r-plus")
LLM.create(provider=LLMProvider.OPENROUTER, model_name="openai/gpt-4o")
LLM.create(provider=LLMProvider.DEEPSEEK, model_name="deepseek-chat")
LLM.create(provider=LLMProvider.OLLAMA, model_name="llama2")
```

### ReliableLLM (Failover)

```python
from SimplerLLM.language.llm.reliable import ReliableLLM

primary = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
secondary = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022")

reliable_llm = ReliableLLM(primary_llm=primary, secondary_llm=secondary)
response = reliable_llm.generate_response(prompt="...")

# Get which provider was used
response, provider, model_name = reliable_llm.generate_response(prompt="...", return_provider=True)
```

---

## Structured JSON Output

```python
from pydantic import BaseModel, Field
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

class MovieRecommendation(BaseModel):
    title: str = Field(description="Movie title")
    genre: str
    year: int
    rating: float

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

result = generate_pydantic_json_model(
    model_class=MovieRecommendation,
    prompt="Recommend a sci-fi movie from 2020s",
    llm_instance=llm,
    max_retries=3,  # default: 3
    max_tokens=4096,
    full_response=False
)

print(result.title, result.year)  # Access validated fields

# Check for errors
if isinstance(result, str):
    print(f"Error: {result}")
```

**Async:** `generate_pydantic_json_model_async()`
**With ReliableLLM:** `generate_pydantic_json_model_reliable()`

---

## Embeddings

3 providers: OPENAI, VOYAGE, COHERE

```python
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

# OpenAI
embeddings = EmbeddingsLLM.create(
    provider=EmbeddingsProvider.OPENAI,
    model_name="text-embedding-3-small"  # or "text-embedding-3-large"
)
embedding = embeddings.generate_embeddings("text to embed")
embedding_list = embeddings.generate_embeddings(["text1", "text2"])

# Voyage AI (with advanced options)
voyage = EmbeddingsLLM.create(provider=EmbeddingsProvider.VOYAGE, model_name="voyage-3")
query_emb = voyage.generate_embeddings("search query", input_type="query", output_dimension=1024)
doc_emb = voyage.generate_embeddings("document text", input_type="document")

# Cohere
cohere = EmbeddingsLLM.create(provider=EmbeddingsProvider.COHERE, model_name="embed-english-v3.0")
emb = cohere.generate_embeddings("text", input_type="search_document")

# Async
embedding = await embeddings.generate_embeddings_async("text")
```

---

## Vector Databases

2 providers: LOCAL (in-memory), QDRANT

```python
from SimplerLLM.vectors.vector_db import VectorDB, VectorProvider

# Local vector DB
vector_db = VectorDB.create(provider=VectorProvider.LOCAL)

# Qdrant
vector_db = VectorDB.create(
    provider=VectorProvider.QDRANT,
    url="http://localhost:6333",
    collection_name="my_collection",
    vector_size=1536,
    distance="Cosine"
)

# Add vectors
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small")
for i, doc in enumerate(["doc1", "doc2", "doc3"]):
    emb = embeddings.generate_embeddings(doc)
    vector_db.add_vector(id=f"doc_{i}", vector=emb, metadata={"text": doc})

# Search
query_emb = embeddings.generate_embeddings("search query")
results = vector_db.search(query_emb, top_k=3)

for r in results:
    print(f"Score: {r['score']}, Text: {r['metadata']['text']}")
```

**Methods:** `add_vector()`, `add_vectors()`, `search()`, `delete_vector()`, `get_vector()`, `clear()`

---

## LLM Router

```python
from SimplerLLM.language.llm_router.router import LLMRouter

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
router = LLMRouter(llm, confidence_threshold=0.5)

# Add choices
router.add_choices([
    ("Machine Learning Tutorial", {"difficulty": "beginner"}),
    ("Advanced Deep Learning", {"difficulty": "advanced"}),
    ("Python Guide", {"difficulty": "intermediate"})
])

# Route to best match
result = router.route("I want to learn ML basics")
if result:
    choice, metadata = router.get_choice(result.selected_index)
    print(f"Selected: {choice}, Confidence: {result.confidence_score}")

# Top K matches
top_3 = router.route_top_k("programming tutorial", k=3)
for r in top_3:
    choice, _ = router.get_choice(r.selected_index)
    print(f"{choice}: {r.confidence_score}")

# Async
result = await router.route_async("input")
```

---

## MiniAgents & Flows

Linear workflow automation system.

```python
from SimplerLLM.language.flow.flow import MiniAgent
from pydantic import BaseModel, Field

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

agent = MiniAgent(
    name="YouTube Summarizer",
    llm_instance=llm,
    system_prompt="You are a helpful AI assistant",
    max_steps=3,
    verbose=True
)

# Add LLM step
agent.add_step(
    step_type="llm",
    prompt="Summarize: {previous_output}",
    params={"max_tokens": 500}
)

# Add LLM step with structured JSON output
class Summary(BaseModel):
    title: str
    key_points: list[str]

agent.add_step(
    step_type="llm",
    prompt="Extract key info from: {previous_output}",
    output_model=Summary,
    max_retries=3
)

# Add tool step
agent.add_step(
    step_type="tool",
    tool_name="youtube_transcript"
)

# Run flow
result = agent.run(user_input="https://youtube.com/watch?v=xyz")

if result.success:
    print(result.final_output)
    print(f"Duration: {result.total_duration_seconds:.2f}s")
else:
    print(f"Failed: {result.error}")

# Async
result = await agent.run_async(user_input)

# Concurrent flows
results = await asyncio.gather(
    agent1.run_async("input1"),
    agent2.run_async("input2")
)
```

### Variable Interpolation
- `{previous_output}` - Previous step's output
- `{input}` - Initial flow input
- If no placeholder, input appended automatically

### Built-in Tools
`youtube_transcript`, `load_content`, `chunk_by_sentences`, `chunk_by_paragraphs`, `chunk_by_semantics`, `chunk_by_size`, `web_search_serper`

### Custom Tools
```python
from SimplerLLM.language.flow.tool_registry import ToolRegistry

def my_tool(input_text: str) -> str:
    return input_text.upper()

ToolRegistry.register_tool("my_tool", my_tool)
agent.add_step(step_type="tool", tool_name="my_tool")
```

### Complete Flow Example

```python
class VideoSummary(BaseModel):
    title: str
    key_points: list[str]

agent = MiniAgent("Video Processor", llm, max_steps=3)

agent.add_step(step_type="tool", tool_name="youtube_transcript")
agent.add_step(
    step_type="llm",
    prompt="Analyze: {previous_output}",
    output_model=VideoSummary
)
agent.add_step(
    step_type="llm",
    prompt="Write blog post from: {previous_output}"
)

result = agent.run("https://youtube.com/watch?v=xyz")
```

---

## Prompt Management

```python
from SimplerLLM.prompts.prompt_builder import create_prompt_template, create_multi_value_prompts

# Simple template
template = create_prompt_template("Write a {style} article about {topic}")
filled = template.assign_parms(style="technical", topic="AI")
print(filled)  # "Write a technical article about AI"

# Multi-value prompts
multi = create_multi_value_prompts("Hello {name}, meeting on {date}")
prompts = multi.generate_prompts([
    {"name": "Alice", "date": "Monday"},
    {"name": "Bob", "date": "Tuesday"}
])
# ["Hello Alice, meeting on Monday", "Hello Bob, meeting on Tuesday"]
```

---

## Tools Module

### Content Loading

```python
from SimplerLLM.tools.generic_loader import load_content

doc = load_content("document.pdf")  # or URL, DOCX, TXT
print(doc.content, doc.word_count, doc.title)
```

### Text Chunking

```python
from SimplerLLM.tools.text_chunker import (
    chunk_by_max_chunk_size, chunk_by_sentences,
    chunk_by_paragraphs, chunk_by_semantics
)

text = "Long document..."

# By size
chunks = chunk_by_max_chunk_size(text, max_chunk_size=1000, preserve_sentence_structure=True)

# By sentences
chunks = chunk_by_sentences(text)

# By paragraphs
chunks = chunk_by_paragraphs(text)

# Semantic (groups related sentences)
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small")
chunks = chunk_by_semantics(text, embeddings, threshold_percentage=90)

# Access chunks
for chunk_info in chunks.chunk_list:
    print(chunk_info.text, chunk_info.num_words)
```

### Search Integration

```python
from SimplerLLM.tools.serp import search_with_serper_api

results = search_with_serper_api("latest AI developments", num_results=5)
for r in results:
    print(r['title'], r['link'], r['snippet'])
```

### YouTube

```python
from SimplerLLM.tools.youtube import get_youtube_transcript

transcript = get_youtube_transcript("https://youtube.com/watch?v=VIDEO_ID")
```

---

## Advanced Features

### Cost Calculation

```python
from SimplerLLM.language.llm_addons import calculate_text_generation_costs

cost = calculate_text_generation_costs(
    input="input text",
    response="output text",
    cost_per_million_input_tokens=2.50,
    cost_per_million_output_tokens=10.00,
    approximate=True  # Fast (chars/4) vs tiktoken
)

print(f"Total: ${cost['total_cost']:.6f}")
print(f"Tokens: {cost['input_tokens']} in, {cost['output_tokens']} out")
```

### Verbose Logging

```python
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o", verbose=True)
agent = MiniAgent("Agent", llm, verbose=True)
```

### Full Response Objects

```python
response_obj = llm.generate_response(prompt="...", full_response=True)
print(response_obj.generated_text)
print(response_obj.input_token_count, response_obj.output_token_count)
```

---

## API Quick Reference

### LLM
```python
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o", temperature=0.7)
response = llm.generate_response(prompt="...", max_tokens=300)
response = await llm.generate_response_async(prompt="...")

reliable = ReliableLLM(primary_llm, secondary_llm)
response = reliable.generate_response(prompt="...", return_provider=False)
```

### JSON
```python
result = generate_pydantic_json_model(model_class=MyModel, prompt="...", llm_instance=llm, max_retries=3)
result = await generate_pydantic_json_model_async(...)
result, provider, model = generate_pydantic_json_model_reliable(model_class=MyModel, prompt="...", reliable_llm=reliable)
```

### Embeddings
```python
embeddings = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small")
emb = embeddings.generate_embeddings("text")
emb = await embeddings.generate_embeddings_async("text")
```

### Vector DB
```python
vector_db = VectorDB.create(provider=VectorProvider.LOCAL)
vector_db.add_vector(id="1", vector=[...], metadata={})
results = vector_db.search(query_vector=[...], top_k=5)
```

### Router
```python
router = LLMRouter(llm, confidence_threshold=0.5)
router.add_choice(content="...", metadata={})
result = router.route("input")
results = router.route_top_k("input", k=3)
```

### MiniAgents
```python
agent = MiniAgent(name="Agent", llm_instance=llm, max_steps=3)
agent.add_step(step_type="llm", prompt="...", output_model=None, params={})
agent.add_step(step_type="tool", tool_name="...", params={})
result = agent.run(user_input)
result = await agent.run_async(user_input)
```

### Prompts
```python
template = create_prompt_template("Text {placeholder}")
filled = template.assign_parms(placeholder="value")

multi = create_multi_value_prompts("Template {key}")
prompts = multi.generate_prompts([{"key": "val1"}, {"key": "val2"}])
```

### Tools
```python
doc = load_content("file.pdf")
chunks = chunk_by_max_chunk_size(text, max_chunk_size=1000, preserve_sentence_structure=True)
chunks = chunk_by_semantics(text, embeddings_model, threshold_percentage=90)
results = search_with_serper_api("query", num_results=5)
transcript = get_youtube_transcript("youtube_url")
```

---

## Common Model Names

**LLM Models:**
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- Gemini: `gemini-1.5-pro`, `gemini-1.5-flash`
- Cohere: `command-r-plus`, `command-r`
- DeepSeek: `deepseek-chat`, `deepseek-coder`

**Embedding Models:**
- OpenAI: `text-embedding-3-large` (3072d), `text-embedding-3-small` (1536d)
- Voyage: `voyage-3-large` (2048d), `voyage-3` (1024d)
- Cohere: `embed-english-v3.0` (1024d)

---

**For detailed documentation:** [SimplerLLM_Complete_Reference.md](SimplerLLM_Complete_Reference.md)
**Official docs:** https://docs.simplerllm.com/
