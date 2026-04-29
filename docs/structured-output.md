# Structured Output

Generate validated Pydantic models from any LLM provider.

## Basic Usage

```python
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    recommended: bool

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

review = generate_pydantic_json_model(
    model_class=MovieReview,
    prompt="Write a review for the movie Inception",
    llm_instance=llm
)

print(review.title)       # "Inception"
print(review.rating)      # 4.8
print(review.recommended) # True
```

The result is a validated `MovieReview` instance, not raw JSON.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_class` | `Type[BaseModel]` | Required | Pydantic model class to generate |
| `prompt` | `str` | Required | The input prompt |
| `llm_instance` | `LLM` | Required | LLM instance to use |
| `max_retries` | `int` | `3` | Retries on validation failure |
| `max_tokens` | `int` | `4096` | Maximum output tokens |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `top_p` | `float` | `1.0` | Nucleus sampling |
| `full_response` | `bool` | `False` | Return `LLMFullResponse` with metadata |
| `images` | `list` | `None` | List of image URLs or local file paths for vision |
| `detail` | `str` | `"auto"` | Image detail level: `"low"`, `"high"`, or `"auto"` |
| `web_search` | `bool` | `False` | Enable web search before generation |
| `reasoning_effort` | `str` | `None` | `"low"`, `"medium"`, or `"high"` (OpenAI thinking models) |

> **Note:** Validation failures trigger automatic retries with exponential backoff.

## Nested Models

```python
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

class Author(BaseModel):
    name: str
    bio: str

class BlogPost(BaseModel):
    title: str
    author: Author
    tags: list[str]
    word_count: int

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

post = generate_pydantic_json_model(
    model_class=BlogPost,
    prompt="Generate a blog post about async Python programming",
    llm_instance=llm
)

print(post.title)
print(post.author.name)
print(post.tags)
```

Nested models are automatically handled -- the LLM output is validated against the full schema.

## Generating Lists

Use `RootModel` to generate JSON arrays directly.

```python
from typing import List
from pydantic import BaseModel, RootModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

class FAQItem(BaseModel):
    question: str
    answer: str

class FAQList(RootModel[List[FAQItem]]):
    pass

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

faqs = generate_pydantic_json_model(
    model_class=FAQList,
    prompt="Generate 3 FAQs about machine learning",
    llm_instance=llm
)

for item in faqs.root:
    print(f"Q: {item.question}")
    print(f"A: {item.answer}")
```

> **Note:** `RootModel` requires Pydantic v2. It lets you generate arrays directly instead of wrapping them in an object.

## Full Response with Metadata

Set `full_response=True` to get token counts and timing:

```python
response = generate_pydantic_json_model(
    model_class=MovieReview,
    prompt="Review the movie Interstellar",
    llm_instance=llm,
    full_response=True
)

print(response.model_object.title)
print(f"Input tokens: {response.input_token_count}")
print(f"Output tokens: {response.output_token_count}")
print(f"Time: {response.process_time:.2f}s")
```

## Vision / Image Input

Pass images via the `images` parameter to extract structured data from visual content:

```python
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

class ImageDescription(BaseModel):
    objects: list[str]
    scene: str
    mood: str

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Using a URL
result = generate_pydantic_json_model(
    model_class=ImageDescription,
    prompt="Describe this image in detail",
    llm_instance=llm,
    images=["https://example.com/photo.jpg"]
)

print(result.scene)
print(result.objects)
```

You can also use local file paths and pass multiple images:

```python
result = generate_pydantic_json_model(
    model_class=ImageDescription,
    prompt="Describe what you see",
    llm_instance=llm,
    images=["path/to/image1.png", "path/to/image2.jpg"],
    detail="high"  # "low", "high", or "auto"
)
```

| Provider | Vision Support |
|----------|---------------|
| OpenAI | Supported |
| Anthropic | Supported |
| Gemini | Supported |
| Ollama | Supported (llava, llama3.2-vision) |
| Cohere | Supported |

## Web Search

Enable `web_search=True` to ground output in real-time data:

```python
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

class ResearchSummary(BaseModel):
    topic: str
    key_findings: list[str]
    sources_used: int

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

response = generate_pydantic_json_model(
    model_class=ResearchSummary,
    prompt="Summarize recent advances in quantum computing",
    llm_instance=llm,
    web_search=True,
    full_response=True
)

print(response.model_object.key_findings)

for source in response.web_sources:
    print(f"{source['title']}: {source['url']}")
```

| Provider | Web Search |
|----------|-----------|
| OpenAI | Supported |
| Anthropic | Supported |
| Gemini | Supported |
| Perplexity | Always enabled |

## ReliableLLM Fallback

Use `generate_pydantic_json_model_reliable` to automatically fall back to a secondary provider:

```python
from SimplerLLM.language.llm import LLM, LLMProvider, ReliableLLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_reliable

primary = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
secondary = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")
reliable = ReliableLLM(primary_llm=primary, secondary_llm=secondary)

review, provider, model_name = generate_pydantic_json_model_reliable(
    model_class=MovieReview,
    prompt="Review the movie The Matrix",
    reliable_llm=reliable
)

print(f"{review.title} (from {provider.name} / {model_name})")
```

If the primary provider fails, the secondary is used automatically.

## Async Usage

```python
import asyncio
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_async

async def main():
    review = await generate_pydantic_json_model_async(
        model_class=MovieReview,
        prompt="Review the movie Dune",
        llm_instance=llm
    )
    print(review.title)

asyncio.run(main())
```

> **Note:** The async reliable variant is `generate_pydantic_json_model_reliable_async`.

## Error Handling

On failure, the function returns an error string instead of a model instance.

```python
result = generate_pydantic_json_model(
    model_class=MovieReview,
    prompt="Review a movie",
    llm_instance=llm
)

if isinstance(result, str):
    print(f"Error: {result}")
else:
    print(result.title)
```
