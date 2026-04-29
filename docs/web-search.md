# Web Search

Ground LLM responses in real-time web data.

## Basic Usage

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

response = llm.generate_response(
    prompt="What are the latest developments in AI?",
    web_search=True,
    full_response=True
)

print(response.generated_text)
```

> **Note:** Set `full_response=True` to access web sources. Without it, only the generated text is returned.

## Supported Providers

| Provider | Enum Value | Example Model | Web Search |
|----------|-----------|---------------|------------|
| OpenAI | `LLMProvider.OPENAI` | `gpt-4o` | `web_search=True` |
| Anthropic | `LLMProvider.ANTHROPIC` | `claude-sonnet-4-5-20250929` | `web_search=True` |
| Google Gemini | `LLMProvider.GEMINI` | `gemini-2.5-flash` | `web_search=True` |
| Perplexity | `LLMProvider.PERPLEXITY` | `sonar-pro` | Always enabled |

## Accessing Sources

Web sources are available on the `web_sources` field when using `full_response=True`:

```python
response = llm.generate_response(
    prompt="Recent breakthroughs in quantum computing",
    web_search=True,
    full_response=True
)

for source in response.web_sources or []:
    print(f"{source['title']}: {source['url']}")
```

Each source contains:

| Key | Description |
|-----|-------------|
| `title` | Source page title |
| `url` | Source URL |

## Switching Providers

The `web_search` parameter works the same across providers:

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# OpenAI
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Anthropic
llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-5-20250929")

# Gemini
llm = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-2.5-flash")
```

All use the same call:

```python
response = llm.generate_response(
    prompt="Latest news on renewable energy",
    web_search=True,
    full_response=True
)
```

## Perplexity

Perplexity models always include web search -- no need to set `web_search=True`.

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create(provider=LLMProvider.PERPLEXITY, model_name="sonar-pro")

response = llm.generate_response(
    prompt="What happened in tech this week?",
    search_recency_filter="week",
    search_domain_filter=["techcrunch.com", "theverge.com"],
    full_response=True
)

print(response.generated_text)

for source in response.web_sources or []:
    print(f"{source['title']}: {source['url']}")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `search_domain_filter` | `list[str]` | Limit to specific domains (prefix with `-` to exclude) |
| `search_recency_filter` | `str` | `"day"`, `"week"`, `"month"`, or `"year"` |
| `return_images` | `bool` | Include images in search results |
| `return_related_questions` | `bool` | Suggest related queries |

Available models: `sonar`, `sonar-pro`, `sonar-reasoning-pro`, `sonar-deep-research`.

## Structured Output with Web Search

Combine web search with Pydantic model generation:

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

for source in response.web_sources or []:
    print(f"{source['title']}: {source['url']}")
```

See [Structured Output](structured-output.md) for more on Pydantic model generation.

## Vision + Web Search

Combine image analysis with web search to get context-aware responses about visual content:

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

response = llm.generate_response(
    prompt="What is this landmark and what are its current visiting hours?",
    images=["https://example.com/landmark.jpg"],
    web_search=True,
    full_response=True
)

print(response.generated_text)

for source in response.web_sources or []:
    print(f"{source['title']}: {source['url']}")
```

This also works with structured output:

```python
from pydantic import BaseModel
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

class LandmarkInfo(BaseModel):
    name: str
    location: str
    visiting_hours: str
    description: str

result = generate_pydantic_json_model(
    model_class=LandmarkInfo,
    prompt="Identify this landmark and find its current visiting hours",
    llm_instance=llm,
    images=["path/to/photo.jpg"],
    web_search=True
)

print(result.name)
print(result.visiting_hours)
```

## Async Usage

```python
import asyncio
from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

async def main():
    response = await llm.generate_response_async(
        prompt="Latest AI safety research",
        web_search=True,
        full_response=True
    )
    print(response.generated_text)

asyncio.run(main())
```
