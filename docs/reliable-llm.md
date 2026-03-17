# ReliableLLM

Automatic fallback between two LLM providers.

## Setup

```python
from SimplerLLM.language.llm import LLM, LLMProvider, ReliableLLM

primary = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
secondary = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")

reliable = ReliableLLM(primary_llm=primary, secondary_llm=secondary)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_llm` | `LLM` | Required | Primary provider |
| `secondary_llm` | `LLM` | Required | Fallback provider |
| `verbose` | `bool` | `False` | Enable debug logging |
| `skip_validation` | `bool` | `False` | Skip provider validation |
| `lazy_validation` | `bool` | `False` | Defer validation to first call |

## Validation Modes

By default, both providers are validated at initialization.

```python
# Lazy validation - validates on first request
reliable = ReliableLLM(
    primary_llm=primary,
    secondary_llm=secondary,
    lazy_validation=True
)

# Skip validation entirely
reliable = ReliableLLM(
    primary_llm=primary,
    secondary_llm=secondary,
    skip_validation=True
)
```

## Usage

### Basic Generation

```python
response = reliable.generate_response(
    prompt="What is Python?",
    max_tokens=200
)
```

If the primary provider fails, the secondary is used automatically.

### Track Which Provider Was Used

```python
response, provider, model = reliable.generate_response(
    prompt="What is Python?",
    return_provider=True
)

print(f"Response from: {provider.name} ({model})")
```

### Async Generation

```python
import asyncio

async def main():
    response = await reliable.generate_response_async(
        prompt="Explain machine learning",
        max_tokens=300
    )
    return response

result = asyncio.run(main())
```

### Full Response with Metadata

```python
response, provider, model = reliable.generate_response(
    prompt="What is 2+2?",
    return_provider=True,
    full_response=True
)

print(f"Text: {response.generated_text}")
print(f"Tokens: {response.input_token_count} in, {response.output_token_count} out")
print(f"Provider: {provider.name}")
```

## Provider Capabilities

Parameters are automatically filtered based on the provider. Provider-specific parameters are ignored when not supported.

| Provider | Supported Parameters |
|----------|---------------------|
| OpenAI | `reasoning_effort`, `timeout`, `web_search`, `images`, `detail` |
| Anthropic | `thinking_budget`, `prompt_caching`, `cached_input`, `web_search`, `images` |
| Gemini | `thinking_budget`, `thinking_level`, `prompt_caching`, `cache_id`, `response_schema`, `timeout`, `web_search`, `images` |
| DeepSeek | `thinking`, `images` |
| Perplexity | `timeout`, `search_domain_filter`, `search_recency_filter`, `return_images`, `return_related_questions`, `images` |
| OpenRouter | `reasoning_effort`, `timeout`, `site_url`, `site_name`, `images`, `detail` |
| Ollama | `images` |

Universal parameters (`prompt`, `messages`, `system_prompt`, `temperature`, `max_tokens`, `top_p`, `json_mode`, `full_response`) work with all providers.

> **Note:** Enable `verbose=True` to see which parameters are filtered for each provider.
