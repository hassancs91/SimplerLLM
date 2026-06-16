# LLM Interface

Create and use LLM instances to generate text with various providers.

## Creating an LLM Instance

```python
from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
response = llm.generate_response(prompt="What is Python?")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `LLMProvider` | `OPENAI` | The LLM provider to use |
| `model_name` | `str` | `"gpt-4o-mini"` | Model identifier |
| `temperature` | `float` | `0.7` | Sampling temperature (0-2) |
| `top_p` | `float` | `1.0` | Nucleus sampling (0-1) |
| `api_key` | `str` | `None` | API key (defaults to environment variable) |

## Providers

| Provider | Enum Value | Environment Variable |
|----------|------------|---------------------|
| OpenAI | `LLMProvider.OPENAI` | `OPENAI_API_KEY` |
| Google Gemini | `LLMProvider.GEMINI` | `GEMINI_API_KEY` |
| Anthropic | `LLMProvider.ANTHROPIC` | `ANTHROPIC_API_KEY` |
| DeepSeek | `LLMProvider.DEEPSEEK` | `DEEPSEEK_API_KEY` |
| Ollama | `LLMProvider.OLLAMA` | — (local) |
| OpenRouter | `LLMProvider.OPENROUTER` | `OPENROUTER_API_KEY` |
| CometAPI | `LLMProvider.COMETAPI` | `COMETAPI_API_KEY` (or `COMETAPI_KEY`) |
| Cohere | `LLMProvider.COHERE` | `COHERE_API_KEY` |
| Perplexity | `LLMProvider.PERPLEXITY` | `PERPLEXITY_API_KEY` |
| Moonshot | `LLMProvider.MOONSHOT` | `MOONSHOT_API_KEY` |
| HuggingFace Local | `LLMProvider.HUGGING_FACE_LOCAL` | — (local) |

```python
# Example: Using Anthropic
llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")

# Example: Using local Ollama
llm = LLM.create(provider=LLMProvider.OLLAMA, model_name="llama3")
```

## Generation Methods

### generate_response

Generate text synchronously.

```python
# Using a prompt
response = llm.generate_response(prompt="Explain quantum computing")

# Using chat messages
response = llm.generate_response(
    messages=[
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence..."},
        {"role": "user", "content": "Give me an example"}
    ]
)
```

### generate_response_async

Generate text asynchronously.

```python
import asyncio

async def main():
    response = await llm.generate_response_async(prompt="Explain machine learning")
    print(response)

asyncio.run(main())
```

## Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | `None` | Text input (use this OR messages) |
| `messages` | `list[dict]` | `None` | Chat history (use this OR prompt) |
| `system_prompt` | `str` | `"You are a helpful AI Assistant"` | System instruction |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `300` | Maximum output tokens |
| `top_p` | `float` | `1.0` | Nucleus sampling |
| `json_mode` | `bool` | `False` | Force JSON output |
| `full_response` | `bool` | `False` | Return detailed response object |
| `images` | `list[str]` | `None` | List of image URLs or local file paths for vision |
| `detail` | `str` | `"auto"` | Image detail level: `"low"`, `"high"`, or `"auto"` |
| `web_search` | `bool` | `False` | Enable web search |

### Vision / Image Input

Pass images to analyze visual content:

```python
# Using an image URL
response = llm.generate_response(
    prompt="Describe this image",
    images=["https://example.com/photo.jpg"]
)

# Using a local file
response = llm.generate_response(
    prompt="What objects are in these images?",
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
| DeepSeek | Supported |
| OpenRouter | Supported (vision-capable models) |
| CometAPI | Supported (vision-capable models) |

### JSON Mode

```python
response = llm.generate_response(
    prompt="List 3 programming languages as JSON",
    json_mode=True
)
```

### Full Response

Get detailed metadata including token counts and timing.

```python
response = llm.generate_response(
    prompt="What is Python?",
    full_response=True
)

print(response.generated_text)
print(f"Input tokens: {response.input_token_count}")
print(f"Output tokens: {response.output_token_count}")
print(f"Time: {response.process_time}s")
```

> **Note:** Use `prompt` for single queries or `messages` for conversations. Do not use both.
