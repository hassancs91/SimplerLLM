# Quick Start

Generate text with any LLM provider using a unified interface.

## Basic Usage

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# Create an LLM instance
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Generate text
response = llm.generate_response(prompt="What is Python?")
print(response)
```

## Switching Providers

The power of SimplerLLM is that switching providers is a one-line change. The `generate_response()` method works the same way across all providers.

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# OpenAI
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Anthropic Claude
llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-5-20250929")

# Google Gemini
llm = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-2.0-flash")

# DeepSeek
llm = LLM.create(provider=LLMProvider.DEEPSEEK, model_name="deepseek-chat")

# Ollama (local)
llm = LLM.create(provider=LLMProvider.OLLAMA, model_name="llama3")
```

All providers use the same method:

```python
response = llm.generate_response(prompt="Your prompt here")
```

## Available Providers

| Provider | Enum Value | Example Model |
|----------|------------|---------------|
| OpenAI | `LLMProvider.OPENAI` | `gpt-4o` |
| Anthropic | `LLMProvider.ANTHROPIC` | `claude-sonnet-4-5-20250929` |
| Google Gemini | `LLMProvider.GEMINI` | `gemini-2.0-flash` |
| DeepSeek | `LLMProvider.DEEPSEEK` | `deepseek-chat` |
| Cohere | `LLMProvider.COHERE` | `command-r-plus` |
| OpenRouter | `LLMProvider.OPENROUTER` | `openai/gpt-4o` |
| Perplexity | `LLMProvider.PERPLEXITY` | `sonar` |
| Ollama | `LLMProvider.OLLAMA` | `llama3` |

## Common Parameters

```python
response = llm.generate_response(
    prompt="Explain quantum computing",
    system_prompt="You are a helpful assistant",
    temperature=0.7,
    max_tokens=500
)
```

| Parameter | Description |
|-----------|-------------|
| `prompt` | The input text |
| `system_prompt` | Instructions for the model's behavior |
| `temperature` | Randomness (0.0 = deterministic, 1.0+ = creative) |
| `max_tokens` | Maximum length of the response |

## Getting Token Usage

Set `full_response=True` to get token counts and metadata:

```python
response = llm.generate_response(
    prompt="Hello!",
    full_response=True
)

print(response.generated_text)
print(f"Input tokens: {response.input_token_count}")
print(f"Output tokens: {response.output_token_count}")
```
