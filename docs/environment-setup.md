# Environment Setup

SimplerLLM uses environment variables for API keys and configuration. You can set them in a `.env` file, export them in your shell, or pass them directly in code.

## API Keys

Set the API key for each provider you plan to use:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Perplexity | `PERPLEXITY_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Moonshot | `MOONSHOT_API_KEY` |
| Voyage AI (embeddings) | `VOYAGE_API_KEY` |

> **Note:** Ollama and Hugging Face local models don't require API keys.

## Using a .env File

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GEMINI_API_KEY=your-key-here
```

SimplerLLM loads the `.env` file automatically. No extra code needed.

## Passing Keys Directly

You can also pass API keys directly when creating an LLM instance:

```python
llm = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o",
    api_key="sk-your-key-here"
)
```

This overrides any environment variable.

## Global Settings

Optional settings that apply to all providers:

```env
MAX_RETRIES=3      # Number of retry attempts on failure (default: 3)
RETRY_DELAY=2      # Seconds between retries (default: 2)
```

## Local Models

### Ollama

```env
OLLAMA_URL=http://localhost:11434/   # Ollama server URL (default)
OLLAMA_TIMEOUT=120                    # Request timeout in seconds
```

### Hugging Face Local

```env
HF_DEVICE=auto           # "cuda", "cpu", "mps", or "auto"
HF_TORCH_DTYPE=auto      # "float16", "bfloat16", "float32", or "auto"
HF_TIMEOUT=300           # Generation timeout in seconds
HF_LOAD_IN_4BIT=false    # Enable 4-bit quantization
HF_LOAD_IN_8BIT=false    # Enable 8-bit quantization
```

## OpenRouter

Optional metadata for request tracking:

```env
OPENROUTER_API_KEY=your-key-here
OPENROUTER_SITE_URL=https://yoursite.com
OPENROUTER_SITE_NAME=Your App Name
```
