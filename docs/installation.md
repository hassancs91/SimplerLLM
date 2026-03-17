# Installation

## Requirements

- Python 3.6 or higher

## Install

```bash
pip install simplerllm
```

## Optional: Voice Features

For audio playback and voice input capabilities:

```bash
pip install simplerllm[voice]
```

> **Note:** Voice features require PortAudio on your system. On Windows, this is included automatically. On Linux, install with `sudo apt-get install portaudio19-dev`. On macOS, use `brew install portaudio`.

## API Keys

SimplerLLM works with multiple LLM providers. Set the API keys for the providers you plan to use:

```bash
# OpenAI
OPENAI_API_KEY=your-key-here

# Anthropic (Claude)
ANTHROPIC_API_KEY=your-key-here

# Google (Gemini)
GEMENI_API_KEY=your-key-here

# Cohere
COHERE_API_KEY=your-key-here
```

You can set these in a `.env` file in your project root. SimplerLLM will load them automatically.

## Verify Installation

```python
import simplerllm
print(simplerllm.__version__)
```
