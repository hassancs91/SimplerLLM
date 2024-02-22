# SimplerLLM

SimplerLLM is an open-source Python library designed to simplify interactions with Large Language Models (LLMs) for researchers and beginners. It offers a unified interface for different LLM providers and a suite of tools to enhance language model capabilities.

## Features

- **Unified LLM Interface**: Define an LLM instance in one line for providers like OpenAI and Google Gemini. Future versions will support more APIs and LLM providers.
- **Generic Text Loader**: Load text from various sources like DOCX, PDF, TXT files, YouTube scripts, or blog posts.
- **RapidAPI Connector**: Connect with AI services on RapidAPI.
- **SERP Integration**: Perform searches using DuckDuckGo, with more search engines coming soon.
- **Prompt Template Builder**: Easily create and manage prompt templates.

## Installation

[Instructions for installation]

## Usage

### Creating an LLM Instance

```python
from SimplerLLM import LLM, LLMProvider

# For OpenAI
llm_instance = LLM.create(provider=LLMProvider.OPENAI)
# For Google Gemini
gemini_instance = LLM.create(provider=LLMProvider.GEMENI, model_name="gemini-pro")

response = llm_instance.generate_text(user_prompt="generate a 5 words sentence")
