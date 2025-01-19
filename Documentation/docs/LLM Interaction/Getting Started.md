---
sidebar_position: 1
---

# Getting Started

The unified LLM interface in SimplerLLM allows you to easily interact with multiple Large Language Models (LLMs) through a single, consistent function. 

Whether you plan on using OpenAI, Google Gemini, Anthropic, Ollamma local model, or even our own LLM Playgound, SimplerLLM provides a clear and easy way to integrate and switch between these providers while maintaining a consistent code structure.

## Why Use a Unified Interface?

Managing multiple LLM providers can be challenging, especially when each provider has its own API structure, unique methods, and configurations. The unified interface solves this problem by standardizing the way you interact with these models, making it easier to:
- **Switch between providers**: You can switch between LLM providers by just changing some parameters in the same function, keeping the code structure as is.
- **Reduce provider dependency**: If the LLM provider you're using gets shutdown or stops working for a certain cause, you can easily switch to another LLM provider keeping the code as is.

## How It Works

SimplerLLM’s unified interface is built on the concept of defining a common `LLMProvider` and then creating instances of the `LLM` class based on that provider. The API is designed to be simple and consistent across all supported models.

### Setting Up the LLM Instance

Here’s a basic example that shows how you can easily switch between different providers:

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# For OpenAI
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")

# For Google Gemini
#llm_instance = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-1.5-flash")

# For Anthropic Claude
#llm_instance = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20240620")

# For Ollama (Local Model)
#llm_instance = LLM.create(provider=LLMProvider.OLLAMA, model_name="ollama-local-model")

# Generate a response
response = llm_instance.generate_response(prompt="generate a 5 words sentence")
print(response)
```

As you can see it's very straightforward to switch between LLMs. You just need to create and LLM instance by picking the provider you want and the model name and you're ready to use it.

After that you'll need to call the `generate_response` function which remains the same regarless of the provider you choose; passing your desired prompt to call the API and get the response.

Finally, see the response by printing it in the terminal.

As you can see, switching between LLMs is straightforward. Simply create an LLM instance by selecting the provider and model name. Once set up, you're ready to generate responses using a consistent and simple API.