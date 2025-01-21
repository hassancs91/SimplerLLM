---
sidebar_position: 3
---

# Consistent JSON with LLMs

This section introduces how SimplerLLM helps ensure a consistent JSON-structured response from LLMs. This functionality is useful when integrating LLM outputs into your software, where maintaining a stable JSON format is important for processing or automation.

The feature uses Pydantic models to validate and standardize LLM outputs, ensuring seamless integration into your applications.

In this way you won't need to include in every prompt you give the LLM, how it should make your output in a json structure and in which format.

## Key Functions

SimplerLLM offers two functions for this purpose: one synchronous and one asynchronous. Both rely on a Pydantic model that you define, which acts as the structure for the LLM's response.

> **Note:** You can use any LLM provider by modifying the `llm_instance` variable to include the provider of your choice. To learn more about setting up different LLM providers, refer to the [Choose the Right LLM](./Choose the LLM.md) page in this documentation.

### Synchronous Function

The synchronous function is `generate_pydantic_json_model`, and here's how you can use it:

```python
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

# Define your Pydantic model
class LLMResponse(BaseModel):
    response: str

# Initialize the LLM instance
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
prompt = "Generate a sentence about the importance of AI"

# Generate and parse the JSON response
output = generate_pydantic_json_model(
    llm_instance=llm_instance,
    prompt=prompt,
    model_class=LLMResponse
)
json_output = output.model_dump()
```

The `output` is an object of type `LLMResponse`, and the `model_dump()` method converts it into a dictionary or JSON-like format.

### Asynchronous Function

For asynchronous applications, the `generate_pydantic_json_model_async` function provides the same functionality but in an async context. Here's an example:

```python
import asyncio
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_async

# Define your Pydantic model
class LLMResponse(BaseModel):
    response: str

# Initialize the LLM instance
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
prompt = "Generate a sentence about the importance of AI"

# Asynchronous usage
async def main():
    output = generate_pydantic_json_model_async(
        llm_instance=llm_instance,
        prompt=prompt,
        model_class=LLMResponse
    )
    json_output = output.model_dump()
    return json_output

asyncio.run(main())
```

The asynchronous function is ideal for use cases where you need to fetch results without blocking other operations in your application.

By using these functions, you can effortlessly maintain a stable JSON structure in your LLM responses, making interaction with LLM providers Simpler!