# Dynamic LLM for SimplerLLM

This package provides a stateless interface for chatting with multiple LLM models simultaneously and dynamically switching between providers, models, and parameters during a conversation.

## Files

- `dynamic_llm.py`: Core implementation of the DynamicLLM class and related models
- `fastapi_app.py`: FastAPI application that exposes the DynamicLLM functionality as REST endpoints
- `dynamic_llm_usage_example.py`: Examples of how to use the DynamicLLM class directly

## Prerequisites

- Python 3.8+
- SimplerLLM library installed
- API keys for the LLM providers you want to use (OpenAI, Anthropic, etc.)

## Installation

1. Make sure you have SimplerLLM installed:
   ```bash
   pip install simplerllm
   ```

2. Install additional dependencies:
   ```bash
   pip install fastapi uvicorn python-dotenv
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   # Add other API keys as needed
   ```

## Usage

### Using the DynamicLLM Class Directly

```python
from dynamic_llm import DynamicLLM, MultiLLMRequest, LLMConfig, Message

# Create a request with multiple LLM configurations
request = MultiLLMRequest(
    history=[
        Message(role="system", content="You are a helpful AI assistant"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ],
    message="How are you?",
    llm_configs={
        "gpt": LLMConfig(
            provider="OPENAI",
            model="gpt-4o-mini",
            temperature=0.7,
            api_key="your_openai_api_key"
        ),
        "claude": LLMConfig(
            provider="ANTHROPIC",
            model="claude-3-haiku-20240307",
            temperature=0.5,
            api_key="your_anthropic_api_key"
        )
    },
    max_tokens=1000,
    top_p=0.95
)

# Get responses from multiple models
responses = DynamicLLM.chat_with_multiple_llms(request, verbose=True)

# Process each response
for model_id, response in responses.items():
    print(f"--- {model_id} ---")
    print(response)
```

### Running the FastAPI Application

1. Start the FastAPI server:
   ```bash
   python fastapi_app.py
   ```

2. The API will be available at http://localhost:8000

3. You can access the API documentation at http://localhost:8000/docs

### API Endpoints

- `POST /chat`: Chat with multiple LLM models simultaneously
- `POST /chat/async`: Chat with multiple LLM models using async processing
- `GET /`: Get API information and example usage

## Example Request

```json
{
  "history": [
    {"role": "system", "content": "You are a helpful AI assistant"},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
  ],
  "message": "What can you tell me about Python programming?",
  "llm_configs": {
    "gpt": {
      "provider": "OPENAI",
      "model": "gpt-4o-mini",
      "temperature": 0.7,
      "top_p": 0.9
    },
    "claude": {
      "provider": "ANTHROPIC",
      "model": "claude-3-haiku-20240307",
      "temperature": 0.5
    }
  },
  "max_tokens": 1000,
  "top_p": 0.95
}
```

## Example Response

```json
{
  "responses": {
    "gpt": "Python is a high-level, interpreted programming language...",
    "claude": "Python is a versatile programming language known for..."
  }
}
```

## Features

- Chat with multiple LLM models simultaneously
- Compare responses from different models
- Dynamically switch between providers and models
- Customize parameters for each model
- Maintain conversation history
- Stateless design for easy integration
- Async support for better performance

## Customization

You can easily customize the DynamicLLM class to add support for additional LLM providers or features by modifying the `dynamic_llm.py` file.

## Moving to Another Project

These files are designed to be easily portable. To use them in another project:

1. Copy the files to your project directory
2. Make sure SimplerLLM is installed
3. Update the imports if necessary
4. Use the DynamicLLM class or FastAPI application as needed
