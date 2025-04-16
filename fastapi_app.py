from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import uvicorn
import os
from dotenv import load_dotenv

# Import from our dynamic_llm module
from dynamic_llm import DynamicLLM, MultiLLMRequest, LLMConfig, Message

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Dynamic LLM API", description="API for chatting with multiple LLM models")

@app.post("/chat")
async def chat_endpoint(request: MultiLLMRequest):
    """
    Chat with multiple LLM models simultaneously.
    
    This endpoint accepts a request with conversation history, a new message,
    and configurations for one or more LLM models. It returns responses from
    all specified models.
    """
    try:
        # Process the request with DynamicLLM
        responses = DynamicLLM.chat_with_multiple_llms(request, verbose=True)
        
        # Return the responses
        return {"responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/async")
async def chat_async_endpoint(request: MultiLLMRequest):
    """
    Chat with multiple LLM models simultaneously using async processing.
    
    This endpoint is similar to /chat but uses asynchronous processing,
    which can be more efficient when dealing with multiple models.
    """
    try:
        # Process the request asynchronously with DynamicLLM
        responses = await DynamicLLM.chat_with_multiple_llms_async(request, verbose=True)
        
        # Return the responses
        return {"responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example request model for documentation
class ExampleRequest(BaseModel):
    example: Dict = Field(
        default={
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
    )

@app.get("/")
def read_root():
    """
    Root endpoint providing API information and example usage.
    """
    return {
        "api": "Dynamic LLM API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "Chat with multiple LLM models simultaneously",
            "/chat/async": "Chat with multiple LLM models using async processing"
        },
        "example_request": ExampleRequest().example
    }

if __name__ == "__main__":
    # Run the FastAPI application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
