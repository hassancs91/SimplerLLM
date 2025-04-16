from SimplerLLM.language.llm import DynamicLLM, MultiLLMRequest, LLMConfig, Message, LLMProvider
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    # Example of using DynamicLLM to chat with multiple models
    
    # Create a request with multiple LLM configurations
    request = MultiLLMRequest(
        history=[
            Message(role="system", content="You are a helpful AI assistant"),
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you for asking!")
        ],
        message="What can you tell me about Python programming?",
        llm_configs={
            "gpt": LLMConfig(
                provider="OPENAI",
                model="gpt-4o-mini",
                temperature=0.7,
                top_p=0.9,
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            "claude": LLMConfig(
                provider="ANTHROPIC",
                model="claude-3-haiku-20240307",
                temperature=0.5,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        },
        max_tokens=1000,
        top_p=0.95
    )
    
    # Get responses from multiple models
    try:
        responses = DynamicLLM.chat_with_multiple_llms(request, verbose=True)
        
        print("\n=== Responses ===\n")
        for model_id, response in responses.items():
            print(f"--- {model_id} ---")
            print(response)
            print()
            
    except Exception as e:
        print(f"Error: {str(e)}")

def fastapi_example():
    """
    Example of how to implement a FastAPI endpoint using DynamicLLM
    """
    # This is just example code, not meant to be run directly
    from fastapi import FastAPI, HTTPException

    app = FastAPI()

    @app.post("/chat")
    async def chat_endpoint(request: MultiLLMRequest):
        try:
            responses = DynamicLLM.chat_with_multiple_llms(request)
            return {"responses": responses}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    main()
