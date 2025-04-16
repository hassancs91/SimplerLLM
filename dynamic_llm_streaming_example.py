"""
Example of using DynamicLLM with streaming responses.

Note: This example assumes that the underlying LLM providers support streaming.
Some providers may not support streaming, in which case this example will not work.
"""
import os
import asyncio
from dotenv import load_dotenv
from dynamic_llm import MultiLLMRequest, LLMConfig, Message
from SimplerLLM.language.llm import LLM, LLMProvider

# Load environment variables from .env file
load_dotenv()

async def stream_response(provider_enum, model_name, messages, max_tokens=1000):
    """
    Stream a response from an LLM provider.
    
    Args:
        provider_enum: The LLMProvider enum value
        model_name: The name of the model to use
        messages: The messages to send to the model
        max_tokens: The maximum number of tokens to generate
        
    Returns:
        An async generator that yields chunks of the response
    """
    # Create the LLM instance
    llm = LLM.create(
        provider=provider_enum,
        model_name=model_name,
        temperature=0.7,
        api_key=os.getenv(f"{provider_enum.name}_API_KEY")
    )
    
    # Check if the provider supports streaming
    if not hasattr(llm, 'generate_response_streaming'):
        print(f"Provider {provider_enum.name} does not support streaming")
        response = await llm.generate_response_async(
            messages=messages,
            max_tokens=max_tokens
        )
        yield response
        return
    
    # Stream the response
    async for chunk in llm.generate_response_streaming(
        messages=messages,
        max_tokens=max_tokens
    ):
        yield chunk

async def main():
    """
    Example of streaming responses from multiple LLM models.
    """
    print("Dynamic LLM Streaming Example")
    print("=============================\n")
    
    # Define the message history
    history = [
        Message(role="system", content="You are a helpful AI assistant"),
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm doing well, thank you for asking!")
    ]
    
    # Define the new message
    message = "Write a short poem about artificial intelligence."
    
    # Format messages for LLM processing
    formatted_messages = [{"role": msg.role, "content": msg.content} for msg in history]
    formatted_messages.append({"role": "user", "content": message})
    
    # Define the models to use
    models = [
        {"id": "gpt", "provider": LLMProvider.OPENAI, "model": "gpt-4o-mini"},
        {"id": "claude", "provider": LLMProvider.ANTHROPIC, "model": "claude-3-haiku-20240307"}
    ]
    
    # Stream responses from each model
    for model in models:
        print(f"\n--- {model['id']} ({model['provider'].name}, {model['model']}) ---")
        print("Streaming response: ", end="", flush=True)
        
        try:
            async for chunk in stream_response(
                model["provider"],
                model["model"],
                formatted_messages
            ):
                # In a real application, you would handle the chunks differently
                # For this example, we just print them to the console
                print(chunk, end="", flush=True)
            
            print()  # Add a newline after the response
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
