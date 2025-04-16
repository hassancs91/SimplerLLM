import os
from dotenv import load_dotenv
from dynamic_llm import DynamicLLM, MultiLLMRequest, LLMConfig, Message
from SimplerLLM.language.llm import LLMProvider

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Example of using the DynamicLLM class to chat with multiple LLM models.
    """
    print("Dynamic LLM Usage Example")
    print("=========================\n")
    
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
    
    print("Sending request to multiple LLM models...")
    print(f"Message: {request.message}")
    print(f"Models: {', '.join(request.llm_configs.keys())}")
    print()
    
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

def single_model_example():
    """
    Example of using the DynamicLLM class with a single LLM model.
    """
    print("Single Model Example")
    print("===================\n")
    
    # Create a request with a single LLM configuration
    request = MultiLLMRequest(
        message="Explain the concept of recursion in programming",
        llm_configs=LLMConfig(
            provider="OPENAI",
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        max_tokens=500
    )
    
    print("Sending request to a single LLM model...")
    print(f"Message: {request.message}")
    print(f"Model: {request.llm_configs.provider} - {request.llm_configs.model}")
    print()
    
    # Get response from the model
    try:
        responses = DynamicLLM.chat_with_multiple_llms(request, verbose=True)
        
        print("\n=== Response ===\n")
        for model_id, response in responses.items():
            print(response)
            
    except Exception as e:
        print(f"Error: {str(e)}")

def conversation_example():
    """
    Example of having a multi-turn conversation with multiple LLM models.
    """
    print("Conversation Example")
    print("===================\n")
    
    # Initialize conversation history
    history = [
        Message(role="system", content="You are a helpful AI assistant")
    ]
    
    # Define the models to use
    llm_configs = {
        "gpt": LLMConfig(
            provider="OPENAI",
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        "claude": LLMConfig(
            provider="ANTHROPIC",
            model="claude-3-haiku-20240307",
            temperature=0.5,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    }
    
    # First turn
    request1 = MultiLLMRequest(
        history=history,
        message="What are the main features of Python?",
        llm_configs=llm_configs,
        max_tokens=500
    )
    
    print("Turn 1: What are the main features of Python?")
    responses1 = DynamicLLM.chat_with_multiple_llms(request1, verbose=False)
    
    # Update history with the first turn
    history.append(Message(role="user", content=request1.message))
    for model_id, response in responses1.items():
        print(f"\n--- {model_id} ---")
        print(response)
        # Only add one model's response to the history (using GPT for this example)
        if model_id == "gpt":
            history.append(Message(role="assistant", content=response))
    
    # Second turn
    request2 = MultiLLMRequest(
        history=history,
        message="How does Python handle memory management?",
        llm_configs=llm_configs,
        max_tokens=500
    )
    
    print("\nTurn 2: How does Python handle memory management?")
    responses2 = DynamicLLM.chat_with_multiple_llms(request2, verbose=False)
    
    # Display responses from the second turn
    for model_id, response in responses2.items():
        print(f"\n--- {model_id} ---")
        print(response)

if __name__ == "__main__":
    main()
    # Uncomment to run other examples
    # single_model_example()
    # conversation_example()
