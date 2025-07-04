from SimplerLLM.language.llm import LLM, LLMProvider
import os

def test_cohere_basic():
    """Test basic Cohere functionality"""
    print("Testing Cohere integration...")
    
    # Check if API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("COHERE_API_KEY not found in environment variables")
        return
    
    # Initialize the LLM
    llm_instance = LLM.create(
        provider=LLMProvider.COHERE,
        model_name="command-r-plus",
        verbose=True
    )
    
    if llm_instance is None:
        print("Failed to create Cohere LLM instance")
        return
    
    print("Cohere LLM instance created successfully")
    
    # Test basic text generation
    try:
        response = llm_instance.generate_response(
            prompt="What is the capital of France?",
            max_tokens=50
        )
        print(f"Basic response: {response}")
    except Exception as e:
        print(f"Error in basic generation: {e}")
    
    # Test with messages
    try:
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        response = llm_instance.generate_response(
            messages=messages,
            max_tokens=50
        )
        print(f"Messages response: {response}")
    except Exception as e:
        print(f"Error in messages generation: {e}")
    
    # Test full response
    try:
        response = llm_instance.generate_response(
            prompt="Explain Python in one sentence.",
            max_tokens=30,
            full_response=True
        )
        print(f"Full response type: {type(response)}")
        if hasattr(response, 'generated_text'):
            print(f"Generated text: {response.generated_text}")
            print(f"Model: {response.model}")
            print(f"Input tokens: {response.input_token_count}")
            print(f"Output tokens: {response.output_token_count}")
    except Exception as e:
        print(f"Error in full response generation: {e}")

if __name__ == "__main__":
    test_cohere_basic()