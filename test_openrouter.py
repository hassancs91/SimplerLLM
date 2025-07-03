#!/usr/bin/env python3
"""
Test script for OpenRouter integration with SimplerLLM.
This script demonstrates basic usage of the OpenRouter provider.
"""

from SimplerLLM.language.llm import LLM, LLMProvider
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openrouter_basic():
    """Test basic OpenRouter functionality"""
    print("Testing OpenRouter integration...")
    
    # Check if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("Please set OPENROUTER_API_KEY in your .env file")
        return False
    
    try:
        # Create OpenRouter LLM instance
        llm_instance = LLM.create(
            provider=LLMProvider.OPENROUTER,
            model_name="meta-llama/llama-4-maverick",
            temperature=0.7,
            verbose=True
        )
        
        if llm_instance is None:
            print("❌ Failed to create OpenRouter LLM instance")
            return False
        
        print("✅ OpenRouter LLM instance created successfully")
        
        # Test basic response generation
        prompt = "Generate a simple greeting message in 10 words or less."
        response = llm_instance.generate_response(prompt=prompt, max_tokens=50)
        
        if response:
            print(f"✅ Response generated successfully: {response}")
            return True
        else:
            print("❌ No response generated")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenRouter: {str(e)}")
        return False

def test_openrouter_json_mode():
    """Test OpenRouter JSON mode functionality"""
    print("\nTesting OpenRouter JSON mode...")
    
    try:
        llm_instance = LLM.create(
            provider=LLMProvider.OPENROUTER,
            model_name="meta-llama/llama-4-maverick",
            temperature=0.7
        )
        
        prompt = "Generate a JSON object with fields: name, age, city. Use example data."
        response = llm_instance.generate_response(
            prompt=prompt,
            max_tokens=100,
            json_mode=True
        )
        
        if response:
            print(f"✅ JSON mode response: {response}")
            return True
        else:
            print("❌ No JSON response generated")
            return False
            
    except Exception as e:
        print(f"❌ Error testing JSON mode: {str(e)}")
        return False

def test_openrouter_messages():
    """Test OpenRouter with message format"""
    print("\nTesting OpenRouter with messages...")
    
    try:
        llm_instance = LLM.create(
            provider=LLMProvider.OPENROUTER,
            model_name="meta-llama/llama-4-maverick",
            temperature=0.7
        )
        
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "What's the weather like today?"}
        ]
        
        response = llm_instance.generate_response(
            messages=messages,
            max_tokens=50
        )
        
        if response:
            print(f"✅ Messages response: {response}")
            return True
        else:
            print("❌ No messages response generated")
            return False
            
    except Exception as e:
        print(f"❌ Error testing messages: {str(e)}")
        return False

def test_openrouter_different_models():
    """Test OpenRouter with different models"""
    print("\nTesting OpenRouter with different models...")
    
    models_to_test = [
        "baidu/ernie-4.5-300b-a47b",
        "thedrummer/anubis-70b-v1.1",
        "mistralai/mistral-small-3.2-24b-instruct:free"
    ]
    
    for model in models_to_test:
        try:
            print(f"Testing model: {model}")
            llm_instance = LLM.create(
                provider=LLMProvider.OPENROUTER,
                model_name=model,
                temperature=0.7
            )
            
            response = llm_instance.generate_response(
                prompt="Say 'Hello' in a creative way.",
                max_tokens=30
            )
            
            if response:
                print(f"✅ {model}: {response}")
            else:
                print(f"❌ {model}: No response")
                
        except Exception as e:
            print(f"❌ {model}: Error - {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting OpenRouter integration tests...")
    print("=" * 50)
    
    # Run tests
    test_openrouter_basic()
    test_openrouter_json_mode()
    test_openrouter_messages()
    test_openrouter_different_models()
    
    print("\n" + "=" * 50)
    print("✅ OpenRouter integration tests completed!")
    print("\nNote: Some tests may fail if:")
    print("- OPENROUTER_API_KEY is not set")
    print("- API quota is exceeded")
    print("- Specific models are not available")
    print("- Network connectivity issues")