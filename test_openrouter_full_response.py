#!/usr/bin/env python3
"""
Focused test script for OpenRouter LLMFullResponse functionality.
Tests comprehensive response objects, token counting, timing, and metadata.
"""

import os
import time
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model

# Load environment variables
load_dotenv()

class TestResponse(BaseModel):
    summary: str
    key_points: List[str]
    difficulty_level: str
    estimated_reading_time: int

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"🔍 {title}")
    print(f"{'='*70}")

def print_test(test_name: str):
    """Print formatted test name"""
    print(f"\n🧪 {test_name}")
    print("-" * 50)

def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")

def format_response_details(response):
    """Format and display detailed response information"""
    print(f"📝 Generated Text: {response.generated_text[:200]}...")
    print(f"🤖 Model: {response.model}")
    print(f"⏱️  Process Time: {response.process_time:.3f} seconds")
    print(f"📊 Input Tokens: {response.input_token_count}")
    print(f"📊 Output Tokens: {response.output_token_count}")
    
    if hasattr(response, 'provider') and response.provider:
        print(f"🏭 Provider: {response.provider}")
    if hasattr(response, 'model_name') and response.model_name:
        print(f"🔧 Model Name: {response.model_name}")
    
    # Calculate total tokens and efficiency metrics directly from LLMFullResponse
    if response.input_token_count and response.output_token_count:
        total_tokens = response.input_token_count + response.output_token_count
        print(f"📊 Total Tokens: {total_tokens}")
        
        # Calculate tokens per second
        if response.process_time and response.process_time > 0:
            tokens_per_second = response.output_token_count / response.process_time
            print(f"⚡ Output Speed: {tokens_per_second:.1f} tokens/second")
        
        # Calculate input/output ratio
        token_ratio = response.output_token_count / response.input_token_count if response.input_token_count > 0 else 0
        print(f"📈 Output/Input Ratio: {token_ratio:.2f}x")

def test_basic_full_response():
    """Test basic full response functionality"""
    print_test("Basic Full Response Generation")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_error("OPENROUTER_API_KEY not found")
        return False
    
    try:
        llm = LLM.create(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4o",
            temperature=0.7
        )
        
        prompt = "Explain the concept of machine learning in exactly 100 words."
        
        response = llm.generate_response(
            prompt=prompt,
            max_tokens=150,
            full_response=True
        )
        
        if response and hasattr(response, 'generated_text'):
            print_success("Full response generated successfully")
            format_response_details(response)
            return True
        else:
            print_error("Invalid response object")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_different_models_full_response():
    """Test full response with different OpenRouter models"""
    print_test("Full Response with Different Models")
    
    models = [
        "openai/gpt-4o",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku",
        "meta-llama/llama-3.1-8b-instruct:free"
    ]
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_error("OPENROUTER_API_KEY not found")
        return False
    
    results = {}
    prompt = "What is artificial intelligence? Answer in 50 words."
    
    for model in models:
        print(f"\n🔄 Testing model: {model}")
        try:
            llm = LLM.create(
                provider=LLMProvider.OPENROUTER,
                model_name=model,
                temperature=0.7
            )
            
            start_time = time.time()
            response = llm.generate_response(
                prompt=prompt,
                max_tokens=80,
                full_response=True
            )
            end_time = time.time()
            
            if response and hasattr(response, 'generated_text'):
                results[model] = {
                    'success': True,
                    'tokens_in': response.input_token_count,
                    'tokens_out': response.output_token_count,
                    'time': response.process_time,
                    'wall_time': end_time - start_time,
                    'response_length': len(response.generated_text)
                }
                print_success(f"Model {model} succeeded")
                print(f"   📊 Tokens: {response.input_token_count} → {response.output_token_count}")
                print(f"   ⏱️  Time: {response.process_time:.3f}s (wall: {end_time - start_time:.3f}s)")
            else:
                results[model] = {'success': False, 'error': 'Invalid response'}
                print_error(f"Model {model} failed - invalid response")
                
        except Exception as e:
            results[model] = {'success': False, 'error': str(e)}
            print_error(f"Model {model} failed: {str(e)}")
    
    # Summary
    print(f"\n📈 Model Performance Summary:")
    successful_models = [m for m, r in results.items() if r.get('success')]
    print(f"   ✅ Successful: {len(successful_models)}/{len(models)} models")
    
    if successful_models:
        avg_input_tokens = sum(results[m]['tokens_in'] for m in successful_models if results[m]['tokens_in']) / len(successful_models)
        avg_output_tokens = sum(results[m]['tokens_out'] for m in successful_models if results[m]['tokens_out']) / len(successful_models)
        print(f"   📊 Avg Input Tokens: {avg_input_tokens:.1f}")
        print(f"   📊 Avg Output Tokens: {avg_output_tokens:.1f}")
    
    return len(successful_models) > 0

def test_json_mode_full_response():
    """Test full response with JSON mode"""
    print_test("JSON Mode Full Response")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_error("OPENROUTER_API_KEY not found")
        return False
    
    try:
        llm = LLM.create(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4o",
            temperature=0.7
        )
        
        prompt = """Generate a JSON object with the following structure:
        {
            "name": "person's name",
            "age": number,
            "profession": "job title",
            "skills": ["skill1", "skill2", "skill3"]
        }
        Create data for a fictional software developer."""
        
        response = llm.generate_response(
            prompt=prompt,
            max_tokens=200,
            json_mode=True,
            full_response=True
        )
        
        if response and hasattr(response, 'generated_text'):
            print_success("JSON mode full response generated")
            format_response_details(response)
            
            # Try to parse the JSON
            try:
                json_data = json.loads(response.generated_text)
                print(f"✅ Valid JSON generated:")
                print(f"   👤 Name: {json_data.get('name', 'N/A')}")
                print(f"   📅 Age: {json_data.get('age', 'N/A')}")
                print(f"   💼 Profession: {json_data.get('profession', 'N/A')}")
                print(f"   🛠️  Skills: {len(json_data.get('skills', []))} items")
                return True
            except json.JSONDecodeError as e:
                print_error(f"Invalid JSON generated: {str(e)}")
                print(f"Raw response: {response.generated_text}")
                return False
        else:
            print_error("No response generated")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_pydantic_full_response():
    """Test Pydantic JSON model with full response"""
    print_test("Pydantic Model Full Response")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_error("OPENROUTER_API_KEY not found")
        return False
    
    try:
        llm = LLM.create(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4o",
            temperature=0.7
        )
        
        prompt = "Create a comprehensive summary about renewable energy technologies"
        
        result = generate_pydantic_json_model(
            model_class=TestResponse,
            prompt=prompt,
            llm_instance=llm,
            max_tokens=300,
            full_response=True
        )
        
        if hasattr(result, 'model_object') and result.model_object:
            print_success("Pydantic model with full response generated")
            
            # Display full response details
            format_response_details(result)
            
            # Display parsed model data
            model_obj = result.model_object
            print(f"\n📋 Parsed Model Data:")
            print(f"   📝 Summary: {model_obj.summary[:100]}...")
            print(f"   🔑 Key Points: {len(model_obj.key_points)} items")
            print(f"   📊 Difficulty: {model_obj.difficulty_level}")
            print(f"   ⏰ Reading Time: {model_obj.estimated_reading_time} minutes")
            
            return True
        else:
            print_error(f"Failed to generate Pydantic model: {result}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_large_response_handling():
    """Test handling of large responses with token limits"""
    print_test("Large Response Token Handling")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_error("OPENROUTER_API_KEY not found")
        return False
    
    try:
        llm = LLM.create(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4o",
            temperature=0.7
        )
        
        prompt = """Write a detailed essay about the history and impact of the internet. 
        Cover the following topics:
        1. Early development and ARPANET
        2. World Wide Web creation
        3. Commercialization and dot-com boom
        4. Social media revolution
        5. Mobile internet and smartphones
        6. Current trends and future outlook
        
        Make it comprehensive and detailed."""
        
        response = llm.generate_response(
            prompt=prompt,
            max_tokens=1000,  # Large token limit
            full_response=True
        )
        
        if response and hasattr(response, 'generated_text'):
            print_success("Large response generated successfully")
            format_response_details(response)
            
            # Analyze response characteristics
            word_count = len(response.generated_text.split())
            char_count = len(response.generated_text)
            lines = response.generated_text.count('\n') + 1
            
            print(f"\n📊 Response Analysis:")
            print(f"   📝 Word Count: {word_count}")
            print(f"   🔤 Character Count: {char_count}")
            print(f"   📄 Line Count: {lines}")
            
            if response.output_token_count:
                efficiency = word_count / response.output_token_count if response.output_token_count > 0 else 0
                print(f"   ⚡ Words per Token: {efficiency:.2f}")
            
            return True
        else:
            print_error("Failed to generate large response")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_response_metadata():
    """Test various response metadata fields"""
    print_test("Response Metadata Validation")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_error("OPENROUTER_API_KEY not found")
        return False
    
    try:
        llm = LLM.create(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4o",
            temperature=0.3,  # Lower temperature for consistency
            verbose=True
        )
        
        prompt = "Explain quantum computing in simple terms."
        
        response = llm.generate_response(
            prompt=prompt,
            max_tokens=200,
            full_response=True
        )
        
        if response:
            print_success("Response generated for metadata validation")
            
            # Validate all expected fields
            required_fields = [
                'generated_text', 'model', 'process_time', 
                'input_token_count', 'output_token_count', 'llm_provider_response'
            ]
            
            print(f"\n🔍 Metadata Validation:")
            for field in required_fields:
                value = getattr(response, field, None)
                if value is not None:
                    print_success(f"{field}: {type(value).__name__} = {str(value)[:50]}...")
                else:
                    print_error(f"{field}: Missing")
            
            # Validate data types and ranges
            print(f"\n🧮 Data Validation:")
            
            if response.process_time and response.process_time > 0:
                print_success(f"Process time is positive: {response.process_time:.3f}s")
            else:
                print_error("Process time is invalid or missing")
            
            if response.input_token_count and response.input_token_count > 0:
                print_success(f"Input token count is positive: {response.input_token_count}")
            else:
                print_error("Input token count is invalid or missing")
            
            if response.output_token_count and response.output_token_count > 0:
                print_success(f"Output token count is positive: {response.output_token_count}")
            else:
                print_error("Output token count is invalid or missing")
            
            if response.generated_text and len(response.generated_text) > 0:
                print_success(f"Generated text length: {len(response.generated_text)} chars")
            else:
                print_error("Generated text is empty or missing")
            
            return True
        else:
            print_error("No response generated")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def main():
    """Run all OpenRouter full response tests"""
    print("🚀 OpenRouter LLMFullResponse Testing Suite")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_error("OPENROUTER_API_KEY environment variable not found")
        print_info("Please set your OpenRouter API key in the .env file")
        return
    
    print_success("OpenRouter API key found")
    
    # Run tests
    tests = [
        ("Basic Full Response", test_basic_full_response),
        ("Different Models", test_different_models_full_response),
        ("JSON Mode", test_json_mode_full_response),
        ("Pydantic Models", test_pydantic_full_response),
        ("Large Responses", test_large_response_handling),
        ("Metadata Validation", test_response_metadata),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print_header(test_name)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Final summary
    print_header("Test Summary")
    successful_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    print(f"📊 Results: {successful_tests}/{total_tests} tests passed")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")
    
    success_rate = (successful_tests / total_tests) * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print_success("OpenRouter LLMFullResponse integration is working well!")
    elif success_rate >= 50:
        print_info("OpenRouter integration has some issues that need attention")
    else:
        print_error("OpenRouter integration needs significant debugging")

if __name__ == "__main__":
    main()