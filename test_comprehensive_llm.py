#!/usr/bin/env python3
"""
Comprehensive test script for SimplerLLM functionality.
Tests ReliableLLM, generate_pydantic_json_model, async functions, token counting, and full responses.
"""

import asyncio
import os
import time
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import (
    generate_pydantic_json_model,
    generate_pydantic_json_model_async,
    generate_pydantic_json_model_reliable,
    generate_pydantic_json_model_reliable_async,
    calculate_text_generation_costs
)

# Load environment variables
load_dotenv()

# Test Data Models
class PersonInfo(BaseModel):
    name: str
    age: int
    city: str
    occupation: str

class TaskList(BaseModel):
    title: str
    tasks: List[str]
    priority: str
    deadline: Optional[str] = None

class SimpleResponse(BaseModel):
    response: str
    confidence: float

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_test(test_name: str):
    """Print a formatted test header"""
    print(f"\n🔹 {test_name}")
    print("-" * 40)

def print_result(success: bool, message: str):
    """Print a formatted test result"""
    status = "✅" if success else "❌"
    print(f"{status} {message}")

def check_api_keys():
    """Check if required API keys are available"""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    available_keys = {k: v for k, v in keys.items() if v}
    
    print_section("API Key Status")
    for key, value in keys.items():
        print_result(bool(value), f"{key}: {'Available' if value else 'Not found'}")
    
    return available_keys

def create_llm_instances(available_keys):
    """Create LLM instances based on available API keys"""
    instances = {}
    
    if "OPENAI_API_KEY" in available_keys:
        try:
            instances["openai"] = LLM.create(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o",
                temperature=0.7
            )
            print_result(True, "OpenAI LLM instance created")
        except Exception as e:
            print_result(False, f"OpenAI LLM creation failed: {str(e)}")
    
    if "OPENROUTER_API_KEY" in available_keys:
        try:
            instances["openrouter"] = LLM.create(
                provider=LLMProvider.OPENROUTER,
                model_name="openai/gpt-4o",
                temperature=0.7
            )
            print_result(True, "OpenRouter LLM instance created")
        except Exception as e:
            print_result(False, f"OpenRouter LLM creation failed: {str(e)}")
    
    if "ANTHROPIC_API_KEY" in available_keys:
        try:
            instances["anthropic"] = LLM.create(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                temperature=0.7
            )
            print_result(True, "Anthropic LLM instance created")
        except Exception as e:
            print_result(False, f"Anthropic LLM creation failed: {str(e)}")
    
    return instances

def test_basic_responses(instances):
    """Test basic response generation"""
    print_section("Basic Response Generation")
    
    for name, instance in instances.items():
        print_test(f"Basic Response - {name}")
        try:
            response = instance.generate_response(
                prompt="Generate a simple greeting in 10 words or less.",
                max_tokens=50
            )
            if response:
                print_result(True, f"Response: {response}")
            else:
                print_result(False, "No response generated")
        except Exception as e:
            print_result(False, f"Error: {str(e)}")

def test_full_responses(instances):
    """Test full response generation with token counts"""
    print_section("Full Response with Token Counts")
    
    for name, instance in instances.items():
        print_test(f"Full Response - {name}")
        try:
            response = instance.generate_response(
                prompt="Explain what artificial intelligence is in 50 words.",
                max_tokens=100,
                full_response=True
            )
            if response:
                print_result(True, f"Generated text: {response.generated_text[:100]}...")
                print(f"   📊 Input tokens: {response.input_token_count}")
                print(f"   📊 Output tokens: {response.output_token_count}")
                print(f"   ⏱️ Process time: {response.process_time:.2f}s")
                print(f"   🤖 Model: {response.model}")
            else:
                print_result(False, "No response generated")
        except Exception as e:
            print_result(False, f"Error: {str(e)}")

def test_json_mode(instances):
    """Test JSON mode functionality"""
    print_section("JSON Mode Testing")
    
    for name, instance in instances.items():
        print_test(f"JSON Mode - {name}")
        try:
            response = instance.generate_response(
                prompt="Generate a JSON object with fields: name, age, city. Use example data.",
                max_tokens=100,
                json_mode=True
            )
            if response:
                print_result(True, f"JSON Response: {response}")
            else:
                print_result(False, "No JSON response generated")
        except Exception as e:
            print_result(False, f"Error: {str(e)}")

async def test_async_responses(instances):
    """Test async response generation"""
    print_section("Async Response Generation")
    
    for name, instance in instances.items():
        print_test(f"Async Response - {name}")
        try:
            start_time = time.time()
            response = await instance.generate_response_async(
                prompt="What is the capital of France?",
                max_tokens=50
            )
            end_time = time.time()
            
            if response:
                print_result(True, f"Response: {response}")
                print(f"   ⏱️ Async time: {end_time - start_time:.2f}s")
            else:
                print_result(False, "No async response generated")
        except Exception as e:
            print_result(False, f"Error: {str(e)}")

def test_pydantic_json_models(instances):
    """Test generate_pydantic_json_model functionality"""
    print_section("Pydantic JSON Model Generation")
    
    for name, instance in instances.items():
        print_test(f"Pydantic JSON - {name}")
        try:
            result = generate_pydantic_json_model(
                model_class=PersonInfo,
                prompt="Generate information for a fictional software engineer",
                llm_instance=instance,
                max_tokens=200
            )
            
            if isinstance(result, PersonInfo):
                print_result(True, f"Generated person: {result.name}, {result.age}, {result.city}, {result.occupation}")
            else:
                print_result(False, f"Failed: {result}")
        except Exception as e:
            print_result(False, f"Error: {str(e)}")

def test_pydantic_json_full_response(instances):
    """Test pydantic JSON model with full response"""
    print_section("Pydantic JSON with Full Response")
    
    for name, instance in instances.items():
        print_test(f"Pydantic JSON Full Response - {name}")
        try:
            result = generate_pydantic_json_model(
                model_class=TaskList,
                prompt="Generate a task list for planning a birthday party",
                llm_instance=instance,
                max_tokens=300,
                full_response=True
            )
            
            if hasattr(result, 'model_object') and result.model_object:
                task_list = result.model_object
                print_result(True, f"Task list: {task_list.title}")
                print(f"   📝 Tasks: {len(task_list.tasks)} items")
                print(f"   📊 Input tokens: {result.input_token_count}")
                print(f"   📊 Output tokens: {result.output_token_count}")
                print(f"   ⏱️ Process time: {result.process_time:.2f}s")
            else:
                print_result(False, f"Failed: {result}")
        except Exception as e:
            print_result(False, f"Error: {str(e)}")

async def test_pydantic_json_async(instances):
    """Test async pydantic JSON model generation"""
    print_section("Async Pydantic JSON Generation")
    
    for name, instance in instances.items():
        print_test(f"Async Pydantic JSON - {name}")
        try:
            result = await generate_pydantic_json_model_async(
                model_class=SimpleResponse,
                prompt="Provide a confident response about the importance of testing",
                llm_instance=instance,
                max_tokens=150
            )
            
            if isinstance(result, SimpleResponse):
                print_result(True, f"Response: {result.response[:100]}...")
                print(f"   🎯 Confidence: {result.confidence}")
            else:
                print_result(False, f"Failed: {result}")
        except Exception as e:
            print_result(False, f"Error: {str(e)}")

def test_reliable_llm(instances):
    """Test ReliableLLM functionality"""
    print_section("ReliableLLM Testing")
    
    if len(instances) < 2:
        print_result(False, "Need at least 2 LLM instances for ReliableLLM testing")
        return
    
    instance_names = list(instances.keys())
    primary_name = instance_names[0]
    secondary_name = instance_names[1]
    
    print_test(f"ReliableLLM - Primary: {primary_name}, Secondary: {secondary_name}")
    
    try:
        reliable_llm = ReliableLLM(
            primary_llm=instances[primary_name],
            secondary_llm=instances[secondary_name],
            verbose=True
        )
        
        # Test basic reliable response
        response = reliable_llm.generate_response(
            prompt="What is machine learning?",
            max_tokens=100,
            return_provider=True
        )
        
        if isinstance(response, tuple) and len(response) == 3:
            text, provider, model_name = response
            print_result(True, f"Response from {provider.name}: {text[:100]}...")
            print(f"   🤖 Model: {model_name}")
        else:
            print_result(False, f"Unexpected response format: {response}")
            
    except Exception as e:
        print_result(False, f"Error: {str(e)}")

def test_reliable_llm_pydantic(instances):
    """Test ReliableLLM with pydantic JSON models"""
    print_section("ReliableLLM with Pydantic JSON")
    
    if len(instances) < 2:
        print_result(False, "Need at least 2 LLM instances for ReliableLLM testing")
        return
    
    instance_names = list(instances.keys())
    primary_name = instance_names[0]
    secondary_name = instance_names[1]
    
    print_test(f"ReliableLLM Pydantic - Primary: {primary_name}, Secondary: {secondary_name}")
    
    try:
        reliable_llm = ReliableLLM(
            primary_llm=instances[primary_name],
            secondary_llm=instances[secondary_name],
            verbose=False  # Reduce verbosity for cleaner output
        )
        
        result = generate_pydantic_json_model_reliable(
            model_class=PersonInfo,
            prompt="Generate information for a fictional teacher",
            reliable_llm=reliable_llm,
            max_tokens=200
        )
        
        if isinstance(result, tuple) and len(result) == 3:
            person, provider, model_name = result
            if isinstance(person, PersonInfo):
                print_result(True, f"Person from {provider.name}: {person.name}, {person.occupation}")
                print(f"   🤖 Model: {model_name}")
            else:
                print_result(False, f"Invalid person object: {person}")
        else:
            print_result(False, f"Failed: {result}")
            
    except Exception as e:
        print_result(False, f"Error: {str(e)}")

async def test_reliable_llm_async(instances):
    """Test ReliableLLM async functionality"""
    print_section("ReliableLLM Async Testing")
    
    if len(instances) < 2:
        print_result(False, "Need at least 2 LLM instances for ReliableLLM testing")
        return
    
    instance_names = list(instances.keys())
    primary_name = instance_names[0]
    secondary_name = instance_names[1]
    
    print_test(f"ReliableLLM Async - Primary: {primary_name}, Secondary: {secondary_name}")
    
    try:
        reliable_llm = ReliableLLM(
            primary_llm=instances[primary_name],
            secondary_llm=instances[secondary_name],
            verbose=False
        )
        
        start_time = time.time()
        response = await reliable_llm.generate_response_async(
            prompt="Explain quantum computing in simple terms.",
            max_tokens=150,
            return_provider=True
        )
        end_time = time.time()
        
        if isinstance(response, tuple) and len(response) == 3:
            text, provider, model_name = response
            print_result(True, f"Async response from {provider.name}: {text[:100]}...")
            print(f"   🤖 Model: {model_name}")
            print(f"   ⏱️ Async time: {end_time - start_time:.2f}s")
        else:
            print_result(False, f"Unexpected response format: {response}")
            
    except Exception as e:
        print_result(False, f"Error: {str(e)}")

def test_cost_calculation():
    """Test cost calculation functionality"""
    print_section("Cost Calculation Testing")
    
    print_test("Cost Calculation")
    
    try:
        input_text = "What is artificial intelligence and how does it work?"
        response_text = "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."
        
        # Example pricing (OpenAI GPT-3.5-turbo pricing)
        cost_info = calculate_text_generation_costs(
            input=input_text,
            response=response_text,
            cost_per_million_input_tokens=0.50,  # $0.50 per 1M input tokens
            cost_per_million_output_tokens=1.50,  # $1.50 per 1M output tokens
            approximate=True
        )
        
        print_result(True, "Cost calculation completed")
        print(f"   📊 Input tokens: {cost_info['input_tokens']}")
        print(f"   📊 Output tokens: {cost_info['output_tokens']}")
        print(f"   💰 Input cost: ${cost_info['input_cost']:.6f}")
        print(f"   💰 Output cost: ${cost_info['output_cost']:.6f}")
        print(f"   💰 Total cost: ${cost_info['total_cost']:.6f}")
        
    except Exception as e:
        print_result(False, f"Error: {str(e)}")

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Comprehensive SimplerLLM Testing Suite")
    print("=" * 60)
    
    # Check API keys
    available_keys = check_api_keys()
    
    if not available_keys:
        print_result(False, "No API keys found. Please set at least one API key.")
        return
    
    # Create LLM instances
    print_section("LLM Instance Creation")
    instances = create_llm_instances(available_keys)
    
    if not instances:
        print_result(False, "No LLM instances could be created.")
        return
    
    # Run tests
    test_basic_responses(instances)
    test_full_responses(instances)
    test_json_mode(instances)
    await test_async_responses(instances)
    test_pydantic_json_models(instances)
    test_pydantic_json_full_response(instances)
    await test_pydantic_json_async(instances)
    test_reliable_llm(instances)
    test_reliable_llm_pydantic(instances)
    await test_reliable_llm_async(instances)
    test_cost_calculation()
    
    # Summary
    print_section("Testing Complete")
    print("✅ All tests have been executed!")
    print("\nNote: Some tests may fail due to:")
    print("- Missing API keys")
    print("- API rate limits or quota exceeded")
    print("- Network connectivity issues")
    print("- Model availability")
    print("- Temporary service outages")

if __name__ == "__main__":
    asyncio.run(run_all_tests())