#!/usr/bin/env python3
"""
Comprehensive test script for SimplerLLM Embeddings functionality.
Tests all embedding features including sync/async generation, full responses, and integration capabilities.
"""

import asyncio
import os
import time
import numpy as np
from dotenv import load_dotenv
from typing import List

from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

# Load environment variables
load_dotenv()

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"🧪 {title}")
    print(f"{'='*70}")

def print_test(test_name: str):
    """Print formatted test name"""
    print(f"\n🔹 {test_name}")
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

def format_embedding_info(embedding, label="Embedding"):
    """Format and display embedding information"""
    if isinstance(embedding, list):
        print(f"📊 {label} Vector:")
        print(f"   📏 Dimensions: {len(embedding)}")
        print(f"   📈 Range: [{min(embedding):.6f}, {max(embedding):.6f}]")
        print(f"   🔢 Sample values: {embedding[:5]}")
        print(f"   📐 L2 Norm: {np.linalg.norm(embedding):.6f}")
    else:
        print_error(f"{label} is not a valid list: {type(embedding)}")

def format_full_response_info(response, label="Full Response"):
    """Format and display full response information"""
    print(f"🔍 {label} Details:")
    print(f"   🤖 Model: {response.model}")
    print(f"   ⏱️  Process Time: {response.process_time:.3f} seconds")
    
    if hasattr(response, 'generated_embedding') and response.generated_embedding:
        embedding = response.generated_embedding
        if isinstance(embedding, list):
            print(f"   📏 Dimensions: {len(embedding)}")
            print(f"   📈 Range: [{min(embedding):.6f}, {max(embedding):.6f}]")
            print(f"   📐 L2 Norm: {np.linalg.norm(embedding):.6f}")
        else:
            print(f"   📊 Embedding type: {type(embedding)}")

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

def test_embeddings_instance_creation():
    """Test creating embeddings instances"""
    print_test("Embeddings Instance Creation")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    try:
        # Test basic instance creation
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.OPENAI,
            model_name="text-embedding-3-small"
        )
        
        if embeddings_llm:
            print_success("EmbeddingsLLM instance created successfully")
            print(f"   🏭 Provider: {embeddings_llm.provider}")
            print(f"   🤖 Model: {embeddings_llm.model_name}")
            print(f"   🔑 API Key: {'Set' if embeddings_llm.api_key else 'Not set'}")
            return True
        else:
            print_error("Failed to create EmbeddingsLLM instance")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_basic_embedding_generation():
    """Test basic embedding generation"""
    print_test("Basic Embedding Generation")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.OPENAI,
            model_name="text-embedding-3-small"
        )
        
        test_text = "Machine learning is a subset of artificial intelligence."
        
        embedding = embeddings_llm.generate_embeddings(test_text)
        
        if embedding and isinstance(embedding, list):
            print_success("Basic embedding generated successfully")
            format_embedding_info(embedding, "Generated")
            return True
        else:
            print_error(f"Invalid embedding generated: {type(embedding)}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_full_response_embeddings():
    """Test embeddings with full response"""
    print_test("Full Response Embeddings")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.OPENAI,
            model_name="text-embedding-3-small"
        )
        
        test_text = "Natural language processing enables computers to understand human language."
        
        response = embeddings_llm.generate_embeddings(
            user_input=test_text,
            full_response=True
        )
        
        if response and hasattr(response, 'generated_embedding'):
            print_success("Full response embedding generated successfully")
            format_full_response_info(response, "Embeddings")
            return True
        else:
            print_error(f"Invalid full response: {type(response)}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

async def test_async_embedding_generation():
    """Test async embedding generation"""
    print_test("Async Embedding Generation")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.OPENAI,
            model_name="text-embedding-3-small"
        )
        
        test_text = "Deep learning uses neural networks with multiple layers."
        
        start_time = time.time()
        embedding = await embeddings_llm.generate_embeddings_async(test_text)
        end_time = time.time()
        
        if embedding and isinstance(embedding, list):
            print_success("Async embedding generated successfully")
            format_embedding_info(embedding, "Async Generated")
            print(f"   ⏱️  Async time: {end_time - start_time:.3f} seconds")
            return True
        else:
            print_error(f"Invalid async embedding: {type(embedding)}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

async def test_async_full_response():
    """Test async embedding with full response"""
    print_test("Async Full Response Embeddings")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.OPENAI,
            model_name="text-embedding-3-small"
        )
        
        test_text = "Computer vision enables machines to interpret visual information."
        
        start_time = time.time()
        response = await embeddings_llm.generate_embeddings_async(
            user_input=test_text,
            full_response=True
        )
        end_time = time.time()
        
        if response and hasattr(response, 'generated_embedding'):
            print_success("Async full response embedding generated successfully")
            format_full_response_info(response, "Async Embeddings")
            print(f"   ⏱️  Async wall time: {end_time - start_time:.3f} seconds")
            return True
        else:
            print_error(f"Invalid async full response: {type(response)}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_multiple_texts_embedding():
    """Test embedding generation for multiple texts"""
    print_test("Multiple Texts Embedding")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.OPENAI,
            model_name="text-embedding-3-small"
        )
        
        test_texts = [
            "Artificial intelligence is transforming technology.",
            "Machine learning algorithms learn from data patterns.",
            "Neural networks are inspired by biological neurons."
        ]
        
        embeddings = embeddings_llm.generate_embeddings(test_texts)
        
        if embeddings and isinstance(embeddings, list):
            print_success("Multiple text embeddings generated successfully")
            print(f"   📊 Number of embeddings: {len(embeddings)}")
            
            for i, embedding in enumerate(embeddings):
                if isinstance(embedding, list):
                    print(f"   📏 Text {i+1} dimensions: {len(embedding)}")
                else:
                    print_error(f"Invalid embedding {i+1}: {type(embedding)}")
            
            return True
        else:
            print_error(f"Invalid multiple embeddings: {type(embeddings)}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_different_models():
    """Test different OpenAI embedding models"""
    print_test("Different Embedding Models")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ]
    
    test_text = "This is a test sentence for embedding generation."
    results = {}
    
    for model in models:
        print(f"\n🔄 Testing model: {model}")
        try:
            embeddings_llm = EmbeddingsLLM.create(
                provider=EmbeddingsProvider.OPENAI,
                model_name=model
            )
            
            start_time = time.time()
            response = embeddings_llm.generate_embeddings(
                user_input=test_text,
                full_response=True
            )
            end_time = time.time()
            
            if response and hasattr(response, 'generated_embedding'):
                embedding = response.generated_embedding
                results[model] = {
                    'success': True,
                    'dimensions': len(embedding) if isinstance(embedding, list) else 0,
                    'time': response.process_time,
                    'wall_time': end_time - start_time
                }
                print_success(f"Model {model} succeeded")
                print(f"   📏 Dimensions: {results[model]['dimensions']}")
                print(f"   ⏱️  Time: {results[model]['time']:.3f}s")
            else:
                results[model] = {'success': False, 'error': 'Invalid response'}
                print_error(f"Model {model} failed - invalid response")
                
        except Exception as e:
            results[model] = {'success': False, 'error': str(e)}
            print_error(f"Model {model} failed: {str(e)}")
    
    # Summary
    successful_models = [m for m, r in results.items() if r.get('success')]
    print(f"\n📈 Model Performance Summary:")
    print(f"   ✅ Successful: {len(successful_models)}/{len(models)} models")
    
    if successful_models:
        for model in successful_models:
            print(f"   🤖 {model}: {results[model]['dimensions']}D, {results[model]['time']:.3f}s")
    
    return len(successful_models) > 0

def test_semantic_similarity():
    """Test semantic similarity using embeddings"""
    print_test("Semantic Similarity Testing")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.OPENAI,
            model_name="text-embedding-3-small"
        )
        
        # Test sentences with varying similarity
        sentences = {
            "ai_1": "Machine learning is a branch of artificial intelligence.",
            "ai_2": "AI and machine learning are closely related technologies.",
            "weather": "Today is a sunny day with clear skies.",
            "food": "Pizza is a popular Italian dish with various toppings."
        }
        
        # Generate embeddings for all sentences
        embeddings = {}
        for key, sentence in sentences.items():
            embedding = embeddings_llm.generate_embeddings(sentence)
            if embedding and isinstance(embedding, list):
                embeddings[key] = embedding
            else:
                print_error(f"Failed to generate embedding for {key}")
                return False
        
        print_success("Generated embeddings for similarity testing")
        
        # Calculate similarities
        similarities = {}
        pairs = [
            ("ai_1", "ai_2", "AI-related sentences"),
            ("ai_1", "weather", "AI vs Weather"),
            ("ai_1", "food", "AI vs Food"),
            ("weather", "food", "Weather vs Food")
        ]
        
        print(f"\n🔍 Semantic Similarity Analysis:")
        for key1, key2, description in pairs:
            similarity = calculate_cosine_similarity(embeddings[key1], embeddings[key2])
            similarities[f"{key1}_{key2}"] = similarity
            print(f"   📊 {description}: {similarity:.4f}")
        
        # Validate expected relationships
        ai_similarity = similarities["ai_1_ai_2"]
        cross_domain_avg = (similarities["ai_1_weather"] + similarities["ai_1_food"]) / 2
        
        if ai_similarity > cross_domain_avg:
            print_success(f"Semantic relationships validated (AI: {ai_similarity:.4f} > Cross-domain: {cross_domain_avg:.4f})")
            return True
        else:
            print_error(f"Unexpected similarity relationships")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print_test("Edge Cases and Error Handling")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found")
        return False
    
    success_count = 0
    total_tests = 0
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.OPENAI,
            model_name="text-embedding-3-small"
        )
        
        # Test 1: Empty string
        total_tests += 1
        print("\n🧪 Testing empty string...")
        try:
            embedding = embeddings_llm.generate_embeddings("")
            if embedding:
                print_success("Empty string handled successfully")
                success_count += 1
            else:
                print_info("Empty string returned None (expected)")
                success_count += 1
        except Exception as e:
            print_error(f"Empty string test failed: {str(e)}")
        
        # Test 2: Very long text
        total_tests += 1
        print("\n🧪 Testing very long text...")
        try:
            long_text = "This is a test sentence. " * 1000  # Very long text
            embedding = embeddings_llm.generate_embeddings(long_text)
            if embedding and isinstance(embedding, list):
                print_success("Long text handled successfully")
                format_embedding_info(embedding, "Long Text")
                success_count += 1
            else:
                print_error("Long text test failed")
        except Exception as e:
            print_info(f"Long text appropriately rejected: {str(e)}")
            success_count += 1  # This might be expected behavior
        
        # Test 3: Special characters
        total_tests += 1
        print("\n🧪 Testing special characters...")
        try:
            special_text = "Hello! @#$%^&*()_+ 你好 🚀 émojis and ünïcödë"
            embedding = embeddings_llm.generate_embeddings(special_text)
            if embedding and isinstance(embedding, list):
                print_success("Special characters handled successfully")
                format_embedding_info(embedding, "Special Characters")
                success_count += 1
            else:
                print_error("Special characters test failed")
        except Exception as e:
            print_error(f"Special characters test failed: {str(e)}")
        
        # Test 4: Model name override
        total_tests += 1
        print("\n🧪 Testing model name override...")
        try:
            embedding = embeddings_llm.generate_embeddings(
                user_input="Test model override",
                model_name="text-embedding-ada-002"  # Different model
            )
            if embedding and isinstance(embedding, list):
                print_success("Model name override successful")
                success_count += 1
            else:
                print_error("Model override test failed")
        except Exception as e:
            print_error(f"Model override test failed: {str(e)}")
        
        print(f"\n📊 Edge Cases Summary: {success_count}/{total_tests} tests handled appropriately")
        return success_count >= total_tests * 0.75  # 75% success rate acceptable
        
    except Exception as e:
        print_error(f"Edge cases setup failed: {str(e)}")
        return False

async def run_all_tests():
    """Run all embedding tests"""
    print("🚀 Starting Comprehensive Embeddings Testing Suite")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY environment variable not found")
        print_info("Please set your OpenAI API key in the .env file")
        return
    
    print_success("OpenAI API key found")
    
    # Run tests
    tests = [
        ("Instance Creation", test_embeddings_instance_creation),
        ("Basic Generation", test_basic_embedding_generation),
        ("Full Response", test_full_response_embeddings),
        ("Async Generation", test_async_embedding_generation),
        ("Async Full Response", test_async_full_response),
        ("Multiple Texts", test_multiple_texts_embedding),
        ("Different Models", test_different_models),
        ("Semantic Similarity", test_semantic_similarity),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print_header(test_name)
        try:
            if asyncio.iscoroutinefunction(test_func):
                results[test_name] = await test_func()
            else:
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
        print_success("Embeddings system is working excellently!")
    elif success_rate >= 60:
        print_info("Embeddings system is working with some issues")
    else:
        print_error("Embeddings system needs significant debugging")

if __name__ == "__main__":
    asyncio.run(run_all_tests())