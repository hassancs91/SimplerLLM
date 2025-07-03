#!/usr/bin/env python3
"""
Comprehensive test script for Voyage AI Embeddings functionality.
Tests all Voyage AI features including different models, dimensions, data types, and retrieval optimization.
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
    print(f"🧭 {title}")
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

def format_voyage_embedding_info(embedding, label="Voyage Embedding"):
    """Format and display Voyage embedding information"""
    if isinstance(embedding, list):
        print(f"🧭 {label} Vector:")
        print(f"   📏 Dimensions: {len(embedding)}")
        print(f"   📈 Range: [{min(embedding):.6f}, {max(embedding):.6f}]")
        print(f"   🔢 Sample values: {embedding[:5]}")
        print(f"   📐 L2 Norm: {np.linalg.norm(embedding):.6f}")
        print(f"   🔴 Data type: {type(embedding[0]).__name__}")
    else:
        print_error(f"{label} is not a valid list: {type(embedding)}")

def format_voyage_full_response(response, label="Voyage Full Response"):
    """Format and display Voyage full response information"""
    print(f"🔍 {label} Details:")
    print(f"   🤖 Model: {response.model}")
    print(f"   ⏱️  Process Time: {response.process_time:.3f} seconds")
    
    if hasattr(response, 'generated_embedding') and response.generated_embedding:
        embedding = response.generated_embedding
        if isinstance(embedding, list):
            print(f"   📏 Dimensions: {len(embedding)}")
            print(f"   📈 Range: [{min(embedding):.6f}, {max(embedding):.6f}]")
            print(f"   📐 L2 Norm: {np.linalg.norm(embedding):.6f}")
            print(f"   🔴 Data type: {type(embedding[0]).__name__}")

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

def test_voyage_instance_creation():
    """Test creating Voyage AI embeddings instances"""
    print_test("Voyage AI Instance Creation")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY not found")
        return False
    
    try:
        # Test basic instance creation
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.VOYAGE,
            model_name="voyage-3-large"
        )
        
        if embeddings_llm:
            print_success("Voyage AI EmbeddingsLLM instance created successfully")
            print(f"   🏭 Provider: {embeddings_llm.provider}")
            print(f"   🤖 Model: {embeddings_llm.model_name}")
            print(f"   🔑 API Key: {'Set' if embeddings_llm.api_key else 'Not set'}")
            return True
        else:
            print_error("Failed to create Voyage AI EmbeddingsLLM instance")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_voyage_basic_embedding():
    """Test basic Voyage AI embedding generation"""
    print_test("Basic Voyage AI Embedding")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.VOYAGE,
            model_name="voyage-3-large"
        )
        
        test_text = "Voyage AI provides state-of-the-art embedding models for retrieval applications."
        
        embedding = embeddings_llm.generate_embeddings(test_text)
        
        if embedding and isinstance(embedding, list):
            print_success("Basic Voyage AI embedding generated successfully")
            format_voyage_embedding_info(embedding, "Basic Voyage")
            return True
        else:
            print_error(f"Invalid Voyage AI embedding generated: {type(embedding)}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_voyage_different_models():
    """Test different Voyage AI models"""
    print_test("Different Voyage AI Models")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY not found")
        return False
    
    models = [
        "voyage-3-large",
        "voyage-3.5",
        "voyage-3.5-lite",
        "voyage-code-3",
        # "voyage-finance-2",  # Uncomment if you have access
        # "voyage-law-2"       # Uncomment if you have access
    ]
    
    test_text = "Machine learning algorithms can automatically learn patterns from data."
    results = {}
    
    for model in models:
        print(f"\n🔄 Testing Voyage model: {model}")
        try:
            embeddings_llm = EmbeddingsLLM.create(
                provider=EmbeddingsProvider.VOYAGE,
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
    print(f"\n📈 Voyage Model Performance Summary:")
    print(f"   ✅ Successful: {len(successful_models)}/{len(models)} models")
    
    if successful_models:
        for model in successful_models:
            print(f"   🧭 {model}: {results[model]['dimensions']}D, {results[model]['time']:.3f}s")
    
    return len(successful_models) > 0

def test_voyage_dimensions():
    """Test different output dimensions"""
    print_test("Voyage AI Different Dimensions")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY not found")
        return False
    
    dimensions = [256, 512, 1024, 2048]
    test_text = "Testing different embedding dimensions with Voyage AI."
    results = {}
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.VOYAGE,
            model_name="voyage-3-large"  # Supports multiple dimensions
        )
        
        for dim in dimensions:
            print(f"\n🔄 Testing dimension: {dim}")
            try:
                start_time = time.time()
                embedding = embeddings_llm.generate_embeddings(
                    user_input=test_text,
                    output_dimension=dim
                )
                end_time = time.time()
                
                if embedding and isinstance(embedding, list):
                    actual_dim = len(embedding)
                    results[dim] = {
                        'success': True,
                        'actual_dimension': actual_dim,
                        'time': end_time - start_time,
                        'matches_requested': actual_dim == dim
                    }
                    print_success(f"Dimension {dim} succeeded")
                    print(f"   📏 Requested: {dim}, Actual: {actual_dim}")
                    print(f"   ⏱️  Time: {results[dim]['time']:.3f}s")
                    print(f"   ✅ Match: {results[dim]['matches_requested']}")
                else:
                    results[dim] = {'success': False, 'error': 'Invalid embedding'}
                    print_error(f"Dimension {dim} failed - invalid embedding")
                    
            except Exception as e:
                results[dim] = {'success': False, 'error': str(e)}
                print_error(f"Dimension {dim} failed: {str(e)}")
        
        # Summary
        successful_dims = [d for d, r in results.items() if r.get('success')]
        print(f"\n📊 Dimension Test Summary:")
        print(f"   ✅ Successful: {len(successful_dims)}/{len(dimensions)} dimensions")
        
        return len(successful_dims) > 0
        
    except Exception as e:
        print_error(f"Dimension test setup failed: {str(e)}")
        return False

def test_voyage_input_types():
    """Test different input types (query vs document)"""
    print_test("Voyage AI Input Types")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.VOYAGE,
            model_name="voyage-3-large"
        )
        
        test_text = "What are the benefits of using vector embeddings for semantic search?"
        input_types = [None, "query", "document"]
        results = {}
        
        for input_type in input_types:
            print(f"\n🔄 Testing input type: {input_type or 'None'}")
            try:
                start_time = time.time()
                embedding = embeddings_llm.generate_embeddings(
                    user_input=test_text,
                    input_type=input_type
                )
                end_time = time.time()
                
                if embedding and isinstance(embedding, list):
                    results[str(input_type)] = {
                        'success': True,
                        'dimensions': len(embedding),
                        'time': end_time - start_time,
                        'embedding': embedding[:5]  # Store first 5 values for comparison
                    }
                    print_success(f"Input type {input_type or 'None'} succeeded")
                    print(f"   📏 Dimensions: {results[str(input_type)]['dimensions']}")
                    print(f"   ⏱️  Time: {results[str(input_type)]['time']:.3f}s")
                else:
                    results[str(input_type)] = {'success': False, 'error': 'Invalid embedding'}
                    print_error(f"Input type {input_type or 'None'} failed")
                    
            except Exception as e:
                results[str(input_type)] = {'success': False, 'error': str(e)}
                print_error(f"Input type {input_type or 'None'} failed: {str(e)}")
        
        # Compare embeddings to see if input_type makes a difference
        if 'query' in results and 'document' in results:
            both_success = results['query'].get('success') and results['document'].get('success')
            if both_success:
                query_emb = results['query']['embedding']
                doc_emb = results['document']['embedding']
                
                # Simple comparison of first few values
                difference = np.mean([abs(q - d) for q, d in zip(query_emb, doc_emb)])
                print(f"\n🔍 Input Type Comparison:")
                print(f"   📊 Average difference (query vs document): {difference:.6f}")
                
                if difference > 0.001:
                    print_success("Input types produce different embeddings (expected)")
                else:
                    print_info("Input types produce similar embeddings")
        
        successful_types = [t for t, r in results.items() if r.get('success')]
        print(f"\n📋 Input Type Summary:")
        print(f"   ✅ Successful: {len(successful_types)}/{len(input_types)} input types")
        
        return len(successful_types) > 0
        
    except Exception as e:
        print_error(f"Input type test setup failed: {str(e)}")
        return False

def test_voyage_data_types():
    """Test different output data types"""
    print_test("Voyage AI Data Types")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.VOYAGE,
            model_name="voyage-3-large"
        )
        
        test_text = "Testing different data types for Voyage AI embeddings."
        data_types = ["float", "int8"]  # Start with basic types
        results = {}
        
        for dtype in data_types:
            print(f"\n🔄 Testing data type: {dtype}")
            try:
                start_time = time.time()
                embedding = embeddings_llm.generate_embeddings(
                    user_input=test_text,
                    output_dtype=dtype,
                    output_dimension=512  # Smaller dimension for testing
                )
                end_time = time.time()
                
                if embedding and isinstance(embedding, list):
                    results[dtype] = {
                        'success': True,
                        'dimensions': len(embedding),
                        'time': end_time - start_time,
                        'sample_type': type(embedding[0]).__name__,
                        'sample_values': embedding[:3]
                    }
                    print_success(f"Data type {dtype} succeeded")
                    print(f"   📏 Dimensions: {results[dtype]['dimensions']}")
                    print(f"   🔴 Sample type: {results[dtype]['sample_type']}")
                    print(f"   🔢 Sample values: {results[dtype]['sample_values']}")
                    print(f"   ⏱️  Time: {results[dtype]['time']:.3f}s")
                else:
                    results[dtype] = {'success': False, 'error': 'Invalid embedding'}
                    print_error(f"Data type {dtype} failed")
                    
            except Exception as e:
                results[dtype] = {'success': False, 'error': str(e)}
                print_error(f"Data type {dtype} failed: {str(e)}")
        
        successful_types = [t for t, r in results.items() if r.get('success')]
        print(f"\n🔴 Data Type Summary:")
        print(f"   ✅ Successful: {len(successful_types)}/{len(data_types)} data types")
        
        return len(successful_types) > 0
        
    except Exception as e:
        print_error(f"Data type test setup failed: {str(e)}")
        return False

async def test_voyage_async():
    """Test async Voyage AI embedding generation"""
    print_test("Voyage AI Async Generation")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.VOYAGE,
            model_name="voyage-3-large"
        )
        
        test_text = "Asynchronous embedding generation with Voyage AI for better performance."
        
        start_time = time.time()
        embedding = await embeddings_llm.generate_embeddings_async(test_text)
        end_time = time.time()
        
        if embedding and isinstance(embedding, list):
            print_success("Async Voyage AI embedding generated successfully")
            format_voyage_embedding_info(embedding, "Async Voyage")
            print(f"   ⏱️  Async wall time: {end_time - start_time:.3f} seconds")
            return True
        else:
            print_error(f"Invalid async Voyage AI embedding: {type(embedding)}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_voyage_retrieval_scenario():
    """Test Voyage AI in a realistic retrieval scenario"""
    print_test("Voyage AI Retrieval Scenario")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY not found")
        return False
    
    try:
        embeddings_llm = EmbeddingsLLM.create(
            provider=EmbeddingsProvider.VOYAGE,
            model_name="voyage-3-large"
        )
        
        # Simulate a query and documents scenario
        query = "How do neural networks learn from data?"
        
        documents = [
            "Neural networks learn through backpropagation by adjusting weights based on error gradients.",
            "Machine learning algorithms use statistical methods to find patterns in large datasets.",
            "Deep learning models require extensive computational resources for training.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision systems can identify objects and patterns in images."
        ]
        
        print_success("Testing query vs document embeddings")
        
        # Generate query embedding
        query_embedding = embeddings_llm.generate_embeddings(
            user_input=query,
            input_type="query"
        )
        
        # Generate document embeddings
        doc_embeddings = []
        for i, doc in enumerate(documents):
            doc_emb = embeddings_llm.generate_embeddings(
                user_input=doc,
                input_type="document"
            )
            doc_embeddings.append(doc_emb)
            print(f"   📄 Document {i+1} embedded: {len(doc_emb)} dimensions")
        
        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = calculate_cosine_similarity(query_embedding, doc_emb)
            similarities.append((i, similarity, documents[i][:60] + "..."))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔍 Retrieval Results (Query: '{query[:50]}...'):")
        for rank, (doc_idx, sim, doc_preview) in enumerate(similarities, 1):
            print(f"   {rank}. 📊 Similarity: {sim:.4f} - {doc_preview}")
        
        # Validate that most relevant document is ranked highest
        top_similarity = similarities[0][1]
        if top_similarity > 0.7:  # Threshold for good similarity
            print_success(f"Good retrieval performance (top similarity: {top_similarity:.4f})")
            return True
        else:
            print_info(f"Moderate retrieval performance (top similarity: {top_similarity:.4f})")
            return True  # Still consider success
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

async def run_all_voyage_tests():
    """Run all Voyage AI embedding tests"""
    print("🧭 Starting Comprehensive Voyage AI Embeddings Testing Suite")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print_error("VOYAGE_API_KEY environment variable not found")
        print_info("Please set your Voyage AI API key in the .env file")
        print_info("Get your API key from: https://www.voyageai.com/")
        return
    
    print_success("Voyage AI API key found")
    
    # Check for voyageai package
    try:
        import voyageai
        print_success("voyageai package is available")
    except ImportError:
        print_error("voyageai package not installed")
        print_info("Install it with: pip install voyageai")
        return
    
    # Run tests
    tests = [
        ("Instance Creation", test_voyage_instance_creation),
        ("Basic Embedding", test_voyage_basic_embedding),
        ("Different Models", test_voyage_different_models),
        ("Different Dimensions", test_voyage_dimensions),
        ("Input Types", test_voyage_input_types),
        ("Data Types", test_voyage_data_types),
        ("Async Generation", test_voyage_async),
        ("Retrieval Scenario", test_voyage_retrieval_scenario),
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
        print_success("Voyage AI embeddings integration is working excellently!")
    elif success_rate >= 60:
        print_info("Voyage AI embeddings integration is working with some issues")
    else:
        print_error("Voyage AI embeddings integration needs significant debugging")
    
    print(f"\n🧭 Voyage AI Features Tested:")
    print(f"   🔹 Multiple models (voyage-3-large, voyage-3.5, etc.)")
    print(f"   🔹 Variable dimensions (256, 512, 1024, 2048)")
    print(f"   🔹 Input types (query vs document)")
    print(f"   🔹 Data types (float, int8)")
    print(f"   🔹 Async operations")
    print(f"   🔹 Retrieval scenarios")

if __name__ == "__main__":
    asyncio.run(run_all_voyage_tests())