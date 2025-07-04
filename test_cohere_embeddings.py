from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider
import os

def test_cohere_embeddings():
    """Test Cohere embeddings functionality"""
    print("Testing Cohere embeddings integration...")
    
    # Check if API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("COHERE_API_KEY not found in environment variables")
        return
    
    # Initialize the embeddings instance
    embeddings_llm = EmbeddingsLLM.create(
        provider=EmbeddingsProvider.COHERE,
        model_name="embed-english-v3.0"
    )
    
    if embeddings_llm is None:
        print("Failed to create Cohere embeddings instance")
        return
    
    print("Cohere embeddings instance created successfully")
    
    # Test basic embedding generation
    try:
        text = "This is a test document for embedding generation."
        embedding = embeddings_llm.generate_embeddings(text)
        print(f"Generated embedding length: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error in basic embedding generation: {e}")
        return
    
    # Test with different input types
    try:
        query_embedding = embeddings_llm.generate_embeddings(
            "What is machine learning?",
            input_type="search_query"
        )
        print(f"Query embedding length: {len(query_embedding)}")
        
        doc_embedding = embeddings_llm.generate_embeddings(
            "Machine learning is a subset of artificial intelligence.",
            input_type="search_document"
        )
        print(f"Document embedding length: {len(doc_embedding)}")
    except Exception as e:
        print(f"Error with different input types: {e}")
    
    # Test batch embeddings
    try:
        texts = [
            "First document about AI",
            "Second document about machine learning",
            "Third document about neural networks"
        ]
        batch_embeddings = embeddings_llm.generate_embeddings(texts)
        print(f"Batch embeddings count: {len(batch_embeddings)}")
        print(f"Each embedding length: {len(batch_embeddings[0])}")
    except Exception as e:
        print(f"Error in batch embedding generation: {e}")
    
    # Test full response
    try:
        response = embeddings_llm.generate_embeddings(
            "Test text for full response",
            full_response=True
        )
        print(f"Full response type: {type(response)}")
        if hasattr(response, 'generated_embedding'):
            print(f"Embedding length: {len(response.generated_embedding)}")
            print(f"Model: {response.model}")
            print(f"Process time: {response.process_time:.3f}s")
    except Exception as e:
        print(f"Error in full response generation: {e}")

async def test_cohere_embeddings_async():
    """Test Cohere embeddings async functionality"""
    print("\nTesting Cohere embeddings async integration...")
    
    # Check if API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("COHERE_API_KEY not found in environment variables")
        return
    
    # Initialize the embeddings instance
    embeddings_llm = EmbeddingsLLM.create(
        provider=EmbeddingsProvider.COHERE,
        model_name="embed-english-v3.0"
    )
    
    try:
        text = "This is a test for async embedding generation."
        embedding = await embeddings_llm.generate_embeddings_async(text)
        print(f"Async embedding length: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error in async embedding generation: {e}")

if __name__ == "__main__":
    import asyncio
    
    test_cohere_embeddings()
    asyncio.run(test_cohere_embeddings_async())