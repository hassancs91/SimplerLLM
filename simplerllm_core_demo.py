"""
SimplerLLM Core Demo

A concise demonstration of the three key functionalities:
1. How to create LLMs
2. How to use reliable LLMs
3. How to generate pydantic models

This script provides a minimal working example of each feature.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field

# Import SimplerLLM components
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_reliable


# Define a simple Pydantic model for product recommendations
class ProductFeature(BaseModel):
    name: str
    description: str
    importance: int  # 1-10 scale

class ProductRecommendation(BaseModel):
    product_name: str
    price_range: str
    target_audience: str
    key_features: List[ProductFeature]
    pros: List[str]
    cons: List[str]
    overall_rating: int  # 1-10 scale


def main():
    """
    Main function demonstrating the three key functionalities.
    """
    print("\n=== SimplerLLM Core Demo ===\n")
    
    # 1. Creating LLMs
    print("1. Creating LLMs with different providers")
    print("-" * 40)
    
    # Create an OpenAI LLM
    openai_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        temperature=0.7,
        verbose=True
    )
    print(f"Created OpenAI LLM with model: {openai_llm.model_name}")
    
    # Create an Anthropic LLM
    anthropic_llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.3,
        verbose=True
    )
    print(f"Created Anthropic LLM with model: {anthropic_llm.model_name}")
    
    # Basic usage example
    prompt = "Explain quantum computing in one sentence."
    print(f"\nTesting OpenAI LLM with prompt: '{prompt}'")
    
    response = openai_llm.generate_response(
        prompt=prompt,
        system_prompt="You are a helpful AI assistant that explains complex topics concisely.",
        max_tokens=100
    )
    print(f"Response: {response}")
    
    # 2. Reliable LLMs
    print("\n\n2. Using Reliable LLMs with fallback capability")
    print("-" * 40)
    
    # Create a ReliableLLM with primary and secondary providers
    reliable_llm = ReliableLLM(
        primary_llm=openai_llm,
        secondary_llm=anthropic_llm,
        verbose=True
    )
    print("Created ReliableLLM with OpenAI as primary and Anthropic as secondary")
    
    # Test the reliable LLM
    prompt = "What are three emerging trends in artificial intelligence?"
    print(f"\nTesting ReliableLLM with prompt: '{prompt}'")
    
    response, provider, model_name = reliable_llm.generate_response(
        prompt=prompt,
        system_prompt="You are a technology trend analyst.",
        max_tokens=150,
        return_provider=True
    )
    
    print(f"Response from {provider.name} ({model_name}):")
    print(response)
    
    # 3. Pydantic Generation
    print("\n\n3. Generating structured data with Pydantic models")
    print("-" * 40)
    
    # Create a prompt for product recommendation
    product_prompt = "Provide a detailed recommendation for a high-end smartphone."
    print(f"Generating structured product recommendation with prompt: '{product_prompt}'")
    
    # Generate the model using ReliableLLM
    result = generate_pydantic_json_model_reliable(
        model_class=ProductRecommendation,
        prompt=product_prompt,
        reliable_llm=reliable_llm,
        temperature=0.3,
        max_retries=2,
        full_response=True
    )
    
    # Print the results
    print(f"\nGenerated Product Recommendation:")
    print(f"Product: {result.model_object.product_name}")
    print(f"Price Range: {result.model_object.price_range}")
    print(f"Target Audience: {result.model_object.target_audience}")
    print(f"Overall Rating: {result.model_object.overall_rating}/10")
    
    print("\nKey Features:")
    for feature in result.model_object.key_features:
        print(f"  - {feature.name}: {feature.description} (Importance: {feature.importance}/10)")
    
    print("\nPros:")
    for pro in result.model_object.pros:
        print(f"  + {pro}")
    
    print("\nCons:")
    for con in result.model_object.cons:
        print(f"  - {con}")
    
    print(f"\nProvider used: {result.provider.name}")
    print(f"Model used: {result.model_name}")
    print(f"Input tokens: {result.input_token_count}")
    print(f"Output tokens: {result.output_token_count}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set in environment variables.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set in environment variables.")
    
    # Run the demo
    main()
