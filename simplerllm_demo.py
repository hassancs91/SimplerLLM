"""
SimplerLLM Demo Script

This script demonstrates key functionalities of the SimplerLLM library:
1. How to create LLMs with different providers
2. How to use reliable LLMs with fallback capability
3. How to generate structured data using Pydantic models

Requirements:
- SimplerLLM library
- API keys for LLM providers (set in environment variables)
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Import SimplerLLM components
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import (
    generate_pydantic_json_model,
    generate_pydantic_json_model_reliable,
    generate_pydantic_json_model_async,
    generate_pydantic_json_model_reliable_async
)


#################################################
# PART 1: Creating LLMs with Different Providers #
#################################################

def demo_llm_creation():
    """
    Demonstrates how to create LLM instances with different providers.
    """
    print("\n" + "="*50)
    print("DEMO: Creating LLMs with Different Providers")
    print("="*50)
    
    # Example 1: Create an OpenAI LLM
    openai_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        temperature=0.7,
        top_p=1.0,
        verbose=True
    )
    print(f"Created OpenAI LLM with model: {openai_llm.model_name}")
    
    # Example 2: Create a Gemini LLM
    gemini_llm = LLM.create(
        provider=LLMProvider.GEMINI,
        model_name="gemini-pro",
        temperature=0.5,
        verbose=True
    )
    print(f"Created Gemini LLM with model: {gemini_llm.model_name}")
    
    # Example 3: Create an Anthropic LLM
    anthropic_llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.3,
        verbose=True
    )
    print(f"Created Anthropic LLM with model: {anthropic_llm.model_name}")
    
    # Example 4: Create an Ollama LLM (local)
    ollama_llm = LLM.create(
        provider=LLMProvider.OLLAMA,
        model_name="llama3",
        temperature=0.8,
        verbose=True
    )
    print(f"Created Ollama LLM with model: {ollama_llm.model_name}")
    
    # Example 5: Create a DeepSeek LLM
    deepseek_llm = LLM.create(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        temperature=0.6,
        verbose=True
    )
    print(f"Created DeepSeek LLM with model: {deepseek_llm.model_name}")
    
    # Return the created LLMs for use in other demos
    return {
        "openai": openai_llm,
        "gemini": gemini_llm,
        "anthropic": anthropic_llm,
        "ollama": ollama_llm,
        "deepseek": deepseek_llm
    }


def demo_basic_llm_usage(llms):
    """
    Demonstrates basic usage of LLM instances.
    """
    print("\n" + "="*50)
    print("DEMO: Basic LLM Usage")
    print("="*50)
    
    # Choose an LLM to demonstrate with (OpenAI in this case)
    llm = llms["openai"]
    
    # Example 1: Simple text completion
    prompt = "Explain quantum computing in one paragraph."
    print(f"\nPrompt: {prompt}")
    
    response = llm.generate_response(
        prompt=prompt,
        system_prompt="You are a helpful AI assistant that explains complex topics concisely.",
        max_tokens=150
    )
    print(f"\nResponse:\n{response}")
    
    # Example 2: Chat-based interaction
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"},
    ]
    print(f"\nChat Messages: {messages}")
    
    response = llm.generate_response(
        messages=messages,
        max_tokens=200
    )
    print(f"\nResponse:\n{response}")
    
    # Example 3: Getting the full response with token counts
    prompt = "List five benefits of regular exercise."
    print(f"\nPrompt: {prompt}")
    
    full_response = llm.generate_response(
        prompt=prompt,
        system_prompt="You are a health and fitness expert.",
        max_tokens=200,
        full_response=True
    )
    
    print(f"\nResponse:\n{full_response.generated_text}")
    print(f"Input tokens: {full_response.input_token_count}")
    print(f"Output tokens: {full_response.output_token_count}")


#############################################
# PART 2: Using Reliable LLMs with Fallback #
#############################################

def demo_reliable_llm(llms):
    """
    Demonstrates how to create and use ReliableLLM with fallback capability.
    """
    print("\n" + "="*50)
    print("DEMO: Using Reliable LLMs with Fallback")
    print("="*50)
    
    # Create a ReliableLLM with primary and secondary providers
    reliable_llm = ReliableLLM(
        primary_llm=llms["openai"],
        secondary_llm=llms["anthropic"],
        verbose=True
    )
    print("Created ReliableLLM with OpenAI as primary and Anthropic as secondary")
    
    # Example 1: Basic usage of ReliableLLM
    prompt = "What are three emerging trends in artificial intelligence?"
    print(f"\nPrompt: {prompt}")
    
    response = reliable_llm.generate_response(
        prompt=prompt,
        system_prompt="You are a technology trend analyst.",
        max_tokens=200
    )
    print(f"\nResponse:\n{response}")
    
    # Example 2: Using ReliableLLM with provider information
    prompt = "Suggest three books on machine learning for beginners."
    print(f"\nPrompt: {prompt}")
    
    response, provider, model_name = reliable_llm.generate_response(
        prompt=prompt,
        system_prompt="You are a helpful librarian.",
        max_tokens=200,
        return_provider=True
    )
    
    print(f"\nResponse:\n{response}")
    print(f"Provider used: {provider.name}")
    print(f"Model used: {model_name}")
    
    # Example 3: Demonstrating fallback (simulated)
    print("\nSimulating primary provider failure to demonstrate fallback...")
    
    # Create a ReliableLLM with a deliberately invalid primary provider
    # Note: This is just for demonstration purposes
    invalid_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="non-existent-model",  # Invalid model name
        verbose=True
    )
    
    fallback_reliable_llm = ReliableLLM(
        primary_llm=invalid_llm,
        secondary_llm=llms["anthropic"],
        verbose=True
    )
    
    # The primary provider should fail, and it should fall back to the secondary
    try:
        response, provider, model_name = fallback_reliable_llm.generate_response(
            prompt="What is the capital of France?",
            max_tokens=50,
            return_provider=True
        )
        
        print(f"\nResponse after fallback:\n{response}")
        print(f"Provider used: {provider.name}")
        print(f"Model used: {model_name}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Note: In a real scenario, the fallback would work if the primary provider is properly configured but fails at runtime.")
    
    return reliable_llm


#################################################
# PART 3: Generating Structured Pydantic Models #
#################################################

# Define example Pydantic models
class ProductFeature(BaseModel):
    name: str
    description: str
    importance: int = Field(description="Importance score from 1-10")

class ProductRecommendation(BaseModel):
    product_name: str
    price_range: str
    target_audience: str
    key_features: List[ProductFeature]
    pros: List[str]
    cons: List[str]
    overall_rating: int = Field(description="Overall rating from 1-10")

class WeatherCondition(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    description: str
    precipitation_chance: float

class WeatherForecast(BaseModel):
    location: str
    current_conditions: WeatherCondition
    forecast: List[WeatherCondition]
    alerts: Optional[List[str]] = None

def demo_pydantic_generation(llm, reliable_llm):
    """
    Demonstrates how to generate structured data using Pydantic models.
    """
    print("\n" + "="*50)
    print("DEMO: Generating Structured Pydantic Models")
    print("="*50)
    
    # Example 1: Basic Pydantic model generation
    print("\nExample 1: Basic Pydantic model generation")
    
    # Define a simple model
    class Topic(BaseModel):
        main_topic: str
        sub_topics: List[str]
        description: str
    
    # Create a prompt
    topic_prompt = "Generate a list of subtopics for artificial intelligence."
    
    # Generate the model
    print(f"Prompt: {topic_prompt}")
    result = generate_pydantic_json_model(
        model_class=Topic,
        prompt=topic_prompt,
        llm_instance=llm,
        temperature=0.5,
        max_retries=2,
        full_response=True
    )
    
    # Print the results
    print(f"\nGenerated Topic Model:")
    print(f"Main Topic: {result.model_object.main_topic}")
    print(f"Description: {result.model_object.description}")
    print("Sub-topics:")
    for i, subtopic in enumerate(result.model_object.sub_topics, 1):
        print(f"  {i}. {subtopic}")
    print(f"Input tokens: {result.input_token_count}")
    print(f"Output tokens: {result.output_token_count}")
    
    # Example 2: Complex Pydantic model generation with ReliableLLM
    print("\nExample 2: Complex Pydantic model generation with ReliableLLM")
    
    # Create a prompt for product recommendation
    product_prompt = "Provide a detailed recommendation for a high-end smartphone."
    
    # Generate the model using ReliableLLM
    print(f"Prompt: {product_prompt}")
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
    print("Key Features:")
    for feature in result.model_object.key_features:
        print(f"  - {feature.name}: {feature.description} (Importance: {feature.importance}/10)")
    print("Pros:")
    for pro in result.model_object.pros:
        print(f"  + {pro}")
    print("Cons:")
    for con in result.model_object.cons:
        print(f"  - {con}")
    print(f"Provider used: {result.provider.name}")
    print(f"Model used: {result.model_name}")


async def demo_async_pydantic_generation(llm, reliable_llm):
    """
    Demonstrates asynchronous generation of Pydantic models.
    """
    print("\n" + "="*50)
    print("DEMO: Asynchronous Pydantic Model Generation")
    print("="*50)
    
    # Example: Asynchronous weather forecast generation
    print("\nGenerating weather forecast asynchronously...")
    
    # Create a prompt for weather forecast
    weather_prompt = "Generate a detailed weather forecast for New York City."
    
    # Generate the model asynchronously
    result = await generate_pydantic_json_model_async(
        model_class=WeatherForecast,
        prompt=weather_prompt,
        llm_instance=llm,
        temperature=0.4,
        max_retries=2,
        full_response=True
    )
    
    # Print the results
    print(f"\nGenerated Weather Forecast for {result.model_object.location}:")
    print("Current Conditions:")
    current = result.model_object.current_conditions
    print(f"  Temperature: {current.temperature}°C")
    print(f"  Humidity: {current.humidity}%")
    print(f"  Wind Speed: {current.wind_speed} km/h")
    print(f"  Description: {current.description}")
    print(f"  Precipitation Chance: {current.precipitation_chance*100}%")
    
    print("\nForecast:")
    for i, day in enumerate(result.model_object.forecast, 1):
        print(f"  Day {i}:")
        print(f"    Temperature: {day.temperature}°C")
        print(f"    Description: {day.description}")
        print(f"    Precipitation Chance: {day.precipitation_chance*100}%")
    
    if result.model_object.alerts:
        print("\nAlerts:")
        for alert in result.model_object.alerts:
            print(f"  ! {alert}")
    
    print(f"\nInput tokens: {result.input_token_count}")
    print(f"Output tokens: {result.output_token_count}")


#################################################
# PART 4: Combined Example - Real-world Scenario #
#################################################

class ArticleSection(BaseModel):
    title: str
    content: str
    key_points: List[str]

class Article(BaseModel):
    title: str
    introduction: str
    sections: List[ArticleSection]
    conclusion: str
    references: Optional[List[str]] = None

async def demo_combined_example(reliable_llm):
    """
    Demonstrates a real-world scenario combining all three functionalities.
    """
    print("\n" + "="*50)
    print("DEMO: Combined Example - AI Article Generator")
    print("="*50)
    
    print("\nGenerating a structured article about AI advancements...")
    
    # Create a prompt for article generation
    article_prompt = """
    Generate a comprehensive article about recent advancements in artificial intelligence.
    The article should cover major breakthroughs, applications, and future directions.
    Include specific examples and references where appropriate.
    """
    
    # Generate the article using ReliableLLM with Pydantic
    result = await generate_pydantic_json_model_reliable_async(
        model_class=Article,
        prompt=article_prompt,
        reliable_llm=reliable_llm,
        temperature=0.4,
        max_tokens=2000,
        max_retries=2,
        full_response=True
    )
    
    # Print the article
    article = result.model_object
    print(f"\n{'='*30}")
    print(f"ARTICLE: {article.title.upper()}")
    print(f"{'='*30}")
    print(f"\n{article.introduction}\n")
    
    for i, section in enumerate(article.sections, 1):
        print(f"SECTION {i}: {section.title}")
        print(f"{section.content}\n")
        print("Key Points:")
        for point in section.key_points:
            print(f"  • {point}")
        print()
    
    print(f"CONCLUSION:")
    print(f"{article.conclusion}\n")
    
    if article.references:
        print(f"REFERENCES:")
        for i, ref in enumerate(article.references, 1):
            print(f"  {i}. {ref}")
    
    print(f"\nArticle generated using {result.provider.name} ({result.model_name})")
    print(f"Input tokens: {result.input_token_count}")
    print(f"Output tokens: {result.output_token_count}")


#################################################
# Main Function                                 #
#################################################

async def main():
    """
    Main function to run all demos.
    """
    print("\n" + "#"*70)
    print("# SimplerLLM Demo Script")
    print("# Demonstrating LLM Creation, Reliable LLMs, and Pydantic Generation")
    print("#"*70)
    
    # Part 1: LLM Creation
    llms = demo_llm_creation()
    demo_basic_llm_usage(llms)
    
    # Part 2: Reliable LLM
    reliable_llm = demo_reliable_llm(llms)
    
    # Part 3: Pydantic Generation
    demo_pydantic_generation(llms["openai"], reliable_llm)
    await demo_async_pydantic_generation(llms["openai"], reliable_llm)
    
    # Part 4: Combined Example
    await demo_combined_example(reliable_llm)
    
    print("\n" + "#"*70)
    print("# Demo Complete")
    print("#"*70)


if __name__ == "__main__":
    # Set up environment variables for API keys if not already set
    # In a real application, these would be set in the environment or .env file
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set in environment variables.")
        print("Set your API keys in the environment or .env file before running.")
    
    # Run the main function
    asyncio.run(main())
