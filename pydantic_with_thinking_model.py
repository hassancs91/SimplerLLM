"""
Reference Script: Generating Pydantic Models with Thinking/Reasoning Models (SimplerLLM)

This script demonstrates how to use SimplerLLM to generate validated Pydantic
model instances using OpenAI thinking/reasoning models (GPT-5, o1, o3 series).

THINKING MODELS:
    - gpt-5, gpt-5-mini, gpt-5-turbo (GPT-5 series)
    - o3, o3-mini (O3 series)
    - o1, o1-mini, o1-preview (O1 series)

KEY PARAMETER:
    reasoning_effort: Controls thinking depth
    - "low": Quick reasoning, faster response
    - "medium": Balanced reasoning (recommended)
    - "high": Deep reasoning, slower but more accurate

IMPORTANT NOTES:
    1. Temperature is automatically ignored for thinking models (fixed at 1)
    2. Use max_tokens=4096 or higher (thinking uses tokens before generating output)
    3. The model uses reasoning tokens internally, then generates the JSON output

USAGE:
    python pydantic_with_thinking_model.py
"""

from typing import List, Optional
from pydantic import BaseModel, Field, RootModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import (
    generate_pydantic_json_model,
    generate_pydantic_json_model_async,
)


# =============================================================================
# PYDANTIC MODEL DEFINITIONS
# =============================================================================

# Example 1: Simple flat model
class MovieReview(BaseModel):
    """Simple Pydantic model with basic field types and constraints."""
    title: str = Field(description="Title of the movie")
    rating: float = Field(ge=1.0, le=5.0, description="Rating from 1 to 5 stars")
    summary: str = Field(description="Brief summary of the review")
    recommended: bool = Field(description="Whether the reviewer recommends this movie")


# Example 2: Nested model
class Author(BaseModel):
    """Nested model representing an author."""
    name: str = Field(description="Author's full name")
    expertise: List[str] = Field(description="List of author's areas of expertise")


class BlogPost(BaseModel):
    """Complex model with nested objects and lists."""
    title: str = Field(description="Title of the blog post")
    author: Author = Field(description="Author information")
    tags: List[str] = Field(description="List of tags for the post")
    content: str = Field(description="Main content of the blog post")
    read_time_minutes: int = Field(ge=1, description="Estimated read time in minutes")


# Example 3: List generation using RootModel
class FAQItem(BaseModel):
    """Single FAQ item."""
    question: str = Field(description="The question being asked")
    answer: str = Field(description="The answer to the question")


class FAQList(RootModel[List[FAQItem]]):
    """List of FAQ items using RootModel for direct array generation."""
    pass


# =============================================================================
# LLM SETUP
# =============================================================================

def create_thinking_llm(model_name: str = "gpt-5") -> LLM:
    """
    Create an LLM instance configured for thinking models.

    Args:
        model_name: The thinking model to use. Options:
            - "gpt-5", "gpt-5-mini", "gpt-5-turbo"
            - "o3", "o3-mini"
            - "o1", "o1-mini", "o1-preview"

    Returns:
        Configured LLM instance

    Note:
        Temperature parameter is accepted but ignored by thinking models.
    """
    return LLM.create(
        provider=LLMProvider.OPENAI,
        model_name=model_name,
        # temperature is ignored for thinking models (auto-set to 1)
    )


# =============================================================================
# EXAMPLE 1: Basic Usage with Simple Model
# =============================================================================

def example_simple_model():
    """
    Generate a simple Pydantic model using a thinking model.

    This example shows the basic pattern for structured output generation
    with reasoning_effort parameter.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Model Generation")
    print("="*60)

    llm = create_thinking_llm("gpt-5")

    result = generate_pydantic_json_model(
        model_class=MovieReview,
        prompt="Generate a thoughtful review for the movie 'Inception' (2010)",
        llm_instance=llm,
        max_tokens=4096,           # Important: Use higher value for thinking models
        reasoning_effort="medium",  # Options: "low", "medium", "high"
        max_retries=3,             # Retry on validation failures
    )

    # Check for errors (returns string on failure)
    if isinstance(result, str):
        print(f"Error: {result}")
        return None

    # Success - result is a validated Pydantic model
    print(f"Title: {result.title}")
    print(f"Rating: {result.rating}/5")
    print(f"Summary: {result.summary}")
    print(f"Recommended: {result.recommended}")

    return result


# =============================================================================
# EXAMPLE 2: Nested Model Generation
# =============================================================================

def example_nested_model():
    """
    Generate a complex Pydantic model with nested objects.

    Thinking models excel at generating properly structured nested JSON
    due to their reasoning capabilities.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Nested Model Generation")
    print("="*60)

    llm = create_thinking_llm("gpt-5")

    result = generate_pydantic_json_model(
        model_class=BlogPost,
        prompt="Generate a blog post about 'Getting Started with Python Async Programming'",
        llm_instance=llm,
        max_tokens=4096,
        reasoning_effort="high",  # Higher reasoning for complex structure
    )

    if isinstance(result, str):
        print(f"Error: {result}")
        return None

    print(f"Title: {result.title}")
    print(f"Author: {result.author.name}")
    print(f"Author Expertise: {', '.join(result.author.expertise)}")
    print(f"Tags: {', '.join(result.tags)}")
    print(f"Read Time: {result.read_time_minutes} minutes")
    print(f"Content Preview: {result.content[:200]}...")

    return result


# =============================================================================
# EXAMPLE 3: Full Response with Metadata
# =============================================================================

def example_full_response():
    """
    Generate with full_response=True to get metadata including reasoning tokens.

    The full response includes:
        - model_object: The validated Pydantic model
        - reasoning_tokens: Tokens used for internal reasoning
        - process_time: Time taken for generation
        - input_token_count: Prompt tokens used
        - output_token_count: Completion tokens used
        - is_reasoning_model: True for thinking models
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Full Response with Metadata")
    print("="*60)

    llm = create_thinking_llm("gpt-5")

    response = generate_pydantic_json_model(
        model_class=MovieReview,
        prompt="Generate a detailed review for 'The Matrix' (1999)",
        llm_instance=llm,
        max_tokens=4096,
        reasoning_effort="high",
        full_response=True,  # Enable full response with metadata
    )

    if isinstance(response, str):
        print(f"Error: {response}")
        return None

    # Access metadata
    print("\n--- Metadata ---")
    print(f"Model: {response.model}")
    print(f"Process Time: {response.process_time:.2f} seconds")
    print(f"Input Tokens: {response.input_token_count}")
    print(f"Output Tokens: {response.output_token_count}")
    print(f"Reasoning Tokens: {response.reasoning_tokens}")
    print(f"Is Reasoning Model: {response.is_reasoning_model}")

    # Access the Pydantic model via model_object attribute
    review = response.model_object
    print("\n--- Generated Review ---")
    print(f"Title: {review.title}")
    print(f"Rating: {review.rating}/5")
    print(f"Summary: {review.summary}")

    return response


# =============================================================================
# EXAMPLE 4: List Generation with RootModel
# =============================================================================

def example_list_generation():
    """
    Generate a list of items using RootModel.

    RootModel allows direct array generation without wrapping in an object.
    SimplerLLM automatically handles the array extraction.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: List Generation with RootModel")
    print("="*60)

    llm = create_thinking_llm("gpt-5")

    result = generate_pydantic_json_model(
        model_class=FAQList,
        prompt="Generate 5 frequently asked questions about Python programming for beginners",
        llm_instance=llm,
        max_tokens=4096,
        reasoning_effort="medium",
    )

    if isinstance(result, str):
        print(f"Error: {result}")
        return None

    # FAQList is a RootModel, access items via .root
    faq_items = result.root
    print(f"Generated {len(faq_items)} FAQ items:\n")

    for i, item in enumerate(faq_items, 1):
        print(f"Q{i}: {item.question}")
        print(f"A{i}: {item.answer}\n")

    return result


# =============================================================================
# EXAMPLE 5: Async Generation
# =============================================================================

async def example_async_generation():
    """
    Asynchronous generation for non-blocking operations.

    Use generate_pydantic_json_model_async for async/await patterns.
    Supports all the same parameters as the sync version.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Async Generation")
    print("="*60)

    llm = create_thinking_llm("gpt-5")

    result = await generate_pydantic_json_model_async(
        model_class=MovieReview,
        prompt="Generate a review for 'Interstellar' (2014)",
        llm_instance=llm,
        max_tokens=4096,
        reasoning_effort="medium",
    )

    if isinstance(result, str):
        print(f"Error: {result}")
        return None

    print(f"Title: {result.title}")
    print(f"Rating: {result.rating}/5")
    print(f"Recommended: {result.recommended}")

    return result


# =============================================================================
# EXAMPLE 6: Comparing Reasoning Effort Levels
# =============================================================================

def example_reasoning_effort_comparison():
    """
    Compare different reasoning_effort levels.

    - "low": Faster, less thorough reasoning
    - "medium": Balanced (recommended for most cases)
    - "high": Slower, more thorough reasoning

    Higher reasoning effort may produce better structured output
    but uses more tokens and takes longer.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Reasoning Effort Comparison")
    print("="*60)

    llm = create_thinking_llm("gpt-5")
    prompt = "Generate a review for 'Pulp Fiction' (1994)"

    for effort in ["low", "medium", "high"]:
        print(f"\n--- reasoning_effort='{effort}' ---")

        response = generate_pydantic_json_model(
            model_class=MovieReview,
            prompt=prompt,
            llm_instance=llm,
            max_tokens=4096,
            reasoning_effort=effort,
            full_response=True,
        )

        if isinstance(response, str):
            print(f"Error: {response}")
            continue

        print(f"Reasoning Tokens: {response.reasoning_tokens}")
        print(f"Process Time: {response.process_time:.2f}s")
        print(f"Rating: {response.model_object.rating}/5")


# =============================================================================
# ERROR HANDLING PATTERN
# =============================================================================

def example_error_handling():
    """
    Proper error handling pattern for generate_pydantic_json_model.

    The function returns a string on error, so always check the type:
    - str: Error message
    - BaseModel: Success (validated Pydantic model)
    - LLMFullResponse: Success with full_response=True
    """
    print("\n" + "="*60)
    print("ERROR HANDLING PATTERN")
    print("="*60)

    llm = create_thinking_llm("gpt-5")

    result = generate_pydantic_json_model(
        model_class=MovieReview,
        prompt="Generate a review",
        llm_instance=llm,
        max_tokens=4096,
        reasoning_effort="medium",
    )

    # Pattern 1: Simple check
    if isinstance(result, str):
        print(f"Generation failed: {result}")
        # Handle error (retry, fallback, etc.)
        return None

    # Pattern 2: With full_response
    # response = generate_pydantic_json_model(..., full_response=True)
    # if isinstance(response, str):
    #     print(f"Error: {response}")
    # else:
    #     model = response.model_object
    #     tokens = response.reasoning_tokens

    print("Generation successful!")
    print(f"Result type: {type(result).__name__}")
    return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("SimplerLLM: Pydantic Generation with Thinking Models")
    print("="*60)

    # Run examples (uncomment to execute)
    # example_simple_model()
    # example_nested_model()
    # example_full_response()
    # example_list_generation()
    # asyncio.run(example_async_generation())
    # example_reasoning_effort_comparison()
    # example_error_handling()

    # Quick demo
    print("\nRunning quick demo with simple model...")
    example_simple_model()
