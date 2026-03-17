"""
Reference Script: Web Search + Pydantic Output (SimplerLLM)

This script demonstrates how to combine web search with Pydantic structured output.
Works with BOTH thinking models (GPT-5, o1, o3) AND standard models (gpt-4o).

FEATURES:
    1. WEB SEARCH - Real-time data retrieval from the internet
    2. PYDANTIC OUTPUT - Validated structured JSON responses
    3. THINKING SUPPORT - Optional extended reasoning (GPT-5, o1, o3)

MODEL TYPES:
    STANDARD MODELS (gpt-4o, gpt-4o-mini):
        - Use web_search=True only
        - No reasoning_effort parameter needed
        - Faster, lower cost

    THINKING MODELS (gpt-5, o1, o3):
        - Use web_search=True + reasoning_effort
        - Extended reasoning before output
        - Better for complex analysis

USE CASES:
    - Research summaries with citations
    - News/current events analysis
    - Product comparisons with live data
    - Market research with sources
    - Fact-checking with references

SUPPORTED PROVIDERS FOR WEB SEARCH:
    - OpenAI: gpt-4o, gpt-4o-mini, gpt-5 (via Responses API)
    - Perplexity: sonar, sonar-pro, sonar-reasoning (always enabled)
    - Anthropic: claude-3.5-sonnet+ (native support)
    - Gemini: Latest models (native support)

KEY PARAMETERS:
    web_search: bool = True                   # Enable web search
    reasoning_effort: "low"|"medium"|"high"   # Thinking depth (thinking models only)
    full_response: bool = True                # Get web_sources + metadata

USAGE:
    python websearch_pydantic_thinking_model.py
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, RootModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import (
    generate_pydantic_json_model,
    generate_pydantic_json_model_async,
)


# =============================================================================
# PYDANTIC MODEL DEFINITIONS
# =============================================================================

# Example 1: Research Summary
class ResearchSummary(BaseModel):
    """Model for research queries with web search."""
    topic: str = Field(description="The research topic")
    key_findings: List[str] = Field(description="List of key findings from research")
    sources_count: int = Field(ge=0, description="Number of sources referenced")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence level based on source quality"
    )
    summary: str = Field(description="Brief summary of the research")


# Example 2: News Analysis
class NewsAnalysis(BaseModel):
    """Model for news/current events analysis."""
    headline: str = Field(description="Main headline or topic")
    summary: str = Field(description="Summary of the news")
    key_entities: List[str] = Field(description="People, companies, or places mentioned")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment of the news"
    )
    implications: List[str] = Field(description="Potential implications or impacts")


# Example 3: Product Comparison
class ProductInfo(BaseModel):
    """Single product information."""
    name: str = Field(description="Product name")
    price_range: str = Field(description="Price range (e.g., '$500-$800')")
    pros: List[str] = Field(description="List of advantages")
    cons: List[str] = Field(description="List of disadvantages")


class ProductComparison(BaseModel):
    """Model for product research and comparison."""
    category: str = Field(description="Product category")
    products: List[ProductInfo] = Field(description="List of products compared")
    recommendation: str = Field(description="Overall recommendation")
    best_for: str = Field(description="Best use case scenario")


# Example 4: Fact Check Result
class FactCheckResult(BaseModel):
    """Model for fact-checking claims."""
    claim: str = Field(description="The claim being fact-checked")
    verdict: Literal["true", "mostly_true", "mixed", "mostly_false", "false"] = Field(
        description="Fact-check verdict"
    )
    explanation: str = Field(description="Explanation of the verdict")
    supporting_evidence: List[str] = Field(description="Evidence supporting the verdict")
    sources_referenced: int = Field(ge=0, description="Number of sources checked")


# =============================================================================
# LLM SETUP
# =============================================================================

def create_web_search_llm(model_name: str = "gpt-5") -> LLM:
    """
    Create an LLM instance configured for web search + thinking.

    Args:
        model_name: The model to use. Options:
            - OpenAI: "gpt-5", "gpt-4o", "gpt-4o-mini"
            - Thinking: "o3", "o3-mini", "o1", "o1-mini"

    Returns:
        Configured LLM instance
    """
    return LLM.create(
        provider=LLMProvider.OPENAI,
        model_name=model_name,
    )


# =============================================================================
# EXAMPLE 1: Basic Web Search + Thinking + Pydantic
# =============================================================================

def example_basic_web_search():
    """
    Basic example combining web search, thinking model, and Pydantic output.

    This is the simplest pattern for getting structured web research data.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Web Search + Thinking + Pydantic")
    print("="*60)

    llm = create_web_search_llm("gpt-5")

    result = generate_pydantic_json_model(
        model_class=ResearchSummary,
        prompt="Research the latest developments in quantum computing in 2025",
        llm_instance=llm,
        web_search=True,            # Enable web search
        reasoning_effort="medium",  # Thinking depth
        max_tokens=4096,
    )

    if isinstance(result, str):
        print(f"Error: {result}")
        return None

    print(f"Topic: {result.topic}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources Referenced: {result.sources_count}")
    print("\nKey Findings:")
    for i, finding in enumerate(result.key_findings, 1):
        print(f"  {i}. {finding}")
    print(f"\nSummary: {result.summary}")

    return result


# =============================================================================
# EXAMPLE 2: Full Response with Web Sources (Citations)
# =============================================================================

def example_with_web_sources():
    """
    Get full response including web sources (citations).

    The full_response=True option provides:
    - model_object: Your validated Pydantic model
    - web_sources: List of {"title": "...", "url": "..."} citations
    - reasoning_tokens: Tokens used for thinking
    - process_time: Time taken for generation
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Full Response with Web Sources")
    print("="*60)

    llm = create_web_search_llm("gpt-5")

    response = generate_pydantic_json_model(
        model_class=ResearchSummary,
        prompt="What are the most significant AI breakthroughs announced this month?",
        llm_instance=llm,
        web_search=True,
        reasoning_effort="high",
        full_response=True,  # Enable full response with metadata
        max_tokens=4096,
    )

    if isinstance(response, str):
        print(f"Error: {response}")
        return None

    # Access the Pydantic model
    research = response.model_object
    print(f"\nTopic: {research.topic}")
    print(f"Confidence: {research.confidence}")
    print(f"\nKey Findings:")
    for finding in research.key_findings:
        print(f"  - {finding}")

    # Access web sources (citations)
    print("\n--- Web Sources (Citations) ---")
    if response.web_sources:
        for i, source in enumerate(response.web_sources, 1):
            title = source.get('title', 'No title')
            url = source.get('url', 'No URL')
            print(f"  {i}. {title}")
            print(f"     {url}")
    else:
        print("  No web sources returned")

    # Access metadata
    print("\n--- Metadata ---")
    print(f"Model: {response.model}")
    print(f"Process Time: {response.process_time:.2f}s")
    print(f"Reasoning Tokens: {response.reasoning_tokens}")
    print(f"Input Tokens: {response.input_token_count}")
    print(f"Output Tokens: {response.output_token_count}")

    return response


# =============================================================================
# EXAMPLE 3: News Analysis with Sentiment
# =============================================================================

def example_news_analysis():
    """
    Analyze current news/events with structured output.

    Great for media monitoring, sentiment analysis, and trend tracking.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: News Analysis with Sentiment")
    print("="*60)

    llm = create_web_search_llm("gpt-5")

    response = generate_pydantic_json_model(
        model_class=NewsAnalysis,
        prompt="Analyze the latest news about electric vehicle market trends",
        llm_instance=llm,
        web_search=True,
        reasoning_effort="high",
        full_response=True,
        max_tokens=4096,
    )

    if isinstance(response, str):
        print(f"Error: {response}")
        return None

    news = response.model_object
    print(f"\nHeadline: {news.headline}")
    print(f"Sentiment: {news.sentiment}")
    print(f"\nSummary: {news.summary}")
    print(f"\nKey Entities: {', '.join(news.key_entities)}")
    print(f"\nImplications:")
    for imp in news.implications:
        print(f"  - {imp}")

    # Show sources
    if response.web_sources:
        print(f"\nBased on {len(response.web_sources)} sources")

    return response


# =============================================================================
# EXAMPLE 4: Product Comparison
# =============================================================================

def example_product_comparison():
    """
    Research and compare products with live web data.

    Useful for purchase decisions, market analysis, and competitor research.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Product Comparison")
    print("="*60)

    llm = create_web_search_llm("gpt-5")

    response = generate_pydantic_json_model(
        model_class=ProductComparison,
        prompt="Compare the top 3 noise-canceling headphones available in 2025",
        llm_instance=llm,
        web_search=True,
        reasoning_effort="high",
        full_response=True,
        max_tokens=4096,
    )

    if isinstance(response, str):
        print(f"Error: {response}")
        return None

    comparison = response.model_object
    print(f"\nCategory: {comparison.category}")
    print(f"Best For: {comparison.best_for}")
    print(f"\nProducts:")

    for product in comparison.products:
        print(f"\n  {product.name} ({product.price_range})")
        print(f"    Pros: {', '.join(product.pros)}")
        print(f"    Cons: {', '.join(product.cons)}")

    print(f"\nRecommendation: {comparison.recommendation}")

    return response


# =============================================================================
# EXAMPLE 5: Fact Checking
# =============================================================================

def example_fact_check():
    """
    Fact-check a claim using web search for verification.

    The thinking model reasons through evidence before providing a verdict.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Fact Checking")
    print("="*60)

    llm = create_web_search_llm("gpt-5")

    response = generate_pydantic_json_model(
        model_class=FactCheckResult,
        prompt="Fact check: 'Python is the most popular programming language in 2025'",
        llm_instance=llm,
        web_search=True,
        reasoning_effort="high",  # High reasoning for thorough fact-checking
        full_response=True,
        max_tokens=4096,
    )

    if isinstance(response, str):
        print(f"Error: {response}")
        return None

    fact_check = response.model_object
    print(f"\nClaim: {fact_check.claim}")
    print(f"Verdict: {fact_check.verdict.upper()}")
    print(f"\nExplanation: {fact_check.explanation}")
    print(f"\nSupporting Evidence:")
    for evidence in fact_check.supporting_evidence:
        print(f"  - {evidence}")
    print(f"\nSources Checked: {fact_check.sources_referenced}")

    # Show actual web sources
    if response.web_sources:
        print(f"\nWeb Sources Used:")
        for source in response.web_sources[:3]:  # Show first 3
            print(f"  - {source.get('title', 'N/A')}")

    return response


# =============================================================================
# EXAMPLE 6: Async Web Search
# =============================================================================

async def example_async_web_search():
    """
    Asynchronous web search for non-blocking operations.

    Use generate_pydantic_json_model_async for async/await patterns.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Async Web Search")
    print("="*60)

    llm = create_web_search_llm("gpt-5")

    response = await generate_pydantic_json_model_async(
        model_class=ResearchSummary,
        prompt="Research recent developments in renewable energy storage",
        llm_instance=llm,
        web_search=True,
        reasoning_effort="medium",
        full_response=True,
        max_tokens=4096,
    )

    if isinstance(response, str):
        print(f"Error: {response}")
        return None

    research = response.model_object
    print(f"Topic: {research.topic}")
    print(f"Key Findings: {len(research.key_findings)} findings")
    print(f"Sources: {len(response.web_sources or [])} web sources")

    return response


# =============================================================================
# USING PERPLEXITY (ALWAYS HAS WEB SEARCH)
# =============================================================================

def example_perplexity():
    """
    Using Perplexity which has built-in web search.

    Perplexity models (sonar, sonar-pro, sonar-reasoning) always include
    web search - no need to set web_search=True.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Using Perplexity (Built-in Web Search)")
    print("="*60)

    # Create Perplexity LLM
    llm = LLM.create(
        provider=LLMProvider.PERPLEXITY,
        model_name="sonar-pro",  # or "sonar", "sonar-reasoning"
    )

    response = generate_pydantic_json_model(
        model_class=ResearchSummary,
        prompt="What are the latest trends in artificial intelligence?",
        llm_instance=llm,
        # web_search is always enabled for Perplexity
        full_response=True,
        max_tokens=4096,
    )

    if isinstance(response, str):
        print(f"Error: {response}")
        return None

    research = response.model_object
    print(f"Topic: {research.topic}")
    print(f"Confidence: {research.confidence}")

    # Perplexity always returns web sources
    if response.web_sources:
        print(f"\nSources ({len(response.web_sources)}):")
        for source in response.web_sources:
            print(f"  - {source.get('url', 'N/A')}")

    return response


# =============================================================================
# ERROR HANDLING PATTERN
# =============================================================================

def example_error_handling():
    """
    Proper error handling for web search + pydantic generation.

    Returns:
        - BaseModel: Success (validated Pydantic model)
        - LLMFullResponse: Success with full_response=True
        - str: Error message
    """
    print("\n" + "="*60)
    print("ERROR HANDLING PATTERN")
    print("="*60)

    llm = create_web_search_llm("gpt-5")

    response = generate_pydantic_json_model(
        model_class=ResearchSummary,
        prompt="Research current AI trends",
        llm_instance=llm,
        web_search=True,
        reasoning_effort="medium",
        full_response=True,
        max_tokens=4096,
    )

    # Pattern: Check for error (string) vs success (LLMFullResponse)
    if isinstance(response, str):
        print(f"Error occurred: {response}")
        # Handle error: retry, fallback, log, etc.
        return None

    # Success: Access structured data and sources
    model = response.model_object
    sources = response.web_sources or []

    print(f"Success! Got {len(model.key_findings)} findings from {len(sources)} sources")

    return response


# =============================================================================
# STANDARD MODEL EXAMPLES (gpt-4o, gpt-4o-mini - NO THINKING)
# =============================================================================

def example_standard_model_basic():
    """
    Web search + Pydantic with a STANDARD model (gpt-4o).

    Standard models like gpt-4o do NOT support reasoning_effort.
    Simply omit the reasoning_effort parameter - it will be ignored if passed.

    This is faster and cheaper than thinking models.
    """
    print("\n" + "="*60)
    print("STANDARD MODEL: Basic Web Search (gpt-4o)")
    print("="*60)

    # Use standard model (not a thinking model)
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",  # Standard model
    )

    result = generate_pydantic_json_model(
        model_class=ResearchSummary,
        prompt="Research the latest developments in electric vehicles",
        llm_instance=llm,
        web_search=True,       # Web search works with standard models
        # NO reasoning_effort - not needed for standard models
        max_tokens=4096,
    )

    if isinstance(result, str):
        print(f"Error: {result}")
        return None

    print(f"Topic: {result.topic}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources Referenced: {result.sources_count}")
    print(f"\nKey Findings:")
    for finding in result.key_findings:
        print(f"  - {finding}")

    return result


def example_standard_model_with_sources():
    """
    Standard model (gpt-4o) with full response including web sources.

    Note: reasoning_tokens will be None for standard models.
    """
    print("\n" + "="*60)
    print("STANDARD MODEL: Full Response with Sources (gpt-4o)")
    print("="*60)

    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
    )

    response = generate_pydantic_json_model(
        model_class=NewsAnalysis,
        prompt="What are today's top technology news stories?",
        llm_instance=llm,
        web_search=True,
        full_response=True,
        max_tokens=4096,
    )

    if isinstance(response, str):
        print(f"Error: {response}")
        return None

    news = response.model_object
    print(f"\nHeadline: {news.headline}")
    print(f"Sentiment: {news.sentiment}")
    print(f"Key Entities: {', '.join(news.key_entities)}")

    # Web sources work the same way
    if response.web_sources:
        print(f"\n--- Web Sources ({len(response.web_sources)}) ---")
        for source in response.web_sources[:3]:
            print(f"  - {source.get('title', 'N/A')}")

    # Metadata - note: reasoning_tokens is None for standard models
    print(f"\n--- Metadata ---")
    print(f"Model: {response.model}")
    print(f"Process Time: {response.process_time:.2f}s")
    print(f"Reasoning Tokens: {response.reasoning_tokens}")  # None for standard models
    print(f"Is Reasoning Model: {response.is_reasoning_model}")  # False

    return response


def example_compare_standard_vs_thinking():
    """
    Compare standard model vs thinking model for the same task.

    This shows the key differences in parameters and response metadata.
    """
    print("\n" + "="*60)
    print("COMPARISON: Standard vs Thinking Model")
    print("="*60)

    prompt = "Research the current state of quantum computing"

    # --- Standard Model (gpt-4o) ---
    print("\n--- Using gpt-4o (Standard) ---")
    llm_standard = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    response_standard = generate_pydantic_json_model(
        model_class=ResearchSummary,
        prompt=prompt,
        llm_instance=llm_standard,
        web_search=True,
        # No reasoning_effort for standard models
        full_response=True,
        max_tokens=4096,
    )

    if not isinstance(response_standard, str):
        print(f"Process Time: {response_standard.process_time:.2f}s")
        print(f"Reasoning Tokens: {response_standard.reasoning_tokens}")  # None
        print(f"Output Tokens: {response_standard.output_token_count}")

    # --- Thinking Model (gpt-5) ---
    print("\n--- Using gpt-5 (Thinking) ---")
    llm_thinking = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-5")

    response_thinking = generate_pydantic_json_model(
        model_class=ResearchSummary,
        prompt=prompt,
        llm_instance=llm_thinking,
        web_search=True,
        reasoning_effort="medium",  # Only for thinking models
        full_response=True,
        max_tokens=4096,
    )

    if not isinstance(response_thinking, str):
        print(f"Process Time: {response_thinking.process_time:.2f}s")
        print(f"Reasoning Tokens: {response_thinking.reasoning_tokens}")  # Has value
        print(f"Output Tokens: {response_thinking.output_token_count}")

    return response_standard, response_thinking


# =============================================================================
# HELPER: Print Web Sources
# =============================================================================

def print_web_sources(response):
    """Helper to print web sources from a full response."""
    if not hasattr(response, 'web_sources') or not response.web_sources:
        print("No web sources available")
        return

    print(f"\n--- Web Sources ({len(response.web_sources)}) ---")
    for i, source in enumerate(response.web_sources, 1):
        title = source.get('title', 'Untitled')
        url = source.get('url', 'No URL')
        print(f"{i}. {title}")
        print(f"   {url}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("SimplerLLM: Web Search + Pydantic Output")
    print("="*60)

    # ==========================================================================
    # THINKING MODEL EXAMPLES (gpt-5, o1, o3)
    # ==========================================================================
    # example_basic_web_search()           # Basic thinking + web search
    # example_with_web_sources()           # Full response with citations
    # example_news_analysis()              # News sentiment analysis
    # example_product_comparison()         # Product research
    # example_fact_check()                 # Fact checking with reasoning
    # asyncio.run(example_async_web_search())  # Async variant

    # ==========================================================================
    # STANDARD MODEL EXAMPLES (gpt-4o, gpt-4o-mini)
    # ==========================================================================
    # example_standard_model_basic()       # Basic web search (no thinking)
    # example_standard_model_with_sources() # Full response with sources
    # example_compare_standard_vs_thinking() # Side-by-side comparison

    # ==========================================================================
    # OTHER PROVIDERS
    # ==========================================================================
    # example_perplexity()                 # Perplexity (built-in web search)

    # ==========================================================================
    # UTILITIES
    # ==========================================================================
    # example_error_handling()             # Error handling pattern

    # Quick demo - Standard model (faster)
    print("\nRunning demo: Standard Model (gpt-4o) with Web Search...")
    example_standard_model_with_sources()
