"""
Perplexity LLM Provider - Production-ready interface for Perplexity AI models.

This module provides low-level API functions for interacting with Perplexity's
chat completions API. All Perplexity models have built-in web search, always
returning citations from searched sources.

Model Categories:
    Standard Models:
        - sonar: Fast, cost-effective model with web search ($0.25/$2.50 per 1M tokens)
        - sonar-pro: Higher capability model with more comprehensive search (200K context)

    Reasoning Models (Extended Thinking):
        - sonar-reasoning-pro: Advanced reasoning with step-by-step thinking (128K context)

    Research Models:
        - sonar-deep-research: Deep research with comprehensive source analysis

Vision Support:
    All sonar models support vision via base64 or HTTPS URLs (50MB limit).
    Supported formats: PNG, JPEG, GIF, WebP

Perplexity-Specific Features:
    - Built-in web search (always enabled, returns citations)
    - search_domain_filter: Include/exclude specific domains (max 20)
    - search_recency_filter: Filter by time (day, week, month, year)
    - return_images: Include images in search results
    - return_related_questions: Suggest related queries

Environment Variables:
    PERPLEXITY_API_KEY: API key for authentication
    MAX_RETRIES: Number of retry attempts (default: 3)
    RETRY_DELAY: Base delay between retries in seconds (default: 2)

Example:
    >>> from SimplerLLM.language.llm_providers import perplexity_llm
    >>>
    >>> # Basic generation (with built-in web search)
    >>> response = perplexity_llm.generate_response(
    ...     model_name="sonar",
    ...     messages=[{"role": "user", "content": "What is quantum computing?"}],
    ...     max_tokens=500
    ... )
    >>>
    >>> # With search filters
    >>> response = perplexity_llm.generate_response(
    ...     model_name="sonar-pro",
    ...     messages=[{"role": "user", "content": "Latest AI news"}],
    ...     search_domain_filter=["reuters.com", "bbc.com"],
    ...     search_recency_filter="week",
    ...     full_response=True
    ... )
    >>> for source in response.web_sources or []:
    ...     print(f"Citation: {source['url']}")
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Literal
import os
import time
import asyncio
import logging
import requests
import aiohttp
from dotenv import load_dotenv
from .llm_response_models import LLMFullResponse

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Constants
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# =============================================================================
# Model Detection and Configuration
# =============================================================================

# Model patterns for capability detection
REASONING_MODEL_PATTERNS = ["sonar-reasoning", "reasoning-pro"]
RESEARCH_MODEL_PATTERNS = ["deep-research"]
VISION_CAPABLE_PATTERNS = ["sonar"]  # All sonar models support vision

# Valid recency filter options
VALID_RECENCY_FILTERS = ["day", "week", "month", "year"]


@dataclass
class PerplexityModelCapabilities:
    """
    Capabilities and constraints for a Perplexity model.

    This dataclass encapsulates all the feature flags and constraints
    for a specific model, enabling automatic parameter adjustment.

    Attributes:
        is_reasoning_model: Whether the model supports extended thinking/reasoning.
        is_research_model: Whether the model performs deep research.
        supports_vision: Whether the model can process images.
        supports_json_mode: Whether the model supports JSON output format.
        has_builtin_web_search: Always True for Perplexity (web search is built-in).
        default_max_tokens: Default token limit for this model type.
        recommended_max_tokens: Recommended token limit for reasoning models.
    """
    is_reasoning_model: bool = False
    is_research_model: bool = False
    supports_vision: bool = True
    supports_json_mode: bool = True
    has_builtin_web_search: bool = True  # Always True for Perplexity
    default_max_tokens: int = 300
    recommended_max_tokens: int = 4000


def detect_model_capabilities(model_name: str) -> PerplexityModelCapabilities:
    """
    Detect the capabilities and constraints of a Perplexity model.

    This function analyzes the model name to determine its capabilities,
    constraints, and the appropriate parameters to use when calling the API.

    Args:
        model_name: The model identifier (e.g., 'sonar', 'sonar-reasoning-pro').

    Returns:
        PerplexityModelCapabilities: A dataclass containing all model capabilities.

    Example:
        >>> caps = detect_model_capabilities('sonar-reasoning-pro')
        >>> caps.is_reasoning_model
        True
        >>> caps.supports_vision
        True

        >>> caps = detect_model_capabilities('sonar-pro')
        >>> caps.is_reasoning_model
        False
    """
    model_lower = model_name.lower()
    caps = PerplexityModelCapabilities()

    # Check if it's a reasoning model
    for pattern in REASONING_MODEL_PATTERNS:
        if pattern in model_lower:
            caps.is_reasoning_model = True
            caps.default_max_tokens = 4000
            caps.recommended_max_tokens = 8000
            break

    # Check if it's a research model
    for pattern in RESEARCH_MODEL_PATTERNS:
        if pattern in model_lower:
            caps.is_research_model = True
            caps.default_max_tokens = 4000
            break

    # All sonar models support vision
    caps.supports_vision = any(p in model_lower for p in VISION_CAPABLE_PATTERNS)

    return caps


def _validate_recency_filter(recency_filter: Optional[str]) -> None:
    """
    Validate the search_recency_filter parameter.

    Args:
        recency_filter: The recency filter value to validate.

    Raises:
        ValueError: If the recency filter is invalid.
    """
    if recency_filter is not None and recency_filter not in VALID_RECENCY_FILTERS:
        raise ValueError(
            f"Invalid search_recency_filter '{recency_filter}'. "
            f"Must be one of: {VALID_RECENCY_FILTERS}"
        )


def _build_api_params(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    json_mode: bool,
    search_domain_filter: Optional[List[str]],
    search_recency_filter: Optional[str],
    return_images: bool,
    return_related_questions: bool,
    caps: PerplexityModelCapabilities,
) -> Dict[str, Any]:
    """
    Build the API parameters dictionary based on model capabilities.

    Args:
        model_name: The model identifier.
        messages: List of message dictionaries.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        json_mode: Whether to force JSON output format.
        search_domain_filter: List of domains to include/exclude.
        search_recency_filter: Time filter for search results.
        return_images: Whether to include images in results.
        return_related_questions: Whether to return related questions.
        caps: PerplexityModelCapabilities instance for the target model.

    Returns:
        Dict[str, Any]: Parameters ready for the Perplexity API call.
    """
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    # Note: Perplexity doesn't support response_format like OpenAI.
    # JSON mode is handled via prompt instructions in the wrapper layer.
    # The json_mode parameter is accepted for API compatibility but has no effect here.

    # Add Perplexity-specific search parameters
    if search_domain_filter:
        params["search_domain_filter"] = search_domain_filter
    if search_recency_filter:
        params["search_recency_filter"] = search_recency_filter
    if return_images:
        params["return_images"] = return_images
    if return_related_questions:
        params["return_related_questions"] = return_related_questions

    return params


def _extract_web_sources(result: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Extract web sources/citations from the API response.

    Perplexity returns citations in the 'citations' field of the response.

    Args:
        result: The raw API response dictionary.

    Returns:
        Optional[List[Dict]]: List of web sources with 'title', 'url', and 'date' keys.
    """
    web_sources = []

    # Check for citations array (URLs) - newer API format
    citations = result.get("citations", [])
    if citations:
        for i, url in enumerate(citations):
            web_sources.append({
                "title": f"Source {i + 1}",
                "url": url,
                "date": "",
            })

    # Also check for search_results if present (older API format)
    search_results = result.get("search_results", [])
    if search_results:
        for source in search_results:
            web_sources.append({
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "date": source.get("date", ""),
            })

    return web_sources if web_sources else None


def generate_response(
    model_name: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    search_domain_filter: Optional[List[str]] = None,
    search_recency_filter: Optional[Literal["day", "week", "month", "year"]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response using Perplexity's Chat Completions API.

    All Perplexity models have built-in web search - every request performs
    a web search and returns citations. This is a core feature, not optional.

    Args:
        model_name: The Perplexity model to use. Options include:
            - 'sonar': Fast, cost-effective model
            - 'sonar-pro': Higher capability model
            - 'sonar-reasoning-pro': Advanced reasoning model
            - 'sonar-deep-research': Deep research model
        messages: List of message dicts with 'role' and 'content' keys.
            For vision, content can include image_url objects.
        temperature: Sampling temperature (0-2). Default: 0.7
        max_tokens: Maximum tokens to generate. Default: 300
            (4000+ recommended for reasoning models)
        top_p: Nucleus sampling parameter. Default: 1.0
        full_response: If True, returns LLMFullResponse with metadata
            including web_sources. If False, returns just the text. Default: False
        api_key: Perplexity API key. Falls back to PERPLEXITY_API_KEY env var.
        json_mode: Accepted for API compatibility. For JSON output with Perplexity,
            include "Return JSON" or similar instructions in your prompt. Default: False
        search_domain_filter: List of domains to filter search results.
            Prefix with "-" to exclude (e.g., ["-example.com"]). Max 20 domains.
            Default: None
        search_recency_filter: Filter results by time. Options:
            "day", "week", "month", "year". Default: None (no filter)
        return_images: If True, include images in search results. Default: False
        return_related_questions: If True, return related query suggestions.
            Default: False
        timeout: Request timeout in seconds. Default: None (no timeout)

    Returns:
        str: Generated text if full_response=False
        LLMFullResponse: Full response object if full_response=True, including:
            - generated_text: The model's response
            - model: Model name used
            - process_time: Time taken in seconds
            - input_token_count: Prompt tokens used
            - output_token_count: Completion tokens used
            - web_sources: List of citations from web search
            - finish_reason: Why generation stopped
            - is_reasoning_model: True if reasoning model was used

    Raises:
        ValueError: If messages is None or empty, or invalid parameters
        Exception: On API errors after retries exhausted

    Example:
        >>> # Basic usage (with built-in web search)
        >>> response = generate_response(
        ...     model_name="sonar",
        ...     messages=[{"role": "user", "content": "What is AI?"}],
        ...     max_tokens=500
        ... )

        >>> # With domain filter and recency
        >>> response = generate_response(
        ...     model_name="sonar-pro",
        ...     messages=[{"role": "user", "content": "Latest AI news"}],
        ...     search_domain_filter=["reuters.com", "bbc.com"],
        ...     search_recency_filter="week",
        ...     full_response=True
        ... )
        >>> for source in response.web_sources or []:
        ...     print(f"Source: {source['url']}")

        >>> # Reasoning model
        >>> response = generate_response(
        ...     model_name="sonar-reasoning-pro",
        ...     messages=[{"role": "user", "content": "Analyze the impact of..."}],
        ...     max_tokens=4000,
        ...     full_response=True
        ... )

    Notes:
        - Web search is ALWAYS performed - it's a core Perplexity feature
        - Citations are returned in web_sources when full_response=True
        - Reasoning models may use more tokens for internal reasoning
    """
    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    _validate_recency_filter(search_recency_filter)

    start_time = time.time() if full_response else None

    # Get API key
    api_key = api_key or os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY not provided and not found in environment")

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    # Build request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    params = _build_api_params(
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        search_domain_filter=search_domain_filter,
        search_recency_filter=search_recency_filter,
        return_images=return_images,
        return_related_questions=return_related_questions,
        caps=caps,
    )

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                PERPLEXITY_API_URL,
                headers=headers,
                json=params,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()

            # Extract response text
            generated_text = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0].get("finish_reason", "stop")

            # Extract web sources (citations)
            web_sources = _extract_web_sources(result)

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=result.get("usage", {}).get("prompt_tokens", 0),
                    output_token_count=result.get("usage", {}).get("completion_tokens", 0),
                    llm_provider_response=result,
                    web_sources=web_sources,
                    finish_reason=finish_reason,
                    is_reasoning_model=caps.is_reasoning_model,
                )
            return generated_text

        except requests.exceptions.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"HTTP error {e.response.status_code}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                error_detail = ""
                try:
                    error_detail = e.response.json().get("error", {}).get("message", "")
                except Exception:
                    error_detail = e.response.text[:500] if e.response.text else ""
                raise Exception(
                    f"Perplexity API error after {MAX_RETRIES} attempts: "
                    f"{e.response.status_code} - {error_detail}"
                )

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Request timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Perplexity API timeout after {MAX_RETRIES} attempts")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")


async def generate_response_async(
    model_name: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    search_domain_filter: Optional[List[str]] = None,
    search_recency_filter: Optional[Literal["day", "week", "month", "year"]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response using Perplexity's Chat Completions API.

    This is the async version of generate_response(). All Perplexity models
    have built-in web search - every request performs a web search and
    returns citations.

    Args:
        model_name: The Perplexity model to use. Options include:
            - 'sonar': Fast, cost-effective model
            - 'sonar-pro': Higher capability model
            - 'sonar-reasoning-pro': Advanced reasoning model
            - 'sonar-deep-research': Deep research model
        messages: List of message dicts with 'role' and 'content' keys.
            For vision, content can include image_url objects.
        temperature: Sampling temperature (0-2). Default: 0.7
        max_tokens: Maximum tokens to generate. Default: 300
            (4000+ recommended for reasoning models)
        top_p: Nucleus sampling parameter. Default: 1.0
        full_response: If True, returns LLMFullResponse with metadata
            including web_sources. If False, returns just the text. Default: False
        api_key: Perplexity API key. Falls back to PERPLEXITY_API_KEY env var.
        json_mode: Accepted for API compatibility. For JSON output with Perplexity,
            include "Return JSON" or similar instructions in your prompt. Default: False
        search_domain_filter: List of domains to filter search results.
            Prefix with "-" to exclude (e.g., ["-example.com"]). Max 20 domains.
            Default: None
        search_recency_filter: Filter results by time. Options:
            "day", "week", "month", "year". Default: None (no filter)
        return_images: If True, include images in search results. Default: False
        return_related_questions: If True, return related query suggestions.
            Default: False
        timeout: Request timeout in seconds. Default: None (no timeout)

    Returns:
        str: Generated text if full_response=False
        LLMFullResponse: Full response object if full_response=True

    Raises:
        ValueError: If messages is None or empty, or invalid parameters
        Exception: On API errors after retries exhausted

    Example:
        >>> # Async usage
        >>> response = await generate_response_async(
        ...     model_name="sonar",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )

        >>> # With asyncio
        >>> import asyncio
        >>> response = asyncio.run(generate_response_async(
        ...     model_name="sonar-pro",
        ...     messages=[{"role": "user", "content": "Latest news"}],
        ...     full_response=True
        ... ))
        >>> print(response.web_sources)

    See Also:
        generate_response: Synchronous version of this function.
    """
    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    _validate_recency_filter(search_recency_filter)

    start_time = time.time() if full_response else None

    # Get API key
    api_key = api_key or os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY not provided and not found in environment")

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    # Build request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    params = _build_api_params(
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        search_domain_filter=search_domain_filter,
        search_recency_filter=search_recency_filter,
        return_images=return_images,
        return_related_questions=return_related_questions,
        caps=caps,
    )

    # Configure timeout for aiohttp
    client_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None

    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    PERPLEXITY_API_URL,
                    headers=headers,
                    json=params
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    # Extract response text
                    generated_text = result["choices"][0]["message"]["content"]
                    finish_reason = result["choices"][0].get("finish_reason", "stop")

                    # Extract web sources (citations)
                    web_sources = _extract_web_sources(result)

                    if full_response:
                        end_time = time.time()
                        process_time = end_time - start_time
                        return LLMFullResponse(
                            generated_text=generated_text,
                            model=model_name,
                            process_time=process_time,
                            input_token_count=result.get("usage", {}).get("prompt_tokens", 0),
                            output_token_count=result.get("usage", {}).get("completion_tokens", 0),
                            llm_provider_response=result,
                            web_sources=web_sources,
                            finish_reason=finish_reason,
                            is_reasoning_model=caps.is_reasoning_model,
                        )
                    return generated_text

            except aiohttp.ClientResponseError as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"HTTP error {e.status}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(
                        f"Perplexity API error after {MAX_RETRIES} attempts: {e.status}"
                    )

            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Request timeout, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Perplexity API timeout after {MAX_RETRIES} attempts")

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Error: {e}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")
