"""
Anthropic LLM Provider

This module provides integration with Anthropic's Claude models, including support for:
- Basic text generation
- Extended thinking (reasoning models)
- Vision (images)
- Web search with citations
- Prompt caching

Run with: pytest tests/test_anthropic_provider.py -v
"""

from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import os
import re
import time
from dotenv import load_dotenv
from anthropic import Anthropic, AsyncAnthropic, RateLimitError, APIError
from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv(override=True)

# Constants
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

# =============================================================================
# Model Capability Detection
# =============================================================================

# Thinking-capable models (from official Anthropic docs)
THINKING_MODEL_PATTERNS = [
    r"claude-opus-4-6",           # Opus 4.6 (adaptive thinking recommended)
    r"claude-opus-4-5",           # Opus 4.5
    r"claude-opus-4-1",           # Opus 4.1
    r"claude-opus-4",             # Opus 4
    r"claude-sonnet-4-5",         # Sonnet 4.5
    r"claude-sonnet-4",           # Sonnet 4
    r"claude-3-7-sonnet",         # Sonnet 3.7 (deprecated but supported)
    r"claude-haiku-4-5",          # Haiku 4.5
]


@dataclass
class AnthropicModelCapabilities:
    """Capabilities of an Anthropic Claude model."""
    is_thinking_model: bool = False
    supports_vision: bool = True  # All Claude 3+ support vision
    supports_web_search: bool = True
    max_output_tokens: int = 64000  # 128k for Opus 4.6
    min_thinking_budget: int = 1024
    uses_summarized_thinking: bool = True  # Claude 4 uses summarized; 3.7 uses full


def detect_model_capabilities(model_name: str) -> AnthropicModelCapabilities:
    """
    Detect capabilities based on model name.

    Args:
        model_name: The model identifier (e.g., 'claude-sonnet-4-5-20250929')

    Returns:
        AnthropicModelCapabilities: Detected capabilities for the model
    """
    model_lower = model_name.lower()

    is_thinking = any(re.search(pattern, model_lower) for pattern in THINKING_MODEL_PATTERNS)
    is_opus_46 = "claude-opus-4-6" in model_lower
    is_sonnet_37 = "claude-3-7-sonnet" in model_lower

    return AnthropicModelCapabilities(
        is_thinking_model=is_thinking,
        max_output_tokens=128000 if is_opus_46 else 64000,
        uses_summarized_thinking=not is_sonnet_37,  # 3.7 returns full thinking
    )


# =============================================================================
# Response Extraction Helpers
# =============================================================================

def _extract_response_content(response) -> Tuple[str, Optional[str]]:
    """
    Extract text and thinking content from response.

    Args:
        response: Anthropic API response object

    Returns:
        Tuple of (generated_text, thinking_content)
    """
    generated_text = ""
    thinking_content = ""

    for block in response.content:
        if block.type == "thinking":
            thinking_content += block.thinking
        elif block.type == "text":
            generated_text += block.text

    return generated_text, thinking_content if thinking_content else None


def _build_api_params(
    model_name: str,
    messages: List[Dict],
    system: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking_budget: Optional[int],
    extra_headers: Dict,
) -> Dict:
    """
    Build API parameters for the messages.create call.

    Args:
        model_name: Model identifier
        messages: List of message dictionaries
        system: System prompt (str or list for caching)
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        thinking_budget: Token budget for thinking (if enabled)
        extra_headers: Additional headers

    Returns:
        Dict of API parameters
    """
    params = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }

    # Add thinking configuration if budget is specified
    if thinking_budget is not None:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }
        # When thinking is enabled, top_p can be 0.95-1.0, temperature is not used
        if top_p != 1.0 and 0.95 <= top_p <= 1.0:
            params["top_p"] = top_p
    else:
        # Normal mode: use temperature or top_p (not both)
        # Anthropic doesn't allow both temperature and top_p together
        if top_p != 1.0:
            params["top_p"] = top_p
        else:
            params["temperature"] = temperature

    # Add headers if present
    if extra_headers:
        params["extra_headers"] = extra_headers

    return params


# =============================================================================
# Main Generation Functions
# =============================================================================

def generate_response(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    prompt_caching: bool = False,
    cached_input: str = "",
    cache_control_type: str = "ephemeral",
    api_key: Optional[str] = None,
    json_mode: bool = False,
    thinking_budget: Optional[int] = None,
) -> Optional[Dict]:
    """
    Generate a response using the Anthropic API via the official SDK.

    Args:
        model_name: The model to use (e.g., 'claude-sonnet-4-5-20250514')
        system_prompt: The system prompt
        messages: List of message dictionaries
        temperature: Sampling temperature (0-1). Ignored when thinking is enabled.
        max_tokens: Maximum tokens for the response
        top_p: Nucleus sampling parameter. When thinking enabled, must be 0.95-1.0.
        full_response: If True, returns LLMFullResponse
        prompt_caching: Enable prompt caching
        cached_input: Cached input text
        cache_control_type: Cache control type (ephemeral or persistent)
        api_key: Anthropic API key
        json_mode: Flag for JSON mode (handled via prompt engineering)
        thinking_budget: Token budget for extended thinking (min 1024, must be < max_tokens)

    Returns:
        str or LLMFullResponse: Generated text or full response

    Raises:
        ValueError: If messages is empty/None, API key missing, or invalid thinking_budget
    """
    # Input validation
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it in environment or pass api_key parameter."
        )

    # Validate thinking_budget
    if thinking_budget is not None:
        if thinking_budget < 1024:
            raise ValueError("thinking_budget must be at least 1024 tokens")
        if thinking_budget >= max_tokens:
            raise ValueError("thinking_budget must be less than max_tokens")

    start_time = time.time()
    caps = detect_model_capabilities(model_name)

    client = Anthropic(api_key=api_key)

    # Build system parameter
    if prompt_caching:
        system = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": cached_input, "cache_control": {"type": cache_control_type}}
        ]
    else:
        system = system_prompt

    # Build extra headers
    extra_headers = {}
    if prompt_caching:
        extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"

    # Build API parameters
    params = _build_api_params(
        model_name=model_name,
        messages=messages,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        thinking_budget=thinking_budget,
        extra_headers=extra_headers,
    )

    # Execute with retry logic
    response = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(**params)
            break
        except RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise
        except APIError as e:
            last_error = e
            # Retry on server errors (5xx)
            if hasattr(e, 'status_code') and e.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise
            else:
                raise

    if response is None:
        raise last_error or RuntimeError("Failed to get response after retries")

    # Extract content
    generated_text, thinking_content = _extract_response_content(response)

    if full_response:
        return LLMFullResponse(
            generated_text=generated_text,
            model=model_name,
            process_time=time.time() - start_time,
            input_token_count=response.usage.input_tokens,
            output_token_count=response.usage.output_tokens,
            llm_provider_response=response,
            thinking_content=thinking_content,
            finish_reason=response.stop_reason,
            is_reasoning_model=caps.is_thinking_model and thinking_budget is not None,
        )
    return generated_text


async def generate_response_async(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    prompt_caching: bool = False,
    cached_input: str = "",
    cache_control_type: str = "ephemeral",
    api_key: Optional[str] = None,
    json_mode: bool = False,
    thinking_budget: Optional[int] = None,
) -> Optional[Dict]:
    """
    Asynchronously generate a response using the Anthropic API via the official SDK.

    Args:
        model_name: The model to use (e.g., 'claude-sonnet-4-5-20250514')
        system_prompt: The system prompt
        messages: List of message dictionaries
        temperature: Sampling temperature (0-1). Ignored when thinking is enabled.
        max_tokens: Maximum tokens for the response
        top_p: Nucleus sampling parameter. When thinking enabled, must be 0.95-1.0.
        full_response: If True, returns LLMFullResponse
        prompt_caching: Enable prompt caching
        cached_input: Cached input text
        cache_control_type: Cache control type (ephemeral or persistent)
        api_key: Anthropic API key
        json_mode: Flag for JSON mode (handled via prompt engineering)
        thinking_budget: Token budget for extended thinking (min 1024, must be < max_tokens)

    Returns:
        str or LLMFullResponse: Generated text or full response

    Raises:
        ValueError: If messages is empty/None, API key missing, or invalid thinking_budget
    """
    # Input validation
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it in environment or pass api_key parameter."
        )

    # Validate thinking_budget
    if thinking_budget is not None:
        if thinking_budget < 1024:
            raise ValueError("thinking_budget must be at least 1024 tokens")
        if thinking_budget >= max_tokens:
            raise ValueError("thinking_budget must be less than max_tokens")

    start_time = time.time()
    caps = detect_model_capabilities(model_name)

    client = AsyncAnthropic(api_key=api_key)

    # Build system parameter
    if prompt_caching:
        system = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": cached_input, "cache_control": {"type": cache_control_type}}
        ]
    else:
        system = system_prompt

    # Build extra headers
    extra_headers = {}
    if prompt_caching:
        extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"

    # Build API parameters
    params = _build_api_params(
        model_name=model_name,
        messages=messages,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        thinking_budget=thinking_budget,
        extra_headers=extra_headers,
    )

    # Execute with retry logic
    import asyncio
    response = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(**params)
            break
        except RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise
        except APIError as e:
            last_error = e
            # Retry on server errors (5xx)
            if hasattr(e, 'status_code') and e.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise
            else:
                raise

    if response is None:
        raise last_error or RuntimeError("Failed to get response after retries")

    # Extract content
    generated_text, thinking_content = _extract_response_content(response)

    if full_response:
        return LLMFullResponse(
            generated_text=generated_text,
            model=model_name,
            process_time=time.time() - start_time,
            input_token_count=response.usage.input_tokens,
            output_token_count=response.usage.output_tokens,
            llm_provider_response=response,
            thinking_content=thinking_content,
            finish_reason=response.stop_reason,
            is_reasoning_model=caps.is_thinking_model and thinking_budget is not None,
        )
    return generated_text


# =============================================================================
# Web Search Functions
# =============================================================================

def generate_response_with_web_search(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    max_tokens: int = 300,
    full_response: bool = False,
    api_key: Optional[str] = None,
) -> Optional[Dict]:
    """
    Generate a response using Anthropic's Messages API with web search enabled.

    Args:
        model_name: The model to use (e.g., 'claude-sonnet-4-5-20250514')
        system_prompt: The system prompt
        messages: List of message dictionaries
        max_tokens: Maximum tokens for the response
        full_response: If True, returns LLMFullResponse with web_sources
        api_key: Anthropic API key

    Returns:
        str or LLMFullResponse: Generated text or full response with web sources

    Raises:
        ValueError: If messages is empty/None or API key is missing
    """
    # Input validation
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it in environment or pass api_key parameter."
        )

    start_time = time.time() if full_response else None

    client = Anthropic(api_key=api_key)

    # Execute with retry logic
    response = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                }]
            )
            break
        except RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise
        except APIError as e:
            last_error = e
            if hasattr(e, 'status_code') and e.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise
            else:
                raise

    if response is None:
        raise last_error or RuntimeError("Failed to get response after retries")

    # Extract text and citations from response content
    generated_text = ""
    web_sources = []

    for content_block in response.content:
        if content_block.type == "text":
            generated_text += content_block.text
            # Extract citations if present
            if hasattr(content_block, 'citations') and content_block.citations:
                for citation in content_block.citations:
                    if hasattr(citation, 'type') and citation.type == "web_search_result_location":
                        web_sources.append({
                            "title": getattr(citation, 'title', ''),
                            "url": getattr(citation, 'url', ''),
                            "cited_text": getattr(citation, 'cited_text', ''),
                        })

    if full_response:
        return LLMFullResponse(
            generated_text=generated_text,
            model=model_name,
            process_time=time.time() - start_time,
            input_token_count=response.usage.input_tokens,
            output_token_count=response.usage.output_tokens,
            llm_provider_response=response,
            web_sources=web_sources if web_sources else None,
            finish_reason=response.stop_reason,
        )
    return generated_text


async def generate_response_with_web_search_async(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    max_tokens: int = 300,
    full_response: bool = False,
    api_key: Optional[str] = None,
) -> Optional[Dict]:
    """
    Asynchronously generate a response using Anthropic's Messages API with web search enabled.

    Args:
        model_name: The model to use (e.g., 'claude-sonnet-4-5-20250514')
        system_prompt: The system prompt
        messages: List of message dictionaries
        max_tokens: Maximum tokens for the response
        full_response: If True, returns LLMFullResponse with web_sources
        api_key: Anthropic API key

    Returns:
        str or LLMFullResponse: Generated text or full response with web sources

    Raises:
        ValueError: If messages is empty/None or API key is missing
    """
    import asyncio

    # Input validation
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it in environment or pass api_key parameter."
        )

    start_time = time.time() if full_response else None

    client = AsyncAnthropic(api_key=api_key)

    # Execute with retry logic
    response = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                }]
            )
            break
        except RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise
        except APIError as e:
            last_error = e
            if hasattr(e, 'status_code') and e.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise
            else:
                raise

    if response is None:
        raise last_error or RuntimeError("Failed to get response after retries")

    # Extract text and citations from response content
    generated_text = ""
    web_sources = []

    for content_block in response.content:
        if content_block.type == "text":
            generated_text += content_block.text
            # Extract citations if present
            if hasattr(content_block, 'citations') and content_block.citations:
                for citation in content_block.citations:
                    if hasattr(citation, 'type') and citation.type == "web_search_result_location":
                        web_sources.append({
                            "title": getattr(citation, 'title', ''),
                            "url": getattr(citation, 'url', ''),
                            "cited_text": getattr(citation, 'cited_text', ''),
                        })

    if full_response:
        return LLMFullResponse(
            generated_text=generated_text,
            model=model_name,
            process_time=time.time() - start_time,
            input_token_count=response.usage.input_tokens,
            output_token_count=response.usage.output_tokens,
            llm_provider_response=response,
            web_sources=web_sources if web_sources else None,
            finish_reason=response.stop_reason,
        )
    return generated_text
