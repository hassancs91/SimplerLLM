"""
Google Gemini LLM Provider - Production-ready implementation.

This module provides low-level API functions for interacting with Google's
Gemini language models, including support for:

- All Gemini models (1.5, 2.5, 3 series)
- Thinking models with configurable levels (Gemini 2.5/3)
- Vision capabilities (image analysis)
- Web search grounding with source attribution
- JSON mode for structured outputs
- Native system instructions
- Prompt caching

Example:
    >>> from SimplerLLM.language.llm_providers.gemini_llm import generate_response
    >>>
    >>> # Basic generation
    >>> response = generate_response(
    ...     model_name="gemini-2.5-flash",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>>
    >>> # With thinking (Gemini 3)
    >>> response = generate_response(
    ...     model_name="gemini-3-pro-preview",
    ...     messages=[{"role": "user", "content": "Solve this puzzle..."}],
    ...     thinking_level="high",
    ...     max_tokens=8000,
    ...     full_response=True
    ... )
    >>> print(f"Thinking tokens: {response.reasoning_tokens}")
    >>> print(f"Thinking process: {response.thinking_content}")
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Union, Literal
import os
import time
import json
import logging
import asyncio

from dotenv import load_dotenv
import requests
import aiohttp

from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv(override=True)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

# Model pattern constants for capability detection
GEMINI_3_PATTERNS = ["gemini-3-pro", "gemini-3-flash"]
GEMINI_2_5_PATTERNS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
VISION_CAPABLE_PATTERNS = ["gemini-1.5", "gemini-2", "gemini-3"]

# Thinking configuration options
GEMINI_3_THINKING_LEVELS = ["minimal", "low", "medium", "high"]
GEMINI_2_5_THINKING_BUDGET_MIN = -1  # -1 = dynamic
GEMINI_2_5_THINKING_BUDGET_MAX = 32768

# Finish reason mapping (Gemini -> normalized)
FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "other",
    "FINISH_REASON_UNSPECIFIED": "unknown",
}


# =============================================================================
# Model Capability Detection
# =============================================================================

@dataclass
class GeminiModelCapabilities:
    """
    Capabilities and constraints for a Gemini model.

    This dataclass captures what features a specific Gemini model supports,
    allowing the provider to automatically configure API requests correctly.

    Attributes:
        is_thinking_model: Whether the model supports extended thinking.
        supports_vision: Whether the model can process images.
        supports_web_search: Whether Google Search grounding is available.
        supports_json_mode: Whether structured JSON output is supported.
        supports_system_instruction: Whether native systemInstruction is supported.
        thinking_config_type: "level" for Gemini 3, "budget" for Gemini 2.5, None otherwise.
        is_gemini_3: True if this is a Gemini 3 series model.
        default_max_tokens: Default output token limit.
        recommended_thinking_tokens: Recommended max_tokens for thinking models.

    Example:
        >>> caps = detect_model_capabilities('gemini-3-pro-preview')
        >>> caps.is_thinking_model
        True
        >>> caps.thinking_config_type
        'level'
    """
    is_thinking_model: bool = False
    supports_vision: bool = True
    supports_web_search: bool = True
    supports_json_mode: bool = True
    supports_system_instruction: bool = True
    thinking_config_type: Optional[Literal["level", "budget"]] = None
    is_gemini_3: bool = False
    default_max_tokens: int = 300
    recommended_thinking_tokens: int = 8000


def detect_model_capabilities(model_name: str) -> GeminiModelCapabilities:
    """
    Detect the capabilities and constraints of a Gemini model.

    Automatically determines what features a model supports based on its name,
    including thinking capabilities, vision support, and configuration type.

    Args:
        model_name: The model identifier (e.g., 'gemini-2.5-pro', 'gemini-3-flash').

    Returns:
        GeminiModelCapabilities: Dataclass containing all model capabilities.

    Example:
        >>> caps = detect_model_capabilities('gemini-3-pro-preview')
        >>> caps.is_thinking_model
        True
        >>> caps.thinking_config_type
        'level'

        >>> caps = detect_model_capabilities('gemini-2.5-flash')
        >>> caps.thinking_config_type
        'budget'

        >>> caps = detect_model_capabilities('gemini-1.5-flash')
        >>> caps.is_thinking_model
        False
    """
    model_lower = model_name.lower()
    caps = GeminiModelCapabilities()

    # Check for Gemini 3 series (uses thinking levels)
    for pattern in GEMINI_3_PATTERNS:
        if pattern in model_lower:
            caps.is_thinking_model = True
            caps.thinking_config_type = "level"
            caps.is_gemini_3 = True
            caps.recommended_thinking_tokens = 16000
            break

    # Check for Gemini 2.5 series (uses thinking budget)
    if not caps.is_thinking_model:
        for pattern in GEMINI_2_5_PATTERNS:
            if pattern in model_lower:
                caps.is_thinking_model = True
                caps.thinking_config_type = "budget"
                caps.recommended_thinking_tokens = 8000
                break

    # Check vision support (all modern Gemini models support it)
    caps.supports_vision = any(p in model_lower for p in VISION_CAPABLE_PATTERNS)

    return caps


# =============================================================================
# API Parameter Building Functions
# =============================================================================

def _build_generation_config(
    temperature: float,
    max_tokens: int,
    top_p: float,
    json_mode: bool = False,
    response_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the generationConfig object for Gemini API.

    Args:
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum output tokens.
        top_p: Nucleus sampling parameter.
        json_mode: Whether to force JSON output.
        response_schema: Optional JSON schema for structured output.

    Returns:
        Dict containing generationConfig parameters.
    """
    config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "topP": top_p,
    }

    # JSON mode configuration
    if json_mode:
        config["responseMimeType"] = "application/json"
        if response_schema:
            config["responseSchema"] = response_schema

    return config


def _build_thinking_config(
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    caps: Optional[GeminiModelCapabilities] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build the thinkingConfig object for Gemini thinking models.

    Args:
        thinking_level: For Gemini 3: "minimal", "low", "medium", "high".
        thinking_budget: For Gemini 2.5: Token budget (0-32768, -1 for dynamic).
        caps: Model capabilities to determine config type.

    Returns:
        Dict containing thinkingConfig, or None if not applicable.

    Raises:
        ValueError: If invalid thinking configuration for model type.
    """
    if not caps or not caps.is_thinking_model:
        return None

    if caps.thinking_config_type == "level":
        # Gemini 3 uses thinking levels
        if thinking_level:
            if thinking_level not in GEMINI_3_THINKING_LEVELS:
                raise ValueError(
                    f"Invalid thinking_level '{thinking_level}'. "
                    f"Must be one of: {GEMINI_3_THINKING_LEVELS}"
                )
            return {"thinkingLevel": thinking_level}
        return None  # Use model default

    elif caps.thinking_config_type == "budget":
        # Gemini 2.5 uses thinking budget
        if thinking_budget is not None:
            if thinking_budget != -1 and not (0 <= thinking_budget <= GEMINI_2_5_THINKING_BUDGET_MAX):
                raise ValueError(
                    f"Invalid thinking_budget {thinking_budget}. "
                    f"Must be -1 (dynamic) or 0-{GEMINI_2_5_THINKING_BUDGET_MAX}."
                )
            return {"thinkingBudget": thinking_budget}
        return None  # Use model default

    return None


def _build_tools_config(
    web_search: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Build the tools configuration for Gemini API.

    Args:
        web_search: Whether to enable Google Search grounding.

    Returns:
        List of tool configurations, or None if no tools.
    """
    if web_search:
        return [{"google_search": {}}]
    return None


# =============================================================================
# Response Extraction Functions
# =============================================================================

def _extract_response_text(response_json: Dict[str, Any]) -> str:
    """
    Extract generated text from Gemini API response.

    Handles both regular text and thinking model responses where
    thought parts have 'thought: true' flag. Thought parts are
    excluded from the main response text.

    Args:
        response_json: Raw JSON response from Gemini API.

    Returns:
        str: The generated text (excluding thought parts).
    """
    try:
        candidates = response_json.get("candidates", [])
        if not candidates:
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        # Filter out thought parts and collect text
        text_parts = []
        for part in parts:
            # Skip thought parts (thinking models)
            if part.get("thought", False):
                continue
            if "text" in part:
                text_parts.append(part["text"])

        return "".join(text_parts)
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_thinking_content(response_json: Dict[str, Any]) -> Optional[str]:
    """
    Extract thinking/reasoning content from Gemini API response.

    For thinking models, this extracts the model's internal reasoning
    process (parts marked with 'thought: true').

    Args:
        response_json: Raw JSON response from Gemini API.

    Returns:
        Optional[str]: The thinking content, or None if not available.
    """
    try:
        candidates = response_json.get("candidates", [])
        if not candidates:
            return None

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        # Collect only thought parts
        thought_parts = []
        for part in parts:
            if part.get("thought", False) and "text" in part:
                thought_parts.append(part["text"])

        return "".join(thought_parts) if thought_parts else None
    except (KeyError, IndexError, TypeError):
        return None


def _extract_thinking_tokens(response_json: Dict[str, Any]) -> Optional[int]:
    """
    Extract thinking token count from Gemini API response.

    Args:
        response_json: Raw JSON response from Gemini API.

    Returns:
        Optional[int]: Number of thinking tokens, or None if not available.
    """
    try:
        usage_metadata = response_json.get("usageMetadata", {})
        return usage_metadata.get("thoughtsTokenCount")
    except (KeyError, TypeError):
        return None


def _extract_grounding_metadata(
    response_json: Dict[str, Any]
) -> Optional[List[Dict[str, str]]]:
    """
    Extract web search grounding metadata from Gemini API response.

    Args:
        response_json: Raw JSON response from Gemini API.

    Returns:
        Optional[List[Dict]]: List of web sources with 'title' and 'url' keys.
    """
    try:
        candidates = response_json.get("candidates", [])
        if not candidates:
            return None

        grounding_metadata = candidates[0].get("groundingMetadata", {})
        grounding_chunks = grounding_metadata.get("groundingChunks", [])

        if not grounding_chunks:
            return None

        web_sources = []
        for chunk in grounding_chunks:
            web = chunk.get("web", {})
            if web:
                web_sources.append({
                    "title": web.get("title", ""),
                    "url": web.get("uri", ""),
                })

        return web_sources if web_sources else None
    except (KeyError, TypeError):
        return None


def _extract_finish_reason(response_json: Dict[str, Any]) -> Optional[str]:
    """
    Extract and normalize finish reason from Gemini API response.

    Maps Gemini finish reasons to common normalized values:
    - STOP -> "stop"
    - MAX_TOKENS -> "length"
    - SAFETY -> "content_filter"
    - RECITATION -> "content_filter"

    Args:
        response_json: Raw JSON response from Gemini API.

    Returns:
        Optional[str]: Normalized finish reason.
    """
    try:
        candidates = response_json.get("candidates", [])
        if not candidates:
            return None

        finish_reason = candidates[0].get("finishReason", "")
        return FINISH_REASON_MAP.get(finish_reason, finish_reason.lower())
    except (KeyError, TypeError):
        return None


def _extract_token_usage(response_json: Dict[str, Any]) -> tuple:
    """
    Extract token usage from Gemini API response.

    Args:
        response_json: Raw JSON response from Gemini API.

    Returns:
        tuple: (input_tokens, output_tokens)
    """
    try:
        usage = response_json.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        return input_tokens, output_tokens
    except (KeyError, TypeError):
        return 0, 0


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
    cache_id: Optional[str] = None,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    response_schema: Optional[Dict[str, Any]] = None,
    # Thinking parameters
    thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
    thinking_budget: Optional[int] = None,
    # Vision parameters (images should be pre-processed into message parts)
    # Web search
    web_search: bool = False,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response using Google's Gemini API.

    Supports all Gemini models including thinking models (2.5/3 series),
    vision capabilities, web search grounding, and JSON mode.

    Args:
        model_name: The Gemini model to use (e.g., 'gemini-2.5-pro', 'gemini-3-flash').
        system_prompt: System instruction for the model. Uses native systemInstruction.
        messages: List of message dicts. Each dict should have:
            - 'role': 'user' or 'model'/'assistant'
            - 'content': str or list of parts (for vision)
        temperature: Sampling temperature (0-2). Default: 0.7
        max_tokens: Maximum output tokens. Default: 300
        top_p: Nucleus sampling parameter. Default: 1.0
        full_response: Return LLMFullResponse with metadata. Default: False
        prompt_caching: Enable Gemini prompt caching. Default: False
        cache_id: Cached content ID for prompt caching.
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        json_mode: Force JSON output format. Default: False
        response_schema: Optional JSON schema for structured output.
        thinking_level: For Gemini 3: "minimal", "low", "medium", "high".
        thinking_budget: For Gemini 2.5: Token budget (0-32768, -1 for dynamic).
        web_search: Enable Google Search grounding. Default: False
        timeout: Request timeout in seconds. Default: None (no timeout)

    Returns:
        str: Generated text if full_response=False
        LLMFullResponse: Full response with metadata if full_response=True

    Raises:
        ValueError: If messages is None or empty, or invalid thinking config.
        Exception: On API errors after retries exhausted.

    Example:
        >>> # Basic usage
        >>> response = generate_response(
        ...     model_name="gemini-2.5-flash",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )

        >>> # With thinking (Gemini 3)
        >>> response = generate_response(
        ...     model_name="gemini-3-pro-preview",
        ...     messages=[{"role": "user", "content": "Solve this puzzle..."}],
        ...     thinking_level="high",
        ...     max_tokens=8000,
        ...     full_response=True
        ... )
        >>> print(f"Thinking tokens: {response.reasoning_tokens}")
        >>> print(f"Thinking: {response.thinking_content}")

        >>> # With web search
        >>> response = generate_response(
        ...     model_name="gemini-2.5-pro",
        ...     messages=[{"role": "user", "content": "Latest news on AI"}],
        ...     web_search=True,
        ...     full_response=True
        ... )
        >>> print(f"Sources: {response.web_sources}")
    """
    start_time = time.time()

    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    # Get API key
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not provided and not found in environment")

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    # Build generation config
    generation_config = _build_generation_config(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        response_schema=response_schema,
    )

    # Build thinking config if applicable
    thinking_config = _build_thinking_config(
        thinking_level=thinking_level,
        thinking_budget=thinking_budget,
        caps=caps,
    )

    # Build tools config
    tools = _build_tools_config(web_search=web_search)

    # Build payload
    payload: Dict[str, Any] = {
        "generationConfig": generation_config,
    }

    # Add system instruction (native support - better than user/model workaround)
    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    # Build contents from messages
    contents = []
    for msg in messages:
        # Map role (assistant -> model for Gemini)
        role = msg.get("role", "user")
        if role in ("assistant", "model"):
            role = "model"
        else:
            role = "user"

        # Handle content (may be string or list of parts for vision)
        content = msg.get("content", "")

        if isinstance(content, str):
            parts = [{"text": content}]
        elif isinstance(content, list):
            # Already formatted as parts (e.g., for vision)
            parts = content
        else:
            parts = [{"text": str(content)}]

        contents.append({"role": role, "parts": parts})

    payload["contents"] = contents

    # Add thinking config (must be inside generationConfig per API spec)
    if thinking_config:
        payload["generationConfig"]["thinkingConfig"] = thinking_config

    # Add tools
    if tools:
        payload["tools"] = tools

    # Add cache reference
    if prompt_caching and cache_id:
        payload["cachedContent"] = cache_id

    # Build URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
            response.raise_for_status()
            response_json = response.json()

            # Check for API-level errors
            if "error" in response_json:
                error_msg = response_json["error"].get("message", "Unknown error")
                raise Exception(f"Gemini API error: {error_msg}")

            # Extract response components
            generated_text = _extract_response_text(response_json)
            thinking_content = _extract_thinking_content(response_json)
            thinking_tokens = _extract_thinking_tokens(response_json)
            finish_reason = _extract_finish_reason(response_json)
            web_sources = _extract_grounding_metadata(response_json) if web_search else None
            input_tokens, output_tokens = _extract_token_usage(response_json)

            if full_response:
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=time.time() - start_time,
                    input_token_count=input_tokens,
                    output_token_count=output_tokens,
                    llm_provider_response=response_json,
                    reasoning_tokens=thinking_tokens,
                    thinking_content=thinking_content,
                    finish_reason=finish_reason,
                    is_reasoning_model=caps.is_thinking_model,
                    web_sources=web_sources,
                )
            return generated_text

        except requests.exceptions.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"HTTP error {e.response.status_code}, retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                error_detail = ""
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("error", {}).get("message", "")
                except Exception:
                    error_detail = e.response.text[:500] if e.response.text else ""
                raise Exception(
                    f"Gemini API error after {MAX_RETRIES} attempts: "
                    f"{e.response.status_code} - {error_detail}"
                )

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Request timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(
                    f"Gemini API timeout after {MAX_RETRIES} attempts"
                )

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")

    # Should never reach here, but just in case
    raise Exception("Unexpected error in generate_response")


async def generate_response_async(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    prompt_caching: bool = False,
    cache_id: Optional[str] = None,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    response_schema: Optional[Dict[str, Any]] = None,
    thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
    thinking_budget: Optional[int] = None,
    web_search: bool = False,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response using Google's Gemini API.

    This is the async version of generate_response(). See generate_response()
    for full parameter documentation.

    Args:
        model_name: The Gemini model to use.
        system_prompt: System instruction for the model.
        messages: List of message dicts.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum output tokens.
        top_p: Nucleus sampling parameter.
        full_response: Return LLMFullResponse with metadata.
        prompt_caching: Enable Gemini prompt caching.
        cache_id: Cached content ID for prompt caching.
        api_key: Gemini API key.
        json_mode: Force JSON output format.
        response_schema: Optional JSON schema for structured output.
        thinking_level: For Gemini 3 thinking models.
        thinking_budget: For Gemini 2.5 thinking models.
        web_search: Enable Google Search grounding.
        timeout: Request timeout in seconds.

    Returns:
        str or LLMFullResponse depending on full_response parameter.

    Example:
        >>> import asyncio
        >>> async def main():
        ...     response = await generate_response_async(
        ...         model_name="gemini-2.5-flash",
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
        ...     print(response)
        >>> asyncio.run(main())
    """
    start_time = time.time()

    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    # Get API key
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not provided and not found in environment")

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    # Build generation config
    generation_config = _build_generation_config(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        response_schema=response_schema,
    )

    # Build thinking config if applicable
    thinking_config = _build_thinking_config(
        thinking_level=thinking_level,
        thinking_budget=thinking_budget,
        caps=caps,
    )

    # Build tools config
    tools = _build_tools_config(web_search=web_search)

    # Build payload
    payload: Dict[str, Any] = {
        "generationConfig": generation_config,
    }

    # Add system instruction
    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    # Build contents from messages
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        if role in ("assistant", "model"):
            role = "model"
        else:
            role = "user"

        content = msg.get("content", "")

        if isinstance(content, str):
            parts = [{"text": content}]
        elif isinstance(content, list):
            parts = content
        else:
            parts = [{"text": str(content)}]

        contents.append({"role": role, "parts": parts})

    payload["contents"] = contents

    if thinking_config:
        payload["thinkingConfig"] = thinking_config

    if tools:
        payload["tools"] = tools

    if prompt_caching and cache_id:
        payload["cachedContent"] = cache_id

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}

    # Set up timeout
    client_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    response_json = await response.json()

                    # Check for API-level errors
                    if "error" in response_json:
                        error_msg = response_json["error"].get("message", "Unknown error")
                        raise Exception(f"Gemini API error: {error_msg}")

                    # Extract response components
                    generated_text = _extract_response_text(response_json)
                    thinking_content = _extract_thinking_content(response_json)
                    thinking_tokens = _extract_thinking_tokens(response_json)
                    finish_reason = _extract_finish_reason(response_json)
                    web_sources = _extract_grounding_metadata(response_json) if web_search else None
                    input_tokens, output_tokens = _extract_token_usage(response_json)

                    if full_response:
                        return LLMFullResponse(
                            generated_text=generated_text,
                            model=model_name,
                            process_time=time.time() - start_time,
                            input_token_count=input_tokens,
                            output_token_count=output_tokens,
                            llm_provider_response=response_json,
                            reasoning_tokens=thinking_tokens,
                            thinking_content=thinking_content,
                            finish_reason=finish_reason,
                            is_reasoning_model=caps.is_thinking_model,
                            web_sources=web_sources,
                        )
                    return generated_text

        except aiohttp.ClientResponseError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"HTTP error {e.status}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise Exception(
                    f"Gemini API error after {MAX_RETRIES} attempts: {e.status}"
                )

        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Request timeout, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise Exception(f"Gemini API timeout after {MAX_RETRIES} attempts")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Error: {e}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")

    raise Exception("Unexpected error in generate_response_async")


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_response_with_web_search(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    max_tokens: int = 300,
    full_response: bool = False,
    api_key: Optional[str] = None,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response with Google Search grounding enabled.

    Convenience function that wraps generate_response with web_search=True.

    Args:
        model_name: The Gemini model to use.
        system_prompt: System instruction.
        messages: List of message dictionaries.
        max_tokens: Maximum output tokens.
        full_response: Return full response with web sources.
        api_key: Gemini API key.

    Returns:
        str or LLMFullResponse with web_sources populated.

    Example:
        >>> response = generate_response_with_web_search(
        ...     model_name="gemini-2.5-flash",
        ...     messages=[{"role": "user", "content": "Latest AI news"}],
        ...     full_response=True
        ... )
        >>> for source in response.web_sources or []:
        ...     print(f"- {source['title']}: {source['url']}")
    """
    return generate_response(
        model_name=model_name,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=max_tokens,
        full_response=full_response,
        api_key=api_key,
        web_search=True,
    )


async def generate_response_with_web_search_async(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    max_tokens: int = 300,
    full_response: bool = False,
    api_key: Optional[str] = None,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response with Google Search grounding enabled.

    Async version of generate_response_with_web_search().

    Args:
        model_name: The Gemini model to use.
        system_prompt: System instruction.
        messages: List of message dictionaries.
        max_tokens: Maximum output tokens.
        full_response: Return full response with web sources.
        api_key: Gemini API key.

    Returns:
        str or LLMFullResponse with web_sources populated.
    """
    return await generate_response_async(
        model_name=model_name,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=max_tokens,
        full_response=full_response,
        api_key=api_key,
        web_search=True,
    )
