"""
CometAPI LLM Provider - Production-ready interface for models via CometAPI.

This module provides low-level API functions for interacting with CometAPI,
an all-in-one aggregator that routes requests to 500+ AI models including
OpenAI, Anthropic, Google, DeepSeek, Qwen, and many others through a single
OpenAI-compatible API.

CometAPI Model Naming:
    Models use their native names without a provider prefix.
    Examples:
        - gpt-4o-mini
        - gpt-5-mini
        - claude-sonnet-4-6
        - gemini-2.5-pro
        - deepseek-r1

Supported Features:
    - All CometAPI-available models
    - Vision support (for vision-capable models)
    - Reasoning models (o1, o3, gpt-5, deepseek-r1, etc.)
    - JSON mode
    - Embeddings (OpenAI embedding models via CometAPI)
    - Automatic model capability detection

Provider-Specific Handling:
    This module automatically detects the underlying model family and adjusts:
    - Image placement (Claude/Gemini prefer images before text)
    - System message handling (o1-mini/o1-preview don't support system messages)
    - Token parameter names (max_completion_tokens for reasoning models)
    - Temperature support (some reasoning models don't support it)

Environment Variables:
    COMETAPI_API_KEY: API key for authentication (checked first)
    COMETAPI_KEY: API key fallback (CometAPI's documented convention)
    MAX_RETRIES: Number of retry attempts (default: 3)
    RETRY_DELAY: Base delay between retries in seconds (default: 2)
    DEBUG_COMETAPI: Enable debug output (default: false)

Example:
    >>> from SimplerLLM.language.llm_providers import cometapi_llm
    >>>
    >>> # Standard model
    >>> response = cometapi_llm.generate_response(
    ...     model_name="gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
    >>>
    >>> # Different vendor through the same key
    >>> response = cometapi_llm.generate_response(
    ...     model_name="claude-sonnet-4-6",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
"""

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Literal, Union
import asyncio
import logging
import os
import time
from .llm_response_models import LLMFullResponse, LLMEmbeddingsResponse

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
DEBUG_COMETAPI = os.getenv("DEBUG_COMETAPI", "false").lower() == "true"

# CometAPI base URL
COMETAPI_BASE_URL = "https://api.cometapi.com/v1"

# =============================================================================
# Model Family and Capability Detection
# =============================================================================

# Model family detection patterns (checked in order, first match wins)
MODEL_FAMILY_PATTERNS = {
    "claude": "anthropic",
    "gemini": "google",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "grok": "xai",
    "llama": "meta",
    "mistral": "mistral",
    "kimi": "moonshot",
    "moonshot": "moonshot",
    "command": "cohere",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "chatgpt": "openai",
}

# Vision-capable model patterns
VISION_MODEL_PATTERNS = [
    "gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-4-turbo", "gpt-4-vision", "gpt-5",
    "chatgpt",
    "claude-3", "claude-4", "claude-sonnet", "claude-opus", "claude-haiku",
    "gemini-1.5", "gemini-2", "gemini-3", "gemini-flash", "gemini-pro",
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "llama-3.2-vision", "llama-4", "llava",
    "grok-vision", "grok-2-vision", "grok-4", "pixtral", "gemma-3",
]

# Reasoning model patterns
REASONING_MODEL_PATTERNS = [
    "o1", "o3", "o4", "gpt-5", "deepseek-r1", "deepseek-reasoner",
]

# Models that don't support system messages
NO_SYSTEM_MESSAGE_PATTERNS = ["o1-mini", "o1-preview"]

# Models that don't support temperature parameter
NO_TEMPERATURE_PATTERNS = ["o1", "o3", "o4"]


@dataclass
class CometAPIModelCapabilities:
    """
    Capabilities and constraints for a model accessed through CometAPI.

    CometAPI routes to multiple vendors, so capabilities depend on the
    underlying model being accessed.

    Attributes:
        model_family: The underlying vendor family (openai, anthropic, google, etc.)
        is_reasoning_model: Whether the model supports extended thinking
        supports_system_messages: Whether system role messages are accepted
        supports_temperature: Whether temperature parameter can be set
        supports_vision: Whether the model can process images
        supports_json_mode: Whether JSON output format is supported
        uses_max_completion_tokens: Whether to use max_completion_tokens vs max_tokens
        supports_reasoning_effort: Whether reasoning_effort parameter is supported
        images_before_text: Whether images should be placed before text (Claude/Gemini style)
        default_max_tokens: Default token limit for this model
        recommended_max_tokens: Recommended token limit for reasoning models
    """
    model_family: str = "unknown"
    is_reasoning_model: bool = False
    supports_system_messages: bool = True
    supports_temperature: bool = True
    supports_vision: bool = False
    supports_json_mode: bool = True
    uses_max_completion_tokens: bool = False
    supports_reasoning_effort: bool = False
    images_before_text: bool = False
    default_max_tokens: int = 300
    recommended_max_tokens: int = 4000


def detect_model_family(model_name: str) -> str:
    """
    Detect the underlying vendor family from a CometAPI model name.

    CometAPI uses native model names without provider prefixes, so the
    family is inferred from the model name itself.

    Args:
        model_name: The CometAPI model identifier (e.g., 'claude-sonnet-4-6')

    Returns:
        str: Normalized vendor family name, or "unknown" if not recognized.

    Example:
        >>> detect_model_family("gpt-4o-mini")
        'openai'
        >>> detect_model_family("claude-sonnet-4-6")
        'anthropic'
        >>> detect_model_family("gemini-2.5-pro")
        'google'
    """
    model_lower = model_name.lower()
    for pattern, family in MODEL_FAMILY_PATTERNS.items():
        if model_lower.startswith(pattern):
            return family
    return "unknown"


def detect_model_capabilities(model_name: str) -> CometAPIModelCapabilities:
    """
    Detect the capabilities and constraints of a CometAPI model.

    This function analyzes the model name to determine its capabilities,
    constraints, and the appropriate parameters to use. It handles ANY
    model through CometAPI with smart defaults.

    Args:
        model_name: The CometAPI model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-6')

    Returns:
        CometAPIModelCapabilities: A dataclass containing all model
            capabilities and constraints for automatic parameter adjustment.

    Example:
        >>> caps = detect_model_capabilities('gpt-4o')
        >>> caps.supports_vision
        True
        >>> caps.model_family
        'openai'

        >>> caps = detect_model_capabilities('o1-mini')
        >>> caps.is_reasoning_model
        True
        >>> caps.supports_system_messages
        False

        >>> caps = detect_model_capabilities('claude-sonnet-4-6')
        >>> caps.images_before_text
        True
    """
    model_lower = model_name.lower()

    caps = CometAPIModelCapabilities(
        model_family=detect_model_family(model_name),
    )

    # Check vision support
    for pattern in VISION_MODEL_PATTERNS:
        if pattern in model_lower:
            caps.supports_vision = True
            break

    # o-series reasoning models support vision, except the text-only early ones
    if model_lower.startswith(("o1", "o3", "o4")) and not model_lower.startswith(("o1-mini", "o1-preview")):
        caps.supports_vision = True

    # Unknown families: don't claim the model lacks vision - leave it to the API
    if caps.model_family == "unknown":
        caps.supports_vision = True

    # Set image placement preference based on family
    if caps.model_family in ("anthropic", "google"):
        caps.images_before_text = True

    # Check reasoning model support
    for pattern in REASONING_MODEL_PATTERNS:
        if model_lower.startswith(pattern):
            caps.is_reasoning_model = True
            caps.uses_max_completion_tokens = True
            caps.supports_reasoning_effort = True
            caps.default_max_tokens = 4000
            break

    # Check system message support
    for pattern in NO_SYSTEM_MESSAGE_PATTERNS:
        if model_lower.startswith(pattern):
            caps.supports_system_messages = False
            break

    # Check temperature support
    for pattern in NO_TEMPERATURE_PATTERNS:
        if model_lower.startswith(pattern):
            caps.supports_temperature = False
            break

    if DEBUG_COMETAPI:
        logger.info(f"Detected capabilities for {model_name}: {caps}")

    return caps


def _resolve_api_key(api_key: Optional[str]) -> str:
    """
    Resolve the CometAPI key from the argument or environment variables.

    Checks COMETAPI_API_KEY first (SimplerLLM convention), then COMETAPI_KEY
    (CometAPI's documented convention).
    """
    return api_key or os.getenv("COMETAPI_API_KEY") or os.getenv("COMETAPI_KEY", "")


# =============================================================================
# Helper Functions
# =============================================================================

def _process_messages_for_model(
    messages: List[Dict[str, Any]],
    caps: CometAPIModelCapabilities,
) -> List[Dict[str, Any]]:
    """
    Process messages to comply with model constraints.

    For models that don't support system messages (like o1-mini, o1-preview),
    this function converts system messages to user messages with a clear prefix.
    For models that don't support vision, image parts are stripped from
    multimodal content (with a warning) so the request doesn't fail.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        caps: CometAPIModelCapabilities instance for the target model.

    Returns:
        List[Dict[str, Any]]: Processed messages compatible with the model.
    """
    processed = []
    for msg in messages:
        content = msg.get("content")

        # Strip image parts for models without vision support
        if not caps.supports_vision and isinstance(content, list):
            text_parts = [
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            if len(text_parts) != len(content):
                logger.warning(
                    "Model does not support vision; image content removed from the request"
                )
                msg = {**msg, "content": " ".join(t for t in text_parts if t)}

        if msg.get("role") == "system" and not caps.supports_system_messages:
            # Convert system message to user message with prefix
            processed.append({
                "role": "user",
                "content": f"[System Instructions]: {msg['content']}"
            })
            if DEBUG_COMETAPI:
                logger.info("Converted system message to user message for model compatibility")
        else:
            processed.append(msg)

    return processed


def _build_api_params(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    json_mode: bool,
    reasoning_effort: Optional[str],
    caps: CometAPIModelCapabilities,
) -> Dict[str, Any]:
    """
    Build the API parameters dictionary based on model capabilities.

    This function creates the appropriate parameters for the CometAPI call,
    automatically adjusting for model-specific requirements.

    Args:
        model_name: The model identifier.
        messages: List of message dictionaries.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        json_mode: Whether to force JSON output format.
        reasoning_effort: Reasoning depth for thinking models.
        caps: CometAPIModelCapabilities instance for the target model.

    Returns:
        Dict[str, Any]: Parameters ready for the CometAPI call.
    """
    params = {
        "model": model_name,
        "messages": messages,
    }

    # Handle max tokens parameter
    if caps.uses_max_completion_tokens:
        # Use smart default for reasoning models if user hasn't specified
        if max_tokens == 300:  # Default value - apply smart default
            actual_max = caps.recommended_max_tokens
            if DEBUG_COMETAPI:
                logger.info(f"Using recommended max_tokens={actual_max} for reasoning model")
        else:
            actual_max = max_tokens
        params["max_completion_tokens"] = actual_max
    else:
        params["max_tokens"] = max_tokens

    # Handle temperature
    if caps.supports_temperature:
        params["temperature"] = temperature
        params["top_p"] = top_p
    else:
        if DEBUG_COMETAPI and temperature != 1.0:
            logger.info(f"Temperature parameter ignored for {model_name} (only supports temperature=1)")

    # Handle reasoning effort for thinking models
    if reasoning_effort and caps.supports_reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    elif reasoning_effort and not caps.supports_reasoning_effort:
        if DEBUG_COMETAPI:
            logger.warning(f"reasoning_effort ignored for non-reasoning model {model_name}")

    # Handle JSON mode - only send response_format if the model supports it.
    # Callers relying on JSON (e.g. generate_pydantic_json_model) still work
    # without it, since they instruct the model via the prompt and extract
    # JSON from the response text.
    if json_mode:
        if caps.supports_json_mode:
            params["response_format"] = {"type": "json_object"}
        else:
            logger.warning(
                f"Model {model_name} does not support response_format; "
                f"json_mode request relies on prompt instructions only"
            )

    return params


def _extract_reasoning_tokens(completion) -> Optional[int]:
    """
    Extract reasoning tokens from the API response.

    Args:
        completion: The CometAPI completion response.

    Returns:
        Optional[int]: Number of reasoning tokens used, or None if not available.
    """
    if hasattr(completion, 'usage') and completion.usage:
        if hasattr(completion.usage, 'completion_tokens_details'):
            details = completion.usage.completion_tokens_details
            if hasattr(details, 'reasoning_tokens'):
                return details.reasoning_tokens
    return None


def _handle_empty_reasoning_response(
    completion,
    reasoning_tokens: Optional[int],
    max_tokens_used: int,
) -> str:
    """
    Generate helpful message when reasoning model returns empty content.

    Args:
        completion: The API completion response.
        reasoning_tokens: Number of reasoning tokens used.
        max_tokens_used: The max_tokens value that was used in the request.

    Returns:
        str: Informative message about why the response is empty.
    """
    finish_reason = completion.choices[0].finish_reason if completion.choices else "unknown"

    if reasoning_tokens and finish_reason == "length":
        msg = (
            f"[Reasoning Model Notice] The model used all {reasoning_tokens} tokens "
            f"for internal reasoning, leaving no tokens for output. "
            f"Increase max_tokens to at least {reasoning_tokens + 500} for a response. "
            f"(Current: {max_tokens_used})"
        )
    else:
        msg = (
            f"[Reasoning Model Notice] Empty response received. "
            f"Finish reason: {finish_reason}"
        )

    if DEBUG_COMETAPI:
        logger.warning(msg)
        if reasoning_tokens:
            logger.info(f"Reasoning tokens used: {reasoning_tokens}")
            logger.info(f"Max tokens allowed: {max_tokens_used}")

    return msg


# =============================================================================
# Main API Functions
# =============================================================================

def generate_response(
    model_name: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response using any model through CometAPI.

    Supports all CometAPI-available models including OpenAI, Anthropic,
    Google, DeepSeek, Qwen, and many others. Automatically detects model
    capabilities and applies appropriate parameters.

    Args:
        model_name: The CometAPI model to use (e.g., 'gpt-4o-mini',
            'claude-sonnet-4-6', 'gemini-2.5-pro').
        messages: List of message dicts with 'role' and 'content' keys.
            For vision, content can be a list of text/image_url objects.
        temperature: Sampling temperature (0-2). Ignored for models that
            don't support it (o1-mini, o1-preview). Default: 0.7
        max_tokens: Maximum tokens to generate. For reasoning models, this
            includes both reasoning and output tokens. Default: 300 (4000
            recommended for reasoning models - applied automatically if
            default is used).
        top_p: Nucleus sampling parameter. Default: 1.0
        full_response: If True, returns LLMFullResponse with metadata.
            If False, returns just the generated text. Default: False
        api_key: CometAPI key. Falls back to COMETAPI_API_KEY then
            COMETAPI_KEY env vars.
        json_mode: If True, forces JSON output format. Default: False
        reasoning_effort: For reasoning models, controls thinking depth.
            Options: "low", "medium", "high". Default: None (model default)
        timeout: Request timeout in seconds. Default: None (no timeout)

    Returns:
        str: Generated text if full_response=False
        LLMFullResponse: Full response object if full_response=True, including:
            - generated_text: The model's response
            - model: Model name used
            - process_time: Time taken in seconds
            - input_token_count: Prompt tokens used
            - output_token_count: Completion tokens used
            - reasoning_tokens: Tokens used for reasoning (reasoning models only)
            - finish_reason: Why generation stopped
            - is_reasoning_model: True if reasoning model was used

    Raises:
        ValueError: If messages is None or empty
        openai.RateLimitError: On rate limit errors (after retries exhausted)
        openai.APIError: On API errors (after retries exhausted)
        Exception: On other errors after retries exhausted

    Example:
        >>> # Standard model usage
        >>> response = generate_response(
        ...     model_name="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )

        >>> # Using a different vendor
        >>> response = generate_response(
        ...     model_name="claude-sonnet-4-6",
        ...     messages=[{"role": "user", "content": "Explain quantum computing"}],
        ...     max_tokens=500
        ... )

        >>> # Reasoning model with full response
        >>> response = generate_response(
        ...     model_name="o3-mini",
        ...     messages=[{"role": "user", "content": "Solve: What is 15! / 13!?"}],
        ...     max_tokens=8000,
        ...     reasoning_effort="high",
        ...     full_response=True
        ... )
        >>> print(f"Reasoning tokens used: {response.reasoning_tokens}")
    """
    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    start_time = time.time() if full_response else None

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    if DEBUG_COMETAPI and caps.is_reasoning_model:
        logger.info(f"Detected reasoning model: {model_name}")

    # Initialize client with optional timeout
    client_kwargs = {
        "api_key": _resolve_api_key(api_key),
        "base_url": COMETAPI_BASE_URL,
    }
    if timeout:
        client_kwargs["timeout"] = timeout
    cometapi_client = OpenAI(**client_kwargs)

    # Process messages for model constraints
    processed_messages = _process_messages_for_model(messages, caps)

    # Build API parameters
    params = _build_api_params(
        model_name=model_name,
        messages=processed_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        reasoning_effort=reasoning_effort,
        caps=caps,
    )

    # Track actual max tokens for error reporting
    actual_max_tokens = params.get("max_completion_tokens", params.get("max_tokens", max_tokens))

    for attempt in range(MAX_RETRIES):
        try:
            completion = cometapi_client.chat.completions.create(**params)

            # Extract text from response
            generated_text = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason

            # Extract reasoning tokens if available
            reasoning_tokens = _extract_reasoning_tokens(completion)

            if DEBUG_COMETAPI and reasoning_tokens:
                logger.info(f"Reasoning tokens used: {reasoning_tokens}")

            # Handle empty responses from reasoning models
            if caps.is_reasoning_model and (generated_text is None or generated_text == ""):
                generated_text = _handle_empty_reasoning_response(
                    completion=completion,
                    reasoning_tokens=reasoning_tokens,
                    max_tokens_used=actual_max_tokens,
                )

            # Build and return response
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text or "",
                    model=model_name,
                    process_time=process_time,
                    input_token_count=completion.usage.prompt_tokens if completion.usage else 0,
                    output_token_count=completion.usage.completion_tokens if completion.usage else 0,
                    llm_provider_response=completion,
                    reasoning_tokens=reasoning_tokens,
                    finish_reason=finish_reason,
                    is_reasoning_model=caps.is_reasoning_model,
                )
            return generated_text or ""

        except RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_COMETAPI:
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise

        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_COMETAPI:
                    logger.warning(f"API error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_response_async(
    model_name: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response using any model through CometAPI.

    This is the async version of generate_response(). It supports all
    CometAPI-available models including OpenAI, Anthropic, Google, DeepSeek,
    and many others. Automatically detects model capabilities and applies
    appropriate parameters.

    Args:
        model_name: The CometAPI model to use (e.g., 'gpt-4o-mini',
            'claude-sonnet-4-6', 'gemini-2.5-pro').
        messages: List of message dicts with 'role' and 'content' keys.
            For vision, content can be a list of text/image_url objects.
        temperature: Sampling temperature (0-2). Ignored for models that
            don't support it (o1-mini, o1-preview). Default: 0.7
        max_tokens: Maximum tokens to generate. For reasoning models, this
            includes both reasoning and output tokens. Default: 300 (4000
            recommended for reasoning models - applied automatically if
            default is used).
        top_p: Nucleus sampling parameter. Default: 1.0
        full_response: If True, returns LLMFullResponse with metadata.
            If False, returns just the generated text. Default: False
        api_key: CometAPI key. Falls back to COMETAPI_API_KEY then
            COMETAPI_KEY env vars.
        json_mode: If True, forces JSON output format. Default: False
        reasoning_effort: For reasoning models, controls thinking depth.
            Options: "low", "medium", "high". Default: None (model default)
        timeout: Request timeout in seconds. Default: None (no timeout)

    Returns:
        str: Generated text if full_response=False
        LLMFullResponse: Full response object if full_response=True

    Raises:
        ValueError: If messages is None or empty
        openai.RateLimitError: On rate limit errors (after retries exhausted)
        openai.APIError: On API errors (after retries exhausted)
        Exception: On other errors after retries exhausted

    Example:
        >>> # Async usage
        >>> response = await generate_response_async(
        ...     model_name="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )

        >>> # With asyncio
        >>> import asyncio
        >>> response = asyncio.run(generate_response_async(
        ...     model_name="claude-sonnet-4-6",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... ))

    See Also:
        generate_response: Synchronous version of this function.
    """
    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    start_time = time.time() if full_response else None

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    if DEBUG_COMETAPI and caps.is_reasoning_model:
        logger.info(f"Detected reasoning model: {model_name}")

    # Initialize async client with optional timeout
    client_kwargs = {
        "api_key": _resolve_api_key(api_key),
        "base_url": COMETAPI_BASE_URL,
    }
    if timeout:
        client_kwargs["timeout"] = timeout
    async_cometapi_client = AsyncOpenAI(**client_kwargs)

    # Process messages for model constraints
    processed_messages = _process_messages_for_model(messages, caps)

    # Build API parameters
    params = _build_api_params(
        model_name=model_name,
        messages=processed_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        reasoning_effort=reasoning_effort,
        caps=caps,
    )

    # Track actual max tokens for error reporting
    actual_max_tokens = params.get("max_completion_tokens", params.get("max_tokens", max_tokens))

    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_cometapi_client.chat.completions.create(**params)

            # Extract text from response
            generated_text = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason

            # Extract reasoning tokens if available
            reasoning_tokens = _extract_reasoning_tokens(completion)

            if DEBUG_COMETAPI and reasoning_tokens:
                logger.info(f"Reasoning tokens used: {reasoning_tokens}")

            # Handle empty responses from reasoning models
            if caps.is_reasoning_model and (generated_text is None or generated_text == ""):
                generated_text = _handle_empty_reasoning_response(
                    completion=completion,
                    reasoning_tokens=reasoning_tokens,
                    max_tokens_used=actual_max_tokens,
                )

            # Build and return response
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text or "",
                    model=model_name,
                    process_time=process_time,
                    input_token_count=completion.usage.prompt_tokens if completion.usage else 0,
                    output_token_count=completion.usage.completion_tokens if completion.usage else 0,
                    llm_provider_response=completion,
                    reasoning_tokens=reasoning_tokens,
                    finish_reason=finish_reason,
                    is_reasoning_model=caps.is_reasoning_model,
                )
            return generated_text or ""

        except RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_COMETAPI:
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                raise

        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_COMETAPI:
                    logger.warning(f"API error: {e}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


# =============================================================================
# Embeddings Functions
# =============================================================================

def generate_embeddings(
    model_name: str,
    user_input: Optional[Union[str, List[str]]] = None,
    full_response: bool = False,
    api_key: Optional[str] = None,
) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]:
    """
    Generate embeddings using a CometAPI-available embedding model.

    Args:
        model_name: The embedding model to use (e.g., 'text-embedding-3-small')
        user_input: Text or list of texts to embed
        full_response: If True, returns LLMEmbeddingsResponse with metadata
        api_key: CometAPI key. Falls back to COMETAPI_API_KEY then
            COMETAPI_KEY env vars.

    Returns:
        List[float]: Single embedding if user_input is a string
        List[List[float]]: List of embeddings if user_input is a list
        LLMEmbeddingsResponse: Full response if full_response=True

    Raises:
        ValueError: If user_input is not provided
    """
    if not user_input:
        raise ValueError("user_input must be provided.")

    start_time = time.time() if full_response else None

    cometapi_client = OpenAI(
        api_key=_resolve_api_key(api_key),
        base_url=COMETAPI_BASE_URL
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = cometapi_client.embeddings.create(
                model=model_name,
                input=user_input
            )

            # Extract actual embedding vectors from the response
            embeddings = [item.embedding for item in response.data]

            # For single input, return single embedding; for multiple inputs, return list
            if isinstance(user_input, str):
                result_embeddings = embeddings[0] if embeddings else []
            else:
                result_embeddings = embeddings

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=result_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=response,
                )
            return result_embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_embeddings_async(
    model_name: str,
    user_input: Optional[Union[str, List[str]]] = None,
    full_response: bool = False,
    api_key: Optional[str] = None,
) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]:
    """
    Asynchronously generate embeddings using a CometAPI-available embedding model.

    Args:
        model_name: The embedding model to use (e.g., 'text-embedding-3-small')
        user_input: Text or list of texts to embed
        full_response: If True, returns LLMEmbeddingsResponse with metadata
        api_key: CometAPI key. Falls back to COMETAPI_API_KEY then
            COMETAPI_KEY env vars.

    Returns:
        List[float]: Single embedding if user_input is a string
        List[List[float]]: List of embeddings if user_input is a list
        LLMEmbeddingsResponse: Full response if full_response=True

    Raises:
        ValueError: If user_input is not provided
    """
    if not user_input:
        raise ValueError("user_input must be provided.")

    start_time = time.time() if full_response else None

    async_cometapi_client = AsyncOpenAI(
        api_key=_resolve_api_key(api_key),
        base_url=COMETAPI_BASE_URL
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = await async_cometapi_client.embeddings.create(
                model=model_name,
                input=user_input
            )

            # Extract actual embedding vectors from the response
            embeddings = [item.embedding for item in response.data]

            # For single input, return single embedding; for multiple inputs, return list
            if isinstance(user_input, str):
                result_embeddings = embeddings[0] if embeddings else []
            else:
                result_embeddings = embeddings

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=result_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=response,
                )
            return result_embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
