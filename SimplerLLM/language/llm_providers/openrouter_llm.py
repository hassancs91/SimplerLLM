"""
OpenRouter LLM Provider - Production-ready interface for models via OpenRouter.

This module provides low-level API functions for interacting with OpenRouter,
a unified gateway that routes requests to multiple AI providers including
OpenAI, Anthropic, Google, Meta, Mistral, and many others.

OpenRouter Model Naming:
    Models use the format: provider/model-name
    Examples:
        - openai/gpt-4o
        - anthropic/claude-3.5-sonnet
        - google/gemini-2.0-flash
        - meta-llama/llama-3.1-70b-instruct
        - deepseek/deepseek-reasoner

Supported Features:
    - All OpenRouter-available models
    - Vision support (for vision-capable models)
    - Reasoning models (o1, o3, gpt-5 via openai/*)
    - JSON mode
    - Embeddings (for embedding-capable models)
    - Automatic model capability detection

Provider-Specific Handling:
    This module automatically detects the underlying provider and adjusts:
    - Image format and placement (Anthropic/Gemini prefer images before text)
    - System message handling (o1-mini/o1-preview don't support system messages)
    - Token parameter names (max_completion_tokens for reasoning models)
    - Temperature support (some reasoning models don't support it)

Environment Variables:
    OPENROUTER_API_KEY: API key for authentication
    OPENROUTER_SITE_URL: Your site URL for tracking (optional)
    OPENROUTER_SITE_NAME: Your site name for tracking (optional)
    MAX_RETRIES: Number of retry attempts (default: 3)
    RETRY_DELAY: Base delay between retries in seconds (default: 2)
    DEBUG_OPENROUTER: Enable debug output (default: false)

Example:
    >>> from SimplerLLM.language.llm_providers import openrouter_llm
    >>>
    >>> # Standard model
    >>> response = openrouter_llm.generate_response(
    ...     model_name="openai/gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
    >>>
    >>> # Different provider
    >>> response = openrouter_llm.generate_response(
    ...     model_name="anthropic/claude-3.5-sonnet",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
    >>>
    >>> # With vision
    >>> response = openrouter_llm.generate_response(
    ...     model_name="openai/gpt-4o",
    ...     messages=[{"role": "user", "content": vision_content}],
    ...     max_tokens=500
    ... )
"""

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Literal, Union, Tuple
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
DEBUG_OPENROUTER = os.getenv("DEBUG_OPENROUTER", "false").lower() == "true"

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# =============================================================================
# Provider and Model Detection
# =============================================================================

# Known providers (maps various prefixes to normalized names)
KNOWN_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "meta-llama": "meta",
    "meta": "meta",
    "mistralai": "mistral",
    "mistral": "mistral",
    "cohere": "cohere",
    "deepseek": "deepseek",
    "perplexity": "perplexity",
    "x-ai": "xai",
    "xai": "xai",
    "qwen": "qwen",
    "nvidia": "nvidia",
    "microsoft": "microsoft",
}

# Vision-capable model patterns by provider
VISION_MODEL_PATTERNS = {
    "openai": ["gpt-4o", "gpt-4-vision", "gpt-4-turbo", "gpt-5", "vision"],
    "anthropic": ["claude-3", "claude-3.5", "claude-4"],
    "google": ["gemini-1.5", "gemini-2", "gemini-pro-vision", "gemini-flash"],
    "meta": ["llama-3.2-vision", "llama-3.3-vision", "llava"],
    "qwen": ["qwen-vl", "qwen2-vl"],
}

# Reasoning model patterns by provider
REASONING_MODEL_PATTERNS = {
    "openai": ["o1", "o3", "gpt-5"],
    "deepseek": ["deepseek-reasoner", "deepseek-r1"],
}

# Models that don't support system messages
NO_SYSTEM_MESSAGE_PATTERNS = {
    "openai": ["o1-mini", "o1-preview"],
}

# Models that don't support temperature parameter
NO_TEMPERATURE_PATTERNS = {
    "openai": ["o1", "o3"],
}


@dataclass
class OpenRouterModelCapabilities:
    """
    Capabilities and constraints for a model accessed through OpenRouter.

    OpenRouter routes to multiple providers, so capabilities depend on the
    underlying provider and model being accessed.

    Attributes:
        provider: The underlying provider (openai, anthropic, google, meta, etc.)
        base_model: The base model name without provider prefix
        is_reasoning_model: Whether the model supports extended thinking
        supports_system_messages: Whether system role messages are accepted
        supports_temperature: Whether temperature parameter can be set
        supports_vision: Whether the model can process images
        supports_json_mode: Whether JSON output format is supported
        uses_max_completion_tokens: Whether to use max_completion_tokens vs max_tokens
        supports_reasoning_effort: Whether reasoning_effort parameter is supported
        images_before_text: Whether images should be placed before text (Anthropic/Gemini style)
        default_max_tokens: Default token limit for this model
        recommended_max_tokens: Recommended token limit for reasoning models
    """
    provider: str = "unknown"
    base_model: str = ""
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


def parse_openrouter_model_name(model_name: str) -> Tuple[str, str]:
    """
    Parse an OpenRouter model name into provider and base model.

    OpenRouter uses the format: provider/model-name

    Args:
        model_name: The full OpenRouter model identifier
            Examples: "openai/gpt-4o", "anthropic/claude-3.5-sonnet"

    Returns:
        Tuple of (provider, base_model)

    Example:
        >>> parse_openrouter_model_name("openai/gpt-4o")
        ('openai', 'gpt-4o')
        >>> parse_openrouter_model_name("anthropic/claude-3.5-sonnet")
        ('anthropic', 'claude-3.5-sonnet')
        >>> parse_openrouter_model_name("meta-llama/llama-3.1-70b")
        ('meta-llama', 'llama-3.1-70b')
    """
    if "/" in model_name:
        parts = model_name.split("/", 1)
        return (parts[0].lower(), parts[1].lower())
    else:
        # Assume OpenAI-compatible if no prefix
        return ("openai", model_name.lower())


def detect_model_capabilities(model_name: str) -> OpenRouterModelCapabilities:
    """
    Detect the capabilities and constraints of an OpenRouter model.

    This function analyzes the OpenRouter model name to determine its
    capabilities, constraints, and the appropriate parameters to use.
    It handles ANY model through OpenRouter with smart defaults.

    Args:
        model_name: The OpenRouter model identifier (e.g., 'openai/gpt-4o')

    Returns:
        OpenRouterModelCapabilities: A dataclass containing all model
            capabilities and constraints for automatic parameter adjustment.

    Example:
        >>> caps = detect_model_capabilities('openai/gpt-4o')
        >>> caps.supports_vision
        True
        >>> caps.provider
        'openai'

        >>> caps = detect_model_capabilities('openai/o1-mini')
        >>> caps.is_reasoning_model
        True
        >>> caps.supports_system_messages
        False

        >>> caps = detect_model_capabilities('anthropic/claude-3.5-sonnet')
        >>> caps.images_before_text
        True
    """
    provider_raw, base_model = parse_openrouter_model_name(model_name)
    normalized_provider = KNOWN_PROVIDERS.get(provider_raw, provider_raw)

    caps = OpenRouterModelCapabilities(
        provider=normalized_provider,
        base_model=base_model,
    )

    # Check vision support
    if normalized_provider in VISION_MODEL_PATTERNS:
        for pattern in VISION_MODEL_PATTERNS[normalized_provider]:
            if pattern in base_model:
                caps.supports_vision = True
                break

    # Set image placement preference based on provider
    if normalized_provider in ("anthropic", "google"):
        caps.images_before_text = True

    # Check reasoning model support
    if normalized_provider in REASONING_MODEL_PATTERNS:
        for pattern in REASONING_MODEL_PATTERNS[normalized_provider]:
            if pattern in base_model:
                caps.is_reasoning_model = True
                caps.uses_max_completion_tokens = True
                caps.supports_reasoning_effort = True
                caps.default_max_tokens = 4000
                break

    # Check system message support
    if normalized_provider in NO_SYSTEM_MESSAGE_PATTERNS:
        for pattern in NO_SYSTEM_MESSAGE_PATTERNS[normalized_provider]:
            if pattern in base_model:
                caps.supports_system_messages = False
                break

    # Check temperature support
    if normalized_provider in NO_TEMPERATURE_PATTERNS:
        for pattern in NO_TEMPERATURE_PATTERNS[normalized_provider]:
            if pattern in base_model:
                caps.supports_temperature = False
                break

    if DEBUG_OPENROUTER:
        logger.info(f"Detected capabilities for {model_name}: {caps}")

    return caps


# =============================================================================
# Helper Functions
# =============================================================================

def _process_messages_for_model(
    messages: List[Dict[str, Any]],
    caps: OpenRouterModelCapabilities,
) -> List[Dict[str, Any]]:
    """
    Process messages to comply with model constraints.

    For models that don't support system messages (like o1-mini, o1-preview),
    this function converts system messages to user messages with a clear prefix.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        caps: OpenRouterModelCapabilities instance for the target model.

    Returns:
        List[Dict[str, Any]]: Processed messages compatible with the model.

    Example:
        >>> caps = detect_model_capabilities("openai/o1-mini")
        >>> messages = [
        ...     {"role": "system", "content": "Be helpful"},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> processed = _process_messages_for_model(messages, caps)
        >>> processed[0]["role"]
        'user'
        >>> "[System Instructions]" in processed[0]["content"]
        True
    """
    if caps.supports_system_messages:
        return messages

    processed = []
    for msg in messages:
        if msg.get("role") == "system":
            # Convert system message to user message with prefix
            processed.append({
                "role": "user",
                "content": f"[System Instructions]: {msg['content']}"
            })
            if DEBUG_OPENROUTER:
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
    caps: OpenRouterModelCapabilities,
) -> Dict[str, Any]:
    """
    Build the API parameters dictionary based on model capabilities.

    This function creates the appropriate parameters for the OpenRouter API call,
    automatically adjusting for model-specific requirements.

    Args:
        model_name: The model identifier.
        messages: List of message dictionaries.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        json_mode: Whether to force JSON output format.
        reasoning_effort: Reasoning depth for thinking models.
        caps: OpenRouterModelCapabilities instance for the target model.

    Returns:
        Dict[str, Any]: Parameters ready for the OpenRouter API call.
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
            if DEBUG_OPENROUTER:
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
        if DEBUG_OPENROUTER and temperature != 1.0:
            logger.info(f"Temperature parameter ignored for {model_name} (only supports temperature=1)")

    # Handle reasoning effort for thinking models
    if reasoning_effort and caps.supports_reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    elif reasoning_effort and not caps.supports_reasoning_effort:
        if DEBUG_OPENROUTER:
            logger.warning(f"reasoning_effort ignored for non-reasoning model {model_name}")

    # Handle JSON mode
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    return params


def _extract_reasoning_tokens(completion) -> Optional[int]:
    """
    Extract reasoning tokens from the API response.

    Args:
        completion: The OpenRouter API completion response.

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

    if DEBUG_OPENROUTER:
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
    site_url: Optional[str] = None,
    site_name: Optional[str] = None,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response using any model through OpenRouter.

    Supports all OpenRouter-available models including OpenAI, Anthropic,
    Google, Meta, and many others. Automatically detects model capabilities
    and applies appropriate parameters.

    Args:
        model_name: The OpenRouter model to use (e.g., 'openai/gpt-4o',
            'anthropic/claude-3.5-sonnet', 'google/gemini-2.0-flash').
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
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        json_mode: If True, forces JSON output format. Default: False
        reasoning_effort: For reasoning models, controls thinking depth.
            Options: "low", "medium", "high". Default: None (model default)
        timeout: Request timeout in seconds. Default: None (no timeout)
        site_url: Your site URL for OpenRouter tracking. Falls back to
            OPENROUTER_SITE_URL env var.
        site_name: Your site name for OpenRouter tracking. Falls back to
            OPENROUTER_SITE_NAME env var.

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
        ...     model_name="openai/gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )

        >>> # Using a different provider
        >>> response = generate_response(
        ...     model_name="anthropic/claude-3.5-sonnet",
        ...     messages=[{"role": "user", "content": "Explain quantum computing"}],
        ...     max_tokens=500
        ... )

        >>> # Reasoning model with full response
        >>> response = generate_response(
        ...     model_name="openai/o1-mini",
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

    if DEBUG_OPENROUTER and caps.is_reasoning_model:
        logger.info(f"Detected reasoning model: {model_name}")

    # Initialize client with optional timeout
    client_kwargs = {
        "api_key": api_key,
        "base_url": OPENROUTER_BASE_URL,
    }
    if timeout:
        client_kwargs["timeout"] = timeout
    openrouter_client = OpenAI(**client_kwargs)

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

    # Add OpenRouter-specific headers for site tracking
    extra_headers = {}
    final_site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "")
    final_site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "")
    if final_site_url:
        extra_headers["HTTP-Referer"] = final_site_url
    if final_site_name:
        extra_headers["X-Title"] = final_site_name

    for attempt in range(MAX_RETRIES):
        try:
            completion = openrouter_client.chat.completions.create(
                extra_headers=extra_headers if extra_headers else None,
                **params
            )

            # Extract text from response
            generated_text = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason

            # Extract reasoning tokens if available
            reasoning_tokens = _extract_reasoning_tokens(completion)

            if DEBUG_OPENROUTER and reasoning_tokens:
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
                if DEBUG_OPENROUTER:
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise

        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_OPENROUTER:
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
    site_url: Optional[str] = None,
    site_name: Optional[str] = None,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response using any model through OpenRouter.

    This is the async version of generate_response(). It supports all
    OpenRouter-available models including OpenAI, Anthropic, Google, Meta,
    and many others. Automatically detects model capabilities and applies
    appropriate parameters.

    Args:
        model_name: The OpenRouter model to use (e.g., 'openai/gpt-4o',
            'anthropic/claude-3.5-sonnet', 'google/gemini-2.0-flash').
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
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        json_mode: If True, forces JSON output format. Default: False
        reasoning_effort: For reasoning models, controls thinking depth.
            Options: "low", "medium", "high". Default: None (model default)
        timeout: Request timeout in seconds. Default: None (no timeout)
        site_url: Your site URL for OpenRouter tracking. Falls back to
            OPENROUTER_SITE_URL env var.
        site_name: Your site name for OpenRouter tracking. Falls back to
            OPENROUTER_SITE_NAME env var.

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
        ...     model_name="openai/gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )

        >>> # With asyncio
        >>> import asyncio
        >>> response = asyncio.run(generate_response_async(
        ...     model_name="anthropic/claude-3.5-sonnet",
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

    if DEBUG_OPENROUTER and caps.is_reasoning_model:
        logger.info(f"Detected reasoning model: {model_name}")

    # Initialize async client with optional timeout
    client_kwargs = {
        "api_key": api_key,
        "base_url": OPENROUTER_BASE_URL,
    }
    if timeout:
        client_kwargs["timeout"] = timeout
    async_openrouter_client = AsyncOpenAI(**client_kwargs)

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

    # Add OpenRouter-specific headers for site tracking
    extra_headers = {}
    final_site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "")
    final_site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "")
    if final_site_url:
        extra_headers["HTTP-Referer"] = final_site_url
    if final_site_name:
        extra_headers["X-Title"] = final_site_name

    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openrouter_client.chat.completions.create(
                extra_headers=extra_headers if extra_headers else None,
                **params
            )

            # Extract text from response
            generated_text = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason

            # Extract reasoning tokens if available
            reasoning_tokens = _extract_reasoning_tokens(completion)

            if DEBUG_OPENROUTER and reasoning_tokens:
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
                if DEBUG_OPENROUTER:
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                raise

        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_OPENROUTER:
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
    Generate embeddings using an OpenRouter-available embedding model.

    Args:
        model_name: The embedding model to use (e.g., 'openai/text-embedding-3-small')
        user_input: Text or list of texts to embed
        full_response: If True, returns LLMEmbeddingsResponse with metadata
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.

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

    openrouter_client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = openrouter_client.embeddings.create(
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
    Asynchronously generate embeddings using an OpenRouter-available embedding model.

    Args:
        model_name: The embedding model to use (e.g., 'openai/text-embedding-3-small')
        user_input: Text or list of texts to embed
        full_response: If True, returns LLMEmbeddingsResponse with metadata
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.

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

    async_openrouter_client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = await async_openrouter_client.embeddings.create(
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
