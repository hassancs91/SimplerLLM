"""
Moonshot LLM Provider - Production-ready interface for Moonshot AI models.

This module provides low-level API functions for interacting with Moonshot AI's
OpenAI-compatible chat completions API. It supports all Moonshot/Kimi models including
standard and thinking/reasoning models.

Model Categories:
    V1 Models (Text):
        - moonshot-v1-8k: 8K context window
        - moonshot-v1-32k: 32K context window
        - moonshot-v1-128k: 128K context window

    Vision Models:
        - moonshot-v1-8k-vision-preview
        - moonshot-v1-32k-vision-preview
        - moonshot-v1-128k-vision-preview

    K2 Models (128K context):
        - kimi-k2-0711-preview: K2 July preview
        - kimi-k2-0905-preview: K2 September preview
        - kimi-k2-thinking: Reasoning model with extended thinking

    Latest Models:
        - kimi-latest-8k: Latest 8K context
        - kimi-latest-32k: Latest 32K context
        - kimi-latest-128k: Latest 128K context

Kimi-Specific Constraints:
    - Temperature range is [0, 1], not [0, 2] like OpenAI
    - Thinking models return reasoning_content in the response
    - Use extra_body={'thinking': {'type': 'disabled'}} to disable thinking

Environment Variables:
    MOONSHOT_API_KEY: API key for authentication
    MAX_RETRIES: Number of retry attempts (default: 3)
    RETRY_DELAY: Base delay between retries in seconds (default: 2)
    DEBUG_MOONSHOT: Enable debug output for Moonshot models (default: false)

Example:
    >>> from SimplerLLM.language.llm_providers import moonshot_llm
    >>>
    >>> # Standard model
    >>> response = moonshot_llm.generate_response(
    ...     model_name="kimi-k2-0905-preview",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
    >>>
    >>> # Thinking model with full response
    >>> response = moonshot_llm.generate_response(
    ...     model_name="kimi-k2-thinking",
    ...     messages=[{"role": "user", "content": "Solve: What is 15! / 13!?"}],
    ...     max_tokens=8000,
    ...     full_response=True
    ... )
    >>> print(f"Thinking: {response.thinking_content}")
    >>> print(f"Answer: {response.generated_text}")
"""

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import asyncio
import logging
import os
import time
from .llm_response_models import LLMFullResponse

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
DEBUG_MOONSHOT = os.getenv("DEBUG_MOONSHOT", "false").lower() == "true"

# =============================================================================
# API Configuration
# =============================================================================

KIMI_API_BASE_URL = "https://api.moonshot.ai/v1"

# Temperature constraints (Kimi uses 0-1 range, not 0-2 like OpenAI)
KIMI_MAX_TEMPERATURE = 1.0
KIMI_RECOMMENDED_TEMPERATURE = 0.6
KIMI_THINKING_TEMPERATURE = 1.0

# =============================================================================
# Model Detection and Configuration
# =============================================================================

# Patterns to identify thinking/reasoning models
THINKING_MODEL_PATTERNS = ["k2-thinking", "k2.5-thinking"]

# Patterns to identify vision-capable models
VISION_MODEL_PATTERNS = ["vision", "k2.5"]

# Patterns to identify extended context models
EXTENDED_CONTEXT_PATTERNS = ["128k", "k2", "k2.5"]


@dataclass
class KimiModelCapabilities:
    """
    Capabilities and constraints for a Kimi/Moonshot model.

    This dataclass encapsulates all the feature flags and constraints
    for a specific model, enabling automatic parameter adjustment.

    Attributes:
        is_thinking_model: Whether the model supports extended thinking/reasoning.
        supports_vision: Whether the model can process images.
        supports_thinking_control: Whether thinking can be enabled/disabled.
        supports_json_mode: Whether the model supports JSON output format.
        max_context: Maximum context window size in tokens.
        default_max_tokens: Default token limit for this model type.
        recommended_max_tokens: Recommended token limit for thinking models.
        recommended_temperature: Recommended temperature for this model type.
    """
    is_thinking_model: bool = False
    supports_vision: bool = False
    supports_thinking_control: bool = False
    supports_json_mode: bool = True
    max_context: int = 8192
    default_max_tokens: int = 300
    recommended_max_tokens: int = 8000
    recommended_temperature: float = 0.6


def detect_model_capabilities(model_name: str) -> KimiModelCapabilities:
    """
    Detect the capabilities and constraints of a Kimi/Moonshot model.

    This function analyzes the model name to determine its capabilities,
    constraints, and the appropriate parameters to use when calling the API.

    Args:
        model_name: The model identifier (e.g., 'kimi-k2-instruct', 'kimi-k2-thinking').

    Returns:
        KimiModelCapabilities: A dataclass containing all model capabilities and
            constraints for automatic parameter adjustment.

    Example:
        >>> caps = detect_model_capabilities('kimi-k2-thinking')
        >>> caps.is_thinking_model
        True
        >>> caps.recommended_temperature
        1.0

        >>> caps = detect_model_capabilities('kimi-k2-instruct')
        >>> caps.is_thinking_model
        False
        >>> caps.recommended_temperature
        0.6
    """
    model_lower = model_name.lower()
    caps = KimiModelCapabilities()

    # Check if it's a thinking/reasoning model
    for pattern in THINKING_MODEL_PATTERNS:
        if pattern in model_lower:
            caps.is_thinking_model = True
            caps.supports_thinking_control = True
            caps.default_max_tokens = 8000
            caps.recommended_max_tokens = 16000
            caps.recommended_temperature = KIMI_THINKING_TEMPERATURE
            break

    # Check vision support
    for pattern in VISION_MODEL_PATTERNS:
        if pattern in model_lower:
            caps.supports_vision = True
            break

    # Determine context window size
    if "k2.5" in model_lower:
        caps.max_context = 262144  # 256K
    elif "128k" in model_lower or "k2" in model_lower:
        caps.max_context = 131072  # 128K
    elif "32k" in model_lower:
        caps.max_context = 32768  # 32K
    else:
        caps.max_context = 8192  # Default 8K

    return caps


def _clamp_temperature(temperature: float, caps: KimiModelCapabilities) -> float:
    """
    Clamp temperature to Kimi's valid range (0-1).

    Kimi API only accepts temperature in range [0, 1], unlike OpenAI's [0, 2].
    This function automatically clamps values and logs a debug message.

    Args:
        temperature: The requested temperature value.
        caps: Model capabilities (unused but kept for consistency).

    Returns:
        float: Temperature clamped to [0, 1] range.
    """
    if temperature > KIMI_MAX_TEMPERATURE:
        if DEBUG_MOONSHOT:
            logger.info(f"Clamping temperature {temperature} to {KIMI_MAX_TEMPERATURE} (Kimi max)")
        return KIMI_MAX_TEMPERATURE
    return max(0.0, temperature)


def _build_api_params(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    json_mode: bool,
    thinking: Optional[bool],
    caps: KimiModelCapabilities,
) -> Dict[str, Any]:
    """
    Build the API parameters dictionary based on model capabilities.

    This function creates the appropriate parameters for the Kimi API call,
    automatically adjusting for model-specific requirements.

    Args:
        model_name: The model identifier.
        messages: List of message dictionaries.
        temperature: Sampling temperature (will be clamped to 0-1).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        json_mode: Whether to force JSON output format.
        thinking: Enable/disable thinking for thinking models. None uses default.
        caps: KimiModelCapabilities instance for the target model.

    Returns:
        Dict[str, Any]: Parameters ready for the Kimi API call.
    """
    # Clamp temperature to Kimi's valid range
    clamped_temp = _clamp_temperature(temperature, caps)

    params = {
        "model": model_name,
        "messages": messages,
        "temperature": clamped_temp,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    # Handle thinking mode for thinking models
    if thinking is not None and caps.supports_thinking_control:
        if not thinking:
            # Disable thinking via extra_body
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        if DEBUG_MOONSHOT:
            logger.info(f"Thinking mode {'disabled' if not thinking else 'enabled'} for {model_name}")
    elif thinking is not None and not caps.supports_thinking_control:
        if DEBUG_MOONSHOT:
            logger.warning(f"thinking parameter ignored for non-thinking model {model_name}")

    # Handle JSON mode
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    return params


def _extract_reasoning_tokens(completion) -> Optional[int]:
    """
    Extract reasoning tokens from the API response.

    Args:
        completion: The Kimi API completion response.

    Returns:
        Optional[int]: Number of reasoning tokens used, or None if not available.
    """
    if hasattr(completion, 'usage') and completion.usage:
        if hasattr(completion.usage, 'completion_tokens_details'):
            details = completion.usage.completion_tokens_details
            if details and hasattr(details, 'reasoning_tokens'):
                return details.reasoning_tokens
    return None


def _extract_response_data(
    completion,
    caps: KimiModelCapabilities,
) -> Dict[str, Any]:
    """
    Extract response data including reasoning_content from thinking models.

    For K2-Thinking and K2.5-Thinking models, the response includes:
    - content: The final answer
    - reasoning_content: The thinking/reasoning process

    Args:
        completion: The Kimi API completion response.
        caps: Model capabilities for context.

    Returns:
        Dict containing extracted response data.
    """
    choice = completion.choices[0]
    message = choice.message

    generated_text = message.content or ""
    reasoning_content = getattr(message, 'reasoning_content', None)
    finish_reason = choice.finish_reason

    # Extract usage data
    usage = completion.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    # Extract reasoning tokens if available
    reasoning_tokens = _extract_reasoning_tokens(completion)

    return {
        "generated_text": generated_text,
        "reasoning_content": reasoning_content,
        "reasoning_tokens": reasoning_tokens,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def generate_response(
    model_name: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.6,
    max_tokens: int = 300,
    top_p: float = 0.95,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    thinking: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response using Kimi's OpenAI-compatible API.

    Uses OpenAI SDK with custom base_url pointing to Kimi's API endpoint.
    Supports all Kimi models including thinking/reasoning models.

    Args:
        model_name: The Kimi model to use (e.g., 'kimi-k2-instruct', 'kimi-k2-thinking').
        messages: List of message dicts with 'role' and 'content' keys.
            For vision, content can be a list of text/image_url objects.
        temperature: Sampling temperature (0-1). Values > 1 are clamped.
            Recommended: 0.6 for standard, 1.0 for thinking models. Default: 0.6
        max_tokens: Maximum tokens to generate. Default: 300 (8000+ recommended
            for thinking models).
        top_p: Nucleus sampling parameter. Recommended: 0.95. Default: 0.95
        full_response: If True, returns LLMFullResponse with metadata.
            If False, returns just the generated text. Default: False
        api_key: Moonshot API key. Falls back to MOONSHOT_API_KEY env var.
        json_mode: If True, forces JSON output format. Default: False
        thinking: For thinking models, explicitly enable/disable thinking.
            None uses model default (thinking enabled). Default: None
        timeout: Request timeout in seconds. Default: None (no timeout)

    Returns:
        str: Generated text if full_response=False
        LLMFullResponse: Full response object if full_response=True, including:
            - generated_text: The model's response
            - model: Model name used
            - process_time: Time taken in seconds
            - input_token_count: Prompt tokens used
            - output_token_count: Completion tokens used
            - thinking_content: The reasoning process (thinking models only)
            - reasoning_tokens: Tokens used for reasoning (if available)
            - finish_reason: Why generation stopped
            - is_reasoning_model: True if thinking model was used

    Raises:
        ValueError: If messages is None or empty
        openai.RateLimitError: On rate limit errors (after retries exhausted)
        openai.APIError: On API errors (after retries exhausted)
        Exception: On other errors after retries exhausted

    Example:
        >>> # Standard model usage
        >>> response = generate_response(
        ...     model_name="kimi-k2-instruct",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )

        >>> # Thinking model with full response
        >>> response = generate_response(
        ...     model_name="kimi-k2-thinking",
        ...     messages=[{"role": "user", "content": "Solve: What is 15! / 13!?"}],
        ...     max_tokens=8000,
        ...     full_response=True
        ... )
        >>> print(f"Thinking: {response.thinking_content}")
        >>> print(f"Answer: {response.generated_text}")
    """
    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    start_time = time.time() if full_response else None

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    if DEBUG_MOONSHOT:
        logger.info(f"Using Kimi model: {model_name}")
        if caps.is_thinking_model:
            logger.info(f"Detected thinking model: {model_name}")

    # Initialize OpenAI client with Kimi base URL
    api_key = api_key or os.getenv("MOONSHOT_API_KEY", "")
    if not api_key:
        raise ValueError("MOONSHOT_API_KEY not found. Set it in environment or pass api_key parameter.")

    client_kwargs = {
        "api_key": api_key,
        "base_url": KIMI_API_BASE_URL
    }
    if timeout:
        client_kwargs["timeout"] = timeout

    client = OpenAI(**client_kwargs)

    # Build API parameters
    params = _build_api_params(
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        thinking=thinking,
        caps=caps,
    )

    # Retry loop with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(**params)

            # Extract response data
            extracted = _extract_response_data(completion, caps)

            if full_response:
                process_time = time.time() - start_time
                return LLMFullResponse(
                    generated_text=extracted["generated_text"],
                    model=model_name,
                    process_time=process_time,
                    input_token_count=extracted["input_tokens"],
                    output_token_count=extracted["output_tokens"],
                    llm_provider_response=completion,
                    reasoning_tokens=extracted["reasoning_tokens"],
                    thinking_content=extracted["reasoning_content"],
                    finish_reason=extracted["finish_reason"],
                    is_reasoning_model=caps.is_thinking_model,
                )
            return extracted["generated_text"]

        except RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_MOONSHOT:
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise

        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_MOONSHOT:
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
    temperature: float = 0.6,
    max_tokens: int = 300,
    top_p: float = 0.95,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    thinking: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response using Kimi's OpenAI-compatible API.

    This is the async version of generate_response(). It supports all Kimi
    models including thinking/reasoning models.

    Args:
        model_name: The Kimi model to use (e.g., 'kimi-k2-instruct', 'kimi-k2-thinking').
        messages: List of message dicts with 'role' and 'content' keys.
        temperature: Sampling temperature (0-1). Values > 1 are clamped. Default: 0.6
        max_tokens: Maximum tokens to generate. Default: 300
        top_p: Nucleus sampling parameter. Default: 0.95
        full_response: If True, returns LLMFullResponse with metadata. Default: False
        api_key: Moonshot API key. Falls back to MOONSHOT_API_KEY env var.
        json_mode: If True, forces JSON output format. Default: False
        thinking: For thinking models, explicitly enable/disable thinking. Default: None
        timeout: Request timeout in seconds. Default: None

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
        ...     model_name="kimi-k2-instruct",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )

    See Also:
        generate_response: Synchronous version of this function.
    """
    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    start_time = time.time() if full_response else None

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    if DEBUG_MOONSHOT:
        logger.info(f"Using Kimi model (async): {model_name}")
        if caps.is_thinking_model:
            logger.info(f"Detected thinking model: {model_name}")

    # Initialize AsyncOpenAI client with Kimi base URL
    api_key = api_key or os.getenv("MOONSHOT_API_KEY", "")
    if not api_key:
        raise ValueError("MOONSHOT_API_KEY not found. Set it in environment or pass api_key parameter.")

    client_kwargs = {
        "api_key": api_key,
        "base_url": KIMI_API_BASE_URL
    }
    if timeout:
        client_kwargs["timeout"] = timeout

    async_client = AsyncOpenAI(**client_kwargs)

    # Build API parameters
    params = _build_api_params(
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        thinking=thinking,
        caps=caps,
    )

    # Retry loop with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_client.chat.completions.create(**params)

            # Extract response data
            extracted = _extract_response_data(completion, caps)

            if full_response:
                process_time = time.time() - start_time
                return LLMFullResponse(
                    generated_text=extracted["generated_text"],
                    model=model_name,
                    process_time=process_time,
                    input_token_count=extracted["input_tokens"],
                    output_token_count=extracted["output_tokens"],
                    llm_provider_response=completion,
                    reasoning_tokens=extracted["reasoning_tokens"],
                    thinking_content=extracted["reasoning_content"],
                    finish_reason=extracted["finish_reason"],
                    is_reasoning_model=caps.is_thinking_model,
                )
            return extracted["generated_text"]

        except RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_MOONSHOT:
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                raise

        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_MOONSHOT:
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
