"""
OpenAI LLM Provider - Production-ready interface for all OpenAI models.

This module provides low-level API functions for interacting with OpenAI's
chat completions, responses, and embeddings APIs. It supports all OpenAI
models including standard and reasoning models.

Model Categories:
    Standard Models:
        - GPT-4o, GPT-4o-mini: Latest multimodal models with vision support
        - GPT-4, GPT-4-turbo: High-capability text models
        - GPT-3.5-turbo: Fast, cost-effective model

    Reasoning Models (Extended Thinking):
        - o1, o1-mini, o1-preview: First-generation reasoning models
        - o3, o3-mini: Advanced reasoning models
        - GPT-5 series: Latest reasoning-capable models

Reasoning Model Constraints:
    Reasoning models have special requirements that this module handles automatically:
    - Use max_completion_tokens instead of max_tokens
    - Temperature must be 1 (some models don't support temperature parameter at all)
    - Some models (o1-mini, o1-preview) don't support system messages
    - Support reasoning_effort parameter (low, medium, high)
    - May use significant tokens for internal reasoning before generating output

Environment Variables:
    OPENAI_API_KEY: API key for authentication
    MAX_RETRIES: Number of retry attempts (default: 3)
    RETRY_DELAY: Base delay between retries in seconds (default: 2)
    DEBUG_REASONING: Enable debug output for reasoning models (default: false)

Example:
    >>> from SimplerLLM.language.llm_providers import openai_llm
    >>>
    >>> # Standard model
    >>> response = openai_llm.generate_response(
    ...     model_name="gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
    >>>
    >>> # Reasoning model with full response
    >>> response = openai_llm.generate_response(
    ...     model_name="o1-mini",
    ...     messages=[{"role": "user", "content": "Solve this puzzle..."}],
    ...     max_tokens=8000,
    ...     reasoning_effort="high",
    ...     full_response=True
    ... )
    >>> print(f"Reasoning tokens: {response.reasoning_tokens}")
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
DEBUG_REASONING = os.getenv("DEBUG_REASONING", "false").lower() == "true"

# Legacy alias for backward compatibility
DEBUG_GPT5 = DEBUG_REASONING

# =============================================================================
# Model Detection and Configuration
# =============================================================================

# Patterns to identify reasoning models
REASONING_MODEL_PATTERNS = ["o1", "o3", "gpt-5"]

# Models that don't support system messages (must use user message with prefix)
NO_SYSTEM_MESSAGE_MODELS = ["o1-mini", "o1-preview"]

# Models that don't support temperature parameter (all reasoning models)
NO_TEMPERATURE_MODELS = REASONING_MODEL_PATTERNS  # ["o1", "o3", "gpt-5"]

# Models with vision capabilities
VISION_CAPABLE_PATTERNS = ["gpt-4o", "gpt-4-vision", "gpt-5", "vision"]


@dataclass
class ModelCapabilities:
    """
    Capabilities and constraints for an OpenAI model.

    This dataclass encapsulates all the feature flags and constraints
    for a specific model, enabling automatic parameter adjustment.

    Attributes:
        is_reasoning_model: Whether the model supports extended thinking/reasoning.
        supports_system_messages: Whether the model accepts system role messages.
        supports_temperature: Whether the temperature parameter can be set.
        supports_vision: Whether the model can process images.
        supports_json_mode: Whether the model supports JSON output format.
        uses_max_completion_tokens: Whether to use max_completion_tokens vs max_tokens.
        supports_reasoning_effort: Whether reasoning_effort parameter is supported.
        default_max_tokens: Default token limit for this model type.
        recommended_max_tokens: Recommended token limit for reasoning models.
    """
    is_reasoning_model: bool = False
    supports_system_messages: bool = True
    supports_temperature: bool = True
    supports_vision: bool = False
    supports_json_mode: bool = True
    uses_max_completion_tokens: bool = False
    supports_reasoning_effort: bool = False
    default_max_tokens: int = 300
    recommended_max_tokens: int = 4000


def detect_model_capabilities(model_name: str) -> ModelCapabilities:
    """
    Detect the capabilities and constraints of an OpenAI model.

    This function analyzes the model name to determine its capabilities,
    constraints, and the appropriate parameters to use when calling the API.

    Args:
        model_name: The model identifier (e.g., 'gpt-4o', 'o1-mini', 'gpt-5').

    Returns:
        ModelCapabilities: A dataclass containing all model capabilities and
            constraints for automatic parameter adjustment.

    Example:
        >>> caps = detect_model_capabilities('o1-mini')
        >>> caps.is_reasoning_model
        True
        >>> caps.supports_system_messages
        False
        >>> caps.uses_max_completion_tokens
        True

        >>> caps = detect_model_capabilities('gpt-4o')
        >>> caps.is_reasoning_model
        False
        >>> caps.supports_vision
        True
    """
    model_lower = model_name.lower()
    caps = ModelCapabilities()

    # Check if it's a reasoning model
    for pattern in REASONING_MODEL_PATTERNS:
        if pattern in model_lower:
            caps.is_reasoning_model = True
            caps.uses_max_completion_tokens = True
            caps.supports_reasoning_effort = True
            caps.default_max_tokens = 4000
            break

    # Check system message support
    for pattern in NO_SYSTEM_MESSAGE_MODELS:
        if pattern in model_lower:
            caps.supports_system_messages = False
            break

    # Check temperature support
    for pattern in NO_TEMPERATURE_MODELS:
        if pattern in model_lower:
            caps.supports_temperature = False
            break

    # Check vision support
    for pattern in VISION_CAPABLE_PATTERNS:
        if pattern in model_lower:
            caps.supports_vision = True
            break

    return caps


def _process_messages_for_model(
    messages: List[Dict[str, Any]],
    caps: ModelCapabilities,
) -> List[Dict[str, Any]]:
    """
    Process messages to comply with model constraints.

    For models that don't support system messages (like o1-mini, o1-preview),
    this function converts system messages to user messages with a clear prefix.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        caps: ModelCapabilities instance for the target model.

    Returns:
        List[Dict[str, Any]]: Processed messages compatible with the model.

    Example:
        >>> caps = detect_model_capabilities("o1-mini")
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
            if DEBUG_REASONING:
                logger.info(f"Converted system message to user message for model compatibility")
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
    caps: ModelCapabilities,
) -> Dict[str, Any]:
    """
    Build the API parameters dictionary based on model capabilities.

    This function creates the appropriate parameters for the OpenAI API call,
    automatically adjusting for model-specific requirements.

    Args:
        model_name: The model identifier.
        messages: List of message dictionaries.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        json_mode: Whether to force JSON output format.
        reasoning_effort: Reasoning depth for thinking models.
        caps: ModelCapabilities instance for the target model.

    Returns:
        Dict[str, Any]: Parameters ready for the OpenAI API call.
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
            if DEBUG_REASONING:
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
        if DEBUG_REASONING and temperature != 1.0:
            logger.info(f"Temperature parameter ignored for {model_name} (only supports temperature=1)")

    # Handle reasoning effort for thinking models
    if reasoning_effort and caps.supports_reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    elif reasoning_effort and not caps.supports_reasoning_effort:
        if DEBUG_REASONING:
            logger.warning(f"reasoning_effort ignored for non-reasoning model {model_name}")

    # Handle JSON mode
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    return params


def _extract_reasoning_tokens(completion) -> Optional[int]:
    """
    Extract reasoning tokens from the API response.

    Args:
        completion: The OpenAI API completion response.

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
        completion: The OpenAI API completion response.
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

    if DEBUG_REASONING:
        logger.warning(msg)
        if reasoning_tokens:
            logger.info(f"Reasoning tokens used: {reasoning_tokens}")
            logger.info(f"Max tokens allowed: {max_tokens_used}")

    return msg


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
    Generate a response using OpenAI's chat completions API.

    Supports all OpenAI models including standard (GPT-4, GPT-4o) and
    reasoning models (o1, o3, GPT-5 series). Automatically detects model
    capabilities and applies appropriate parameters.

    Args:
        model_name: The OpenAI model to use (e.g., 'gpt-4o', 'o1-mini', 'gpt-5').
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
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
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
        ...     model_name="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )

        >>> # Reasoning model with full response
        >>> response = generate_response(
        ...     model_name="o1-mini",
        ...     messages=[{"role": "user", "content": "Solve: What is 15! / 13!?"}],
        ...     max_tokens=8000,
        ...     reasoning_effort="high",
        ...     full_response=True
        ... )
        >>> print(f"Reasoning tokens used: {response.reasoning_tokens}")

    Notes:
        - Reasoning models (o1, o3, GPT-5) automatically use max_completion_tokens
        - Temperature is ignored/set to 1 for some reasoning models
        - o1-mini and o1-preview don't support system messages - they're
          automatically converted to user messages with a prefix
        - Empty responses from reasoning models include diagnostic messages
    """
    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    start_time = time.time() if full_response else None

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    if DEBUG_REASONING and caps.is_reasoning_model:
        logger.info(f"Detected reasoning model: {model_name}")

    # Initialize client with optional timeout
    client_kwargs = {"api_key": api_key}
    if timeout:
        client_kwargs["timeout"] = timeout
    openai_client = OpenAI(**client_kwargs)

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
            completion = openai_client.chat.completions.create(**params)

            # Extract text from response
            generated_text = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason

            # Extract reasoning tokens if available
            reasoning_tokens = _extract_reasoning_tokens(completion)

            if DEBUG_REASONING and reasoning_tokens:
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
                if DEBUG_REASONING:
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise

        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_REASONING:
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
    Asynchronously generate a response using OpenAI's chat completions API.

    This is the async version of generate_response(). It supports all OpenAI
    models including standard (GPT-4, GPT-4o) and reasoning models (o1, o3,
    GPT-5 series). Automatically detects model capabilities and applies
    appropriate parameters.

    Args:
        model_name: The OpenAI model to use (e.g., 'gpt-4o', 'o1-mini', 'gpt-5').
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
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
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
        ...     model_name="gpt-4o",
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

    if DEBUG_REASONING and caps.is_reasoning_model:
        logger.info(f"Detected reasoning model: {model_name}")

    # Initialize client with optional timeout
    client_kwargs = {"api_key": api_key}
    if timeout:
        client_kwargs["timeout"] = timeout
    async_openai_client = AsyncOpenAI(**client_kwargs)

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
            completion = await async_openai_client.chat.completions.create(**params)

            # Extract text from response
            generated_text = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason

            # Extract reasoning tokens if available
            reasoning_tokens = _extract_reasoning_tokens(completion)

            if DEBUG_REASONING and reasoning_tokens:
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
                if DEBUG_REASONING:
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                raise

        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_REASONING:
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


def generate_response_with_web_search(
    model_name,
    input_text,
    max_tokens=300,
    full_response=False,
    api_key=None,
):
    """
    Generate a response using OpenAI's Responses API with web search enabled.

    Args:
        model_name: The model to use (e.g., 'gpt-4o', 'gpt-4o-mini')
        input_text: The input prompt/question
        max_tokens: Maximum tokens for the response
        full_response: If True, returns LLMFullResponse with web_sources
        api_key: OpenAI API key

    Returns:
        str or LLMFullResponse: Generated text or full response with web sources
    """
    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            response = openai_client.responses.create(
                model=model_name,
                input=input_text,
                tools=[{"type": "web_search"}],
            )

            # Extract text and citations from the response
            generated_text = ""
            web_sources = []

            for output_item in response.output:
                if output_item.type == "message":
                    for content_item in output_item.content:
                        if content_item.type == "output_text":
                            generated_text = content_item.text
                            # Extract URL citations from annotations
                            if hasattr(content_item, 'annotations') and content_item.annotations:
                                for annotation in content_item.annotations:
                                    if annotation.type == "url_citation":
                                        web_sources.append({
                                            "title": annotation.title if hasattr(annotation, 'title') else "",
                                            "url": annotation.url if hasattr(annotation, 'url') else "",
                                        })

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=response.usage.input_tokens if hasattr(response, 'usage') and response.usage else 0,
                    output_token_count=response.usage.output_tokens if hasattr(response, 'usage') and response.usage else 0,
                    llm_provider_response=response,
                    web_sources=web_sources if web_sources else None,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_response_with_web_search_async(
    model_name,
    input_text,
    max_tokens=300,
    full_response=False,
    api_key=None,
):
    """
    Asynchronously generate a response using OpenAI's Responses API with web search enabled.

    Args:
        model_name: The model to use (e.g., 'gpt-4o', 'gpt-4o-mini')
        input_text: The input prompt/question
        max_tokens: Maximum tokens for the response
        full_response: If True, returns LLMFullResponse with web_sources
        api_key: OpenAI API key

    Returns:
        str or LLMFullResponse: Generated text or full response with web sources
    """
    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            response = await async_openai_client.responses.create(
                model=model_name,
                input=input_text,
                tools=[{"type": "web_search"}],
            )

            # Extract text and citations from the response
            generated_text = ""
            web_sources = []

            for output_item in response.output:
                if output_item.type == "message":
                    for content_item in output_item.content:
                        if content_item.type == "output_text":
                            generated_text = content_item.text
                            # Extract URL citations from annotations
                            if hasattr(content_item, 'annotations') and content_item.annotations:
                                for annotation in content_item.annotations:
                                    if annotation.type == "url_citation":
                                        web_sources.append({
                                            "title": annotation.title if hasattr(annotation, 'title') else "",
                                            "url": annotation.url if hasattr(annotation, 'url') else "",
                                        })

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=response.usage.input_tokens if hasattr(response, 'usage') and response.usage else 0,
                    output_token_count=response.usage.output_tokens if hasattr(response, 'usage') and response.usage else 0,
                    llm_provider_response=response,
                    web_sources=web_sources if web_sources else None,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


def generate_embeddings(
    model_name,
    user_input=None,
    full_response = False,
    api_key = None
):
    
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None

    openai_client = OpenAI(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            
            response = openai_client.embeddings.create(
                model= model_name,
                input=user_input
            )
            
            # Extract actual embedding vectors from the response
            embeddings = [item.embedding for item in response.data]
            
            # For single input, return single embedding; for multiple inputs, return list
            if isinstance(user_input, str):
                generate_embeddings = embeddings[0] if embeddings else []
            else:
                generate_embeddings = embeddings

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=generate_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=response,
                )
            return generate_embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)

async def generate_embeddings_async(
    model_name,
    user_input=None,
    full_response = False,
    api_key = None,
):
    async_openai_client = AsyncOpenAI(api_key=api_key)
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None
    for attempt in range(MAX_RETRIES):
        try:
            result = await async_openai_client.embeddings.create(
                model=model_name,
                input=user_input,
            )
            
            # Extract actual embedding vectors from the response
            embeddings = [item.embedding for item in result.data]
            
            # For single input, return single embedding; for multiple inputs, return list
            if isinstance(user_input, str):
                generate_embeddings = embeddings[0] if embeddings else []
            else:
                generate_embeddings = embeddings

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=generate_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=result,
                )
            return generate_embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
