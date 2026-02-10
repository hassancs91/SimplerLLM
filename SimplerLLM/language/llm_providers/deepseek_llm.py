"""
DeepSeek LLM Provider - Production-ready interface for DeepSeek models.

This module provides low-level API functions for interacting with DeepSeek's
chat completions API. It supports both standard and reasoning models.

Model Categories:
    Standard Models:
        - deepseek-chat: Latest DeepSeek V3.2 model for general chat

    Reasoning Models:
        - deepseek-reasoner: DeepSeek R1-based reasoning model with thinking capabilities

Reasoning Model Features:
    The deepseek-reasoner model provides chain-of-thought reasoning:
    - Returns reasoning_content with the thinking process
    - Tracks reasoning_tokens in usage statistics
    - Supports thinking parameter to enable/disable thinking mode

Environment Variables:
    DEEPSEEK_API_KEY: API key for authentication
    MAX_RETRIES: Number of retry attempts (default: 3)
    RETRY_DELAY: Base delay between retries in seconds (default: 2)
    DEBUG_DEEPSEEK: Enable debug output for DeepSeek models (default: false)

Example:
    >>> from SimplerLLM.language.llm_providers import deepseek_llm
    >>>
    >>> # Standard model
    >>> response = deepseek_llm.generate_response(
    ...     model_name="deepseek-chat",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
    >>>
    >>> # Reasoning model with full response
    >>> response = deepseek_llm.generate_response(
    ...     model_name="deepseek-reasoner",
    ...     messages=[{"role": "user", "content": "Solve this puzzle..."}],
    ...     max_tokens=8000,
    ...     full_response=True
    ... )
    >>> print(f"Reasoning tokens: {response.reasoning_tokens}")
"""

from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import asyncio
import logging
import os
import time
import requests
import aiohttp
from .llm_response_models import LLMFullResponse

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
DEBUG_DEEPSEEK = os.getenv("DEBUG_DEEPSEEK", "false").lower() == "true"

# API endpoint
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# =============================================================================
# Model Detection and Configuration
# =============================================================================

# Patterns to identify reasoning models
REASONING_MODEL_PATTERNS = ["deepseek-reasoner", "r1"]


@dataclass
class DeepSeekModelCapabilities:
    """
    Capabilities and constraints for a DeepSeek model.

    Attributes:
        is_reasoning_model: Whether the model supports chain-of-thought reasoning.
        supports_thinking: Whether the thinking parameter can be used.
        default_max_tokens: Default token limit for this model type.
        recommended_max_tokens: Recommended token limit for reasoning models.
    """
    is_reasoning_model: bool = False
    supports_thinking: bool = False
    default_max_tokens: int = 300
    recommended_max_tokens: int = 8000


def detect_model_capabilities(model_name: str) -> DeepSeekModelCapabilities:
    """
    Detect the capabilities and constraints of a DeepSeek model.

    Args:
        model_name: The model identifier (e.g., 'deepseek-chat', 'deepseek-reasoner').

    Returns:
        DeepSeekModelCapabilities: A dataclass containing all model capabilities.

    Example:
        >>> caps = detect_model_capabilities('deepseek-reasoner')
        >>> caps.is_reasoning_model
        True
        >>> caps.supports_thinking
        True

        >>> caps = detect_model_capabilities('deepseek-chat')
        >>> caps.is_reasoning_model
        False
    """
    model_lower = model_name.lower()
    caps = DeepSeekModelCapabilities()

    # Check if it's a reasoning model
    for pattern in REASONING_MODEL_PATTERNS:
        if pattern in model_lower:
            caps.is_reasoning_model = True
            caps.supports_thinking = True
            caps.default_max_tokens = 8000
            break

    return caps


def _build_api_params(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    json_mode: bool,
    thinking: Optional[bool],
    caps: DeepSeekModelCapabilities,
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
        thinking: Whether to enable thinking mode for reasoner.
        caps: DeepSeekModelCapabilities instance for the target model.

    Returns:
        Dict[str, Any]: Parameters ready for the DeepSeek API call.
    """
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,
    }

    # Handle thinking mode for reasoning models
    if thinking is not None and caps.supports_thinking:
        params["thinking"] = {"type": "enabled" if thinking else "disabled"}
    elif thinking is not None and not caps.supports_thinking:
        if DEBUG_DEEPSEEK:
            logger.warning(f"thinking parameter ignored for non-reasoning model {model_name}")

    # Handle JSON mode
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    return params


def _extract_response_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract response data including reasoning content from the API response.

    Args:
        result: The raw API response dictionary.

    Returns:
        Dict containing generated_text, reasoning_content, reasoning_tokens, finish_reason.
    """
    choice = result["choices"][0]
    message = choice.get("message", {})

    generated_text = message.get("content", "")
    reasoning_content = message.get("reasoning_content")
    finish_reason = choice.get("finish_reason")

    # Extract usage data
    usage = result.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    reasoning_tokens = usage.get("reasoning_tokens")

    # If reasoning_tokens not in root usage, check completion_tokens_details
    if reasoning_tokens is None:
        details = usage.get("completion_tokens_details", {})
        if details:
            reasoning_tokens = details.get("reasoning_tokens")

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
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    thinking: Optional[bool] = None,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response using DeepSeek's chat completions API.

    Supports both standard (deepseek-chat) and reasoning (deepseek-reasoner) models.
    Automatically detects model capabilities and applies appropriate parameters.

    Args:
        model_name: The DeepSeek model to use (e.g., 'deepseek-chat', 'deepseek-reasoner').
        messages: List of message dicts with 'role' and 'content' keys.
        temperature: Sampling temperature (0-2). Default: 0.7
        max_tokens: Maximum tokens to generate. For reasoning models, this
            includes both reasoning and output tokens. Default: 300
        top_p: Nucleus sampling parameter. Default: 1.0
        full_response: If True, returns LLMFullResponse with metadata.
            If False, returns just the generated text. Default: False
        api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
        json_mode: If True, forces JSON output format. Default: False
        thinking: For deepseek-reasoner, enable (True) or disable (False)
            thinking mode. Default: None (model default)

    Returns:
        Union[str, LLMFullResponse]: The generated text, or full response object
            with metadata if full_response=True.

    Raises:
        Exception: If the API call fails after all retries.

    Example:
        >>> # Basic usage
        >>> text = generate_response(
        ...     model_name="deepseek-chat",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(text)

        >>> # Reasoning model with full response
        >>> response = generate_response(
        ...     model_name="deepseek-reasoner",
        ...     messages=[{"role": "user", "content": "What is 15! / 13!?"}],
        ...     max_tokens=8000,
        ...     full_response=True
        ... )
        >>> print(f"Answer: {response.generated_text}")
        >>> print(f"Reasoning tokens: {response.reasoning_tokens}")
    """
    start_time = time.time() if full_response else None

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    if DEBUG_DEEPSEEK:
        logger.info(f"Model: {model_name}, Reasoning: {caps.is_reasoning_model}")

    # Build headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Build request data
    data = _build_api_params(
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        thinking=thinking,
        caps=caps,
    )

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=data,
                timeout=300  # 5 minute timeout for reasoning models
            )
            response.raise_for_status()
            result = response.json()

            # Extract response data
            extracted = _extract_response_data(result)

            if full_response:
                process_time = time.time() - start_time
                return LLMFullResponse(
                    generated_text=extracted["generated_text"],
                    model=model_name,
                    process_time=process_time,
                    input_token_count=extracted["input_tokens"],
                    output_token_count=extracted["output_tokens"],
                    llm_provider_response=result,
                    reasoning_tokens=extracted["reasoning_tokens"],
                    thinking_content=extracted["reasoning_content"],
                    finish_reason=extracted["finish_reason"],
                    is_reasoning_model=caps.is_reasoning_model,
                )
            return extracted["generated_text"]

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_body = response.json()
                error_detail = f" - {error_body.get('error', {}).get('message', '')}"
            except:
                pass

            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_DEEPSEEK:
                    logger.warning(f"HTTP error (attempt {attempt + 1}/{MAX_RETRIES}): {e}{error_detail}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                error_msg = f"DeepSeek API error after {MAX_RETRIES} attempts: {e}{error_detail}"
                raise Exception(error_msg)

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_DEEPSEEK:
                    logger.warning(f"Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
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
    thinking: Optional[bool] = None,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response using DeepSeek's chat completions API.

    This is the async version of generate_response(). See generate_response()
    for full parameter documentation.

    Args:
        model_name: The DeepSeek model to use.
        messages: List of message dicts with 'role' and 'content' keys.
        temperature: Sampling temperature (0-2). Default: 0.7
        max_tokens: Maximum tokens to generate. Default: 300
        top_p: Nucleus sampling parameter. Default: 1.0
        full_response: If True, returns LLMFullResponse. Default: False
        api_key: DeepSeek API key.
        json_mode: If True, forces JSON output format. Default: False
        thinking: For deepseek-reasoner, enable/disable thinking mode.

    Returns:
        Union[str, LLMFullResponse]: The generated text, or full response object.

    Raises:
        Exception: If the API call fails after all retries.
    """
    start_time = time.time() if full_response else None

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    if DEBUG_DEEPSEEK:
        logger.info(f"Model: {model_name}, Reasoning: {caps.is_reasoning_model} (async)")

    # Build headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Build request data
    data = _build_api_params(
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        thinking=thinking,
        caps=caps,
    )

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    DEEPSEEK_API_URL,
                    headers=headers,
                    json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    # Extract response data
                    extracted = _extract_response_data(result, caps)

                    if full_response:
                        process_time = time.time() - start_time
                        return LLMFullResponse(
                            generated_text=extracted["generated_text"],
                            model=model_name,
                            process_time=process_time,
                            input_token_count=extracted["input_tokens"],
                            output_token_count=extracted["output_tokens"],
                            llm_provider_response=result,
                            reasoning_tokens=extracted["reasoning_tokens"],
                            thinking_content=extracted["reasoning_content"],
                            finish_reason=extracted["finish_reason"],
                            is_reasoning_model=caps.is_reasoning_model,
                        )
                    return extracted["generated_text"]

        except aiohttp.ClientResponseError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_DEEPSEEK:
                    logger.warning(f"HTTP error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                error_msg = f"DeepSeek API error after {MAX_RETRIES} attempts: {e}"
                raise Exception(error_msg)

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_DEEPSEEK:
                    logger.warning(f"Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
