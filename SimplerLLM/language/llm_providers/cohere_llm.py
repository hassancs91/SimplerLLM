"""
Cohere LLM Provider - Production-ready implementation for Cohere V2 API.

This module provides low-level API functions for interacting with Cohere's
language models, including support for:

- All Command models (Command A, Command R+, Command R)
- Reasoning models (Command-A-Reasoning)
- Vision capabilities (Command-A-Vision, Aya Vision)
- JSON mode for structured outputs
- Embeddings

Model Categories:
    Standard Models:
        - command-a-03-2025: Latest flagship model (256K context)
        - command-r-plus-08-2024: Advanced retrieval-augmented model (128K context)
        - command-r-08-2024: Fast, cost-effective model (128K context)
        - command-r7b-12-2024: Small, fast model (128K context)

    Reasoning Models:
        - command-a-reasoning-08-2025: Extended thinking capabilities (32K max output)

    Vision Models:
        - command-a-vision-07-2025: Enterprise vision model
        - c4ai-aya-vision-8b, c4ai-aya-vision-32b: Multilingual vision

Example:
    >>> from SimplerLLM.language.llm_providers import cohere_llm
    >>>
    >>> # Basic generation
    >>> response = cohere_llm.generate_response(
    ...     model_name="command-a-03-2025",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
    >>>
    >>> # Vision with full response
    >>> response = cohere_llm.generate_response(
    ...     model_name="command-a-vision-07-2025",
    ...     messages=[{"role": "user", "content": [...]}],
    ...     max_tokens=500,
    ...     full_response=True
    ... )
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import logging
import os
import time

from .llm_response_models import LLMFullResponse, LLMEmbeddingsResponse

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Configuration constants
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
DEBUG_COHERE = os.getenv("DEBUG_COHERE", "false").lower() == "true"


# =============================================================================
# Model Detection and Configuration
# =============================================================================

# Patterns to identify model types
REASONING_MODEL_PATTERNS = ["reasoning"]
VISION_CAPABLE_PATTERNS = ["vision", "aya-vision"]

# Specific model lists
VISION_MODELS = [
    "command-a-vision-07-2025",
    "c4ai-aya-vision-8b",
    "c4ai-aya-vision-32b",
]

REASONING_MODELS = [
    "command-a-reasoning-08-2025",
]


@dataclass
class CohereModelCapabilities:
    """
    Capabilities and constraints for a Cohere model.

    Attributes:
        is_reasoning_model: Whether the model supports extended reasoning.
        supports_vision: Whether the model can process images.
        supports_json_mode: Whether the model supports JSON output format.
        supports_system_prompt: Whether the model accepts system messages.
        default_max_tokens: Default token limit for this model type.
        recommended_max_tokens: Recommended token limit for reasoning models.
    """
    is_reasoning_model: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = True
    supports_system_prompt: bool = True
    default_max_tokens: int = 300
    recommended_max_tokens: int = 4000


def detect_model_capabilities(model_name: str) -> CohereModelCapabilities:
    """
    Detect the capabilities and constraints of a Cohere model.

    Args:
        model_name: The model identifier (e.g., 'command-a-03-2025').

    Returns:
        CohereModelCapabilities: Dataclass containing all model capabilities.

    Example:
        >>> caps = detect_model_capabilities('command-a-reasoning-08-2025')
        >>> caps.is_reasoning_model
        True
        >>> caps = detect_model_capabilities('command-a-vision-07-2025')
        >>> caps.supports_vision
        True
    """
    model_lower = model_name.lower()
    caps = CohereModelCapabilities()

    # Check if it's a reasoning model
    for pattern in REASONING_MODEL_PATTERNS:
        if pattern in model_lower:
            caps.is_reasoning_model = True
            caps.default_max_tokens = 4000
            caps.recommended_max_tokens = 8000
            break

    # Check vision support
    for pattern in VISION_CAPABLE_PATTERNS:
        if pattern in model_lower:
            caps.supports_vision = True
            break

    if DEBUG_COHERE:
        logger.info(f"Model capabilities for {model_name}: {caps}")

    return caps


# =============================================================================
# Message Processing Functions
# =============================================================================

def _convert_messages_to_v2_format(
    messages: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """
    Convert SimplerLLM message format to Cohere V2 API format.

    V2 API uses a unified 'messages' array with roles:
    - 'system': System instructions
    - 'user': User messages
    - 'assistant': Model responses

    Args:
        messages: List of message dictionaries.
        system_prompt: System prompt to prepend.

    Returns:
        List of messages in Cohere V2 format.
    """
    v2_messages = []

    # Add system prompt as first message
    if system_prompt:
        v2_messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Convert each message
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map roles (V1 used CHATBOT, V2 uses assistant)
        if role.upper() == "CHATBOT" or role == "model":
            role = "assistant"

        v2_messages.append({
            "role": role,
            "content": content
        })

    return v2_messages


def _build_api_params(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    json_mode: bool,
    caps: CohereModelCapabilities,
) -> Dict[str, Any]:
    """
    Build the API parameters dictionary for Cohere V2 API.

    Args:
        model_name: The model identifier.
        messages: List of message dictionaries in V2 format.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        json_mode: Whether to force JSON output format.
        caps: CohereModelCapabilities instance.

    Returns:
        Dict[str, Any]: Parameters ready for the Cohere API call.
    """
    params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "p": top_p,  # Cohere uses 'p' for top_p
    }

    # Handle JSON mode
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    return params


# =============================================================================
# Response Extraction Functions
# =============================================================================

def _extract_response_text(response) -> str:
    """
    Extract generated text from Cohere V2 API response.

    V2 API response structure:
        response.message.content[0].text

    Args:
        response: Cohere API response object.

    Returns:
        str: The generated text.
    """
    try:
        # V2 API: response.message.content is a list of content blocks
        if hasattr(response, 'message') and response.message:
            content_blocks = response.message.content
            if content_blocks and len(content_blocks) > 0:
                # Content block can be TextAssistantMessageResponseContentItem
                first_block = content_blocks[0]
                if hasattr(first_block, 'text'):
                    return first_block.text
                elif isinstance(first_block, dict):
                    return first_block.get('text', '')
        return ""
    except (AttributeError, IndexError, TypeError) as e:
        if DEBUG_COHERE:
            logger.warning(f"Error extracting response text: {e}")
        return ""


def _extract_token_usage(response) -> tuple:
    """
    Extract token usage from Cohere API response.

    Args:
        response: Cohere API response object.

    Returns:
        tuple: (input_tokens, output_tokens)
    """
    try:
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            # V2 API uses billed_units for token counts
            if hasattr(usage, 'billed_units') and usage.billed_units:
                billed = usage.billed_units
                input_tokens = getattr(billed, 'input_tokens', 0) or 0
                output_tokens = getattr(billed, 'output_tokens', 0) or 0
                return (input_tokens, output_tokens)
            # Fallback to direct token counts if available
            input_tokens = getattr(usage, 'input_tokens', 0) or 0
            output_tokens = getattr(usage, 'output_tokens', 0) or 0
            return (input_tokens, output_tokens)
        return (0, 0)
    except (AttributeError, TypeError) as e:
        if DEBUG_COHERE:
            logger.warning(f"Error extracting token usage: {e}")
        return (0, 0)


def _extract_finish_reason(response) -> Optional[str]:
    """
    Extract finish reason from Cohere API response.

    Args:
        response: Cohere API response object.

    Returns:
        Optional[str]: Normalized finish reason.
    """
    try:
        if hasattr(response, 'finish_reason'):
            reason = response.finish_reason
            # Map Cohere finish reasons to common format
            reason_map = {
                "COMPLETE": "stop",
                "END_TURN": "stop",
                "MAX_TOKENS": "length",
                "ERROR": "error",
                "ERROR_TOXIC": "content_filter",
                "ERROR_LIMIT": "length",
            }
            reason_str = str(reason) if reason else ""
            return reason_map.get(reason_str, reason_str.lower())
        return None
    except (AttributeError, TypeError):
        return None


# =============================================================================
# Main Generation Functions
# =============================================================================

def generate_response(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Generate a response using Cohere's V2 Chat API.

    Supports all Cohere Command models including standard, reasoning,
    and vision models. Automatically detects model capabilities.

    Args:
        model_name: The Cohere model to use (e.g., 'command-a-03-2025').
        system_prompt: System instruction for the model.
        messages: List of message dicts with 'role' and 'content' keys.
            For vision, content can be a list of text/image objects.
        temperature: Sampling temperature (0-2). Default: 0.7
        max_tokens: Maximum tokens to generate. Default: 300
        top_p: Nucleus sampling parameter. Default: 1.0
        full_response: If True, returns LLMFullResponse with metadata.
            If False, returns just the generated text. Default: False
        api_key: Cohere API key. Falls back to COHERE_API_KEY env var.
        json_mode: If True, forces JSON output format. Default: False
        timeout: Request timeout in seconds. Default: None (no timeout)

    Returns:
        str: Generated text if full_response=False
        LLMFullResponse: Full response object if full_response=True

    Raises:
        ValueError: If messages is None or empty
        Exception: On API errors after retries exhausted

    Example:
        >>> response = generate_response(
        ...     model_name="command-a-03-2025",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )
    """
    # Import Cohere SDK here to avoid import errors if not installed
    try:
        from cohere import ClientV2
        from cohere.core.api_error import ApiError
    except ImportError:
        raise ImportError(
            "Cohere SDK not installed. Install with: pip install cohere>=5.0"
        )

    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    start_time = time.time() if full_response else None

    # Get API key
    api_key = api_key or os.getenv("COHERE_API_KEY", "")
    if not api_key:
        raise ValueError("COHERE_API_KEY not provided and not found in environment")

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    if DEBUG_COHERE:
        logger.info(f"Using model: {model_name}")
        if caps.is_reasoning_model:
            logger.info("Detected reasoning model")
        if caps.supports_vision:
            logger.info("Model supports vision")

    # Initialize V2 client with optional timeout
    client_kwargs = {"api_key": api_key}
    if timeout:
        client_kwargs["timeout"] = timeout
    client = ClientV2(**client_kwargs)

    # Convert messages to V2 format
    v2_messages = _convert_messages_to_v2_format(messages, system_prompt)

    # Build API parameters
    params = _build_api_params(
        model_name=model_name,
        messages=v2_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        caps=caps,
    )

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            # Use V2 chat endpoint via SDK
            response = client.chat(**params)

            # Extract response components
            generated_text = _extract_response_text(response)
            input_tokens, output_tokens = _extract_token_usage(response)
            finish_reason = _extract_finish_reason(response)

            if full_response:
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=time.time() - start_time,
                    input_token_count=input_tokens,
                    output_token_count=output_tokens,
                    llm_provider_response=response,
                    finish_reason=finish_reason,
                    is_reasoning_model=caps.is_reasoning_model,
                )
            return generated_text

        except ApiError as e:
            # Check if it's a rate limit error (status 429)
            if hasattr(e, 'status_code') and e.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    if DEBUG_COHERE:
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Rate limit exceeded after {MAX_RETRIES} attempts: {e}")
            else:
                # Don't retry other API errors
                raise Exception(f"Cohere API error: {str(e)}")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                if DEBUG_COHERE:
                    logger.warning(f"Error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")


async def generate_response_async(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    timeout: Optional[float] = None,
) -> Union[str, LLMFullResponse]:
    """
    Asynchronously generate a response using Cohere's V2 Chat API.

    This is the async version of generate_response(). See generate_response()
    for full parameter documentation.

    Example:
        >>> import asyncio
        >>> response = asyncio.run(generate_response_async(
        ...     model_name="command-a-03-2025",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... ))
    """
    # Import Cohere SDK here
    try:
        from cohere import AsyncClientV2
        from cohere.core.api_error import ApiError
    except ImportError:
        raise ImportError(
            "Cohere SDK not installed. Install with: pip install cohere>=5.0"
        )

    # Validate inputs
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")

    start_time = time.time() if full_response else None

    # Get API key
    api_key = api_key or os.getenv("COHERE_API_KEY", "")
    if not api_key:
        raise ValueError("COHERE_API_KEY not provided and not found in environment")

    # Detect model capabilities
    caps = detect_model_capabilities(model_name)

    # Initialize async V2 client
    client_kwargs = {"api_key": api_key}
    if timeout:
        client_kwargs["timeout"] = timeout
    client = AsyncClientV2(**client_kwargs)

    # Convert messages to V2 format
    v2_messages = _convert_messages_to_v2_format(messages, system_prompt)

    # Build API parameters
    params = _build_api_params(
        model_name=model_name,
        messages=v2_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_mode=json_mode,
        caps=caps,
    )

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat(**params)

            # Extract response components
            generated_text = _extract_response_text(response)
            input_tokens, output_tokens = _extract_token_usage(response)
            finish_reason = _extract_finish_reason(response)

            if full_response:
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=time.time() - start_time,
                    input_token_count=input_tokens,
                    output_token_count=output_tokens,
                    llm_provider_response=response,
                    finish_reason=finish_reason,
                    is_reasoning_model=caps.is_reasoning_model,
                )
            return generated_text

        except ApiError as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Rate limit exceeded after {MAX_RETRIES} attempts: {e}")
            else:
                raise Exception(f"Cohere API error: {str(e)}")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")


# =============================================================================
# Embeddings Functions
# =============================================================================

def generate_embeddings(
    user_input,
    model_name: str = "embed-english-v3.0",
    full_response: bool = False,
    api_key: Optional[str] = None,
    input_type: str = "search_document",
    embedding_types: Optional[List[str]] = None,
    truncate: str = "END",
) -> any:
    """
    Generate embeddings using Cohere API.

    Args:
        user_input (str or list): Text(s) to embed.
        model_name (str): Model name to use for embeddings.
            Options: embed-english-v3.0, embed-multilingual-v3.0, embed-v4.0
        full_response (bool): Whether to return full response object.
        api_key (str): Cohere API key. Falls back to COHERE_API_KEY env var.
        input_type (str): Type of input for embeddings.
            Options: "search_document", "search_query", "classification", "clustering"
        embedding_types (list): List of embedding types to return.
        truncate (str): How to truncate long inputs. Options: "START", "END", "NONE"

    Returns:
        Embeddings array or LLMEmbeddingsResponse if full_response=True.

    Example:
        >>> embeddings = generate_embeddings("Hello world")
        >>> len(embeddings)
        1024
    """
    try:
        from cohere import ClientV2
    except ImportError:
        raise ImportError(
            "Cohere SDK not installed. Install with: pip install cohere>=5.0"
        )

    start_time = time.time()

    # Get API key
    api_key = api_key or os.getenv("COHERE_API_KEY", "")
    if not api_key:
        raise ValueError("COHERE_API_KEY not provided and not found in environment")

    # Initialize client
    client = ClientV2(api_key=api_key)

    # Prepare texts - ensure it's a list
    if isinstance(user_input, str):
        texts = [user_input]
    else:
        texts = list(user_input)

    # Build parameters
    params = {
        "model": model_name,
        "texts": texts,
        "input_type": input_type,
        "truncate": truncate,
    }

    if embedding_types:
        params["embedding_types"] = embedding_types

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embed(**params)

            # Extract embeddings
            embeddings = response.embeddings

            # Handle different embedding response formats
            if hasattr(embeddings, 'float_') and embeddings.float_:
                embeddings = embeddings.float_
            elif isinstance(embeddings, list):
                pass  # Already in correct format

            # Return single embedding if single input was provided
            if isinstance(user_input, str) and isinstance(embeddings, list) and len(embeddings) > 0:
                embeddings = embeddings[0]

            if full_response:
                return LLMEmbeddingsResponse(
                    generated_embedding=embeddings,
                    model=model_name,
                    process_time=time.time() - start_time,
                    llm_provider_response=response,
                )
            return embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")


async def generate_embeddings_async(
    user_input,
    model_name: str = "embed-english-v3.0",
    full_response: bool = False,
    api_key: Optional[str] = None,
    input_type: str = "search_document",
    embedding_types: Optional[List[str]] = None,
    truncate: str = "END",
) -> any:
    """
    Asynchronously generate embeddings using Cohere API.

    See generate_embeddings() for full parameter documentation.

    Example:
        >>> import asyncio
        >>> embeddings = asyncio.run(generate_embeddings_async("Hello world"))
    """
    try:
        from cohere import AsyncClientV2
    except ImportError:
        raise ImportError(
            "Cohere SDK not installed. Install with: pip install cohere>=5.0"
        )

    start_time = time.time()

    # Get API key
    api_key = api_key or os.getenv("COHERE_API_KEY", "")
    if not api_key:
        raise ValueError("COHERE_API_KEY not provided and not found in environment")

    # Initialize async client
    client = AsyncClientV2(api_key=api_key)

    # Prepare texts
    if isinstance(user_input, str):
        texts = [user_input]
    else:
        texts = list(user_input)

    # Build parameters
    params = {
        "model": model_name,
        "texts": texts,
        "input_type": input_type,
        "truncate": truncate,
    }

    if embedding_types:
        params["embedding_types"] = embedding_types

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.embed(**params)

            # Extract embeddings
            embeddings = response.embeddings

            if hasattr(embeddings, 'float_') and embeddings.float_:
                embeddings = embeddings.float_
            elif isinstance(embeddings, list):
                pass

            if isinstance(user_input, str) and isinstance(embeddings, list) and len(embeddings) > 0:
                embeddings = embeddings[0]

            if full_response:
                return LLMEmbeddingsResponse(
                    generated_embedding=embeddings,
                    model=model_name,
                    process_time=time.time() - start_time,
                    llm_provider_response=response,
                )
            return embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")
