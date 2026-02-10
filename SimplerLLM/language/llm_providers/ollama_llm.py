from typing import Dict, Optional, List, Any
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import time
import json
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv(override=True)

# Configuration from environment
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 120))  # Ollama can be slow for large models
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/")
OLLAMA_URL = OLLAMA_BASE_URL + "api/chat"

# Vision-capable model patterns
OLLAMA_VISION_PATTERNS = ["llava", "llama3.2-vision", "moondream", "bakllava", "minicpm-v"]


def is_vision_model(model_name: str) -> bool:
    """Check if the model supports vision based on model name patterns."""
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in OLLAMA_VISION_PATTERNS)

def generate_response(
    model_name: str,
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    json_mode: bool = False,
    images: list = None,
    detail: str = "auto",
    web_search: bool = False,
) -> Optional[Dict]:
    """
    Makes a POST request to the Ollama API to generate content.

    Args:
        model_name: The Ollama model to use (e.g., "llama3.2", "mistral", "llava")
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Controls randomness (0.0-2.0). Default 0.7
        max_tokens: Maximum tokens to generate. Default 300
        top_p: Nucleus sampling parameter. Default 1.0
        full_response: If True, returns LLMFullResponse with metadata
        json_mode: If True, forces JSON output format
        images: List of base64-encoded images for vision models
        detail: Image detail level (not used by Ollama, included for API compatibility)
        web_search: Not supported by Ollama (parameter ignored with warning)

    Returns:
        Generated text string, or LLMFullResponse if full_response=True

    Raises:
        Exception: If request fails after all retry attempts
    """
    start_time = time.time()
    retry_delay = RETRY_DELAY

    # Define the URL and headers
    url = OLLAMA_URL
    headers = {
        "content-type": "application/json",
    }

    # Create the data payload
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
        "stream": False,
    }

    # Add JSON mode if requested
    if json_mode:
        payload["format"] = "json"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=OLLAMA_TIMEOUT)

            # Check for model not found error
            if response.status_code == 404:
                try:
                    error_body = response.json()
                    if "not found" in error_body.get("error", "").lower():
                        raise Exception(
                            f"Model '{model_name}' not found. "
                            f"Pull it with 'ollama pull {model_name}' or check available models with 'ollama list'."
                        )
                except (ValueError, KeyError):
                    pass

            response.raise_for_status()

            response_json = response.json()

            if full_response:
                return LLMFullResponse(
                    generated_text=response_json["message"]["content"],
                    model=model_name,
                    process_time=time.time() - start_time,
                    input_token_count=response_json.get("prompt_eval_count"),
                    output_token_count=response_json.get("eval_count"),
                    llm_provider_response=response_json,
                )
            else:
                return response_json["message"]["content"]

        except ConnectionError as e:
            raise Exception(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Ensure Ollama is running with 'ollama serve'. "
                f"Original error: {e}"
            )

        except Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(
                    f"Request timed out after {OLLAMA_TIMEOUT}s for model '{model_name}'. "
                    "Consider increasing OLLAMA_TIMEOUT environment variable."
                )

        except RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"Request failed after {MAX_RETRIES} attempts: {e}")

        except Exception as e:
            # Don't retry on non-request exceptions (like model not found)
            if "not found" in str(e).lower() or "cannot connect" in str(e).lower():
                raise
            if attempt < MAX_RETRIES - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")

async def generate_response_async(
    model_name: str,
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    json_mode: bool = False,
    images: list = None,
    detail: str = "auto",
    web_search: bool = False,
) -> Optional[Dict]:
    """
    Makes an asynchronous POST request to the Ollama API to generate content.

    Args:
        model_name: The Ollama model to use (e.g., "llama3.2", "mistral", "llava")
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Controls randomness (0.0-2.0). Default 0.7
        max_tokens: Maximum tokens to generate. Default 300
        top_p: Nucleus sampling parameter. Default 1.0
        full_response: If True, returns LLMFullResponse with metadata
        json_mode: If True, forces JSON output format
        images: List of base64-encoded images for vision models
        detail: Image detail level (not used by Ollama, included for API compatibility)
        web_search: Not supported by Ollama (parameter ignored with warning)

    Returns:
        Generated text string, or LLMFullResponse if full_response=True

    Raises:
        Exception: If request fails after all retry attempts
    """
    start_time = time.time()
    retry_delay = RETRY_DELAY

    # Define the URL and headers
    url = OLLAMA_URL
    headers = {
        "content-type": "application/json",
    }

    # Create the data payload
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
        "stream": False,
    }

    # Add JSON mode if requested
    if json_mode:
        payload["format"] = "json"

    timeout = aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(MAX_RETRIES):
                try:
                    async with session.post(url, headers=headers, json=payload) as response:
                        # Check for model not found error
                        if response.status == 404:
                            try:
                                error_body = await response.json()
                                if "not found" in error_body.get("error", "").lower():
                                    raise Exception(
                                        f"Model '{model_name}' not found. "
                                        f"Pull it with 'ollama pull {model_name}' or check available models with 'ollama list'."
                                    )
                            except (ValueError, KeyError):
                                pass

                        response.raise_for_status()
                        data = await response.json()

                        if full_response:
                            return LLMFullResponse(
                                generated_text=data["message"]["content"],
                                model=model_name,
                                process_time=time.time() - start_time,
                                input_token_count=data.get("prompt_eval_count"),
                                output_token_count=data.get("eval_count"),
                                llm_provider_response=data,
                            )
                        else:
                            return data["message"]["content"]

                except asyncio.TimeoutError:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise Exception(
                            f"Request timed out after {OLLAMA_TIMEOUT}s for model '{model_name}'. "
                            "Consider increasing OLLAMA_TIMEOUT environment variable."
                        )

                except aiohttp.ClientError as e:
                    # Don't retry on model not found
                    if "not found" in str(e).lower():
                        raise
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise Exception(f"Request failed after {MAX_RETRIES} attempts: {e}")

                except Exception as e:
                    # Don't retry on non-retryable exceptions
                    if "not found" in str(e).lower() or "cannot connect" in str(e).lower():
                        raise
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")

    except aiohttp.ClientConnectorError as e:
        raise Exception(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Ensure Ollama is running with 'ollama serve'. "
            f"Original error: {e}"
        )
