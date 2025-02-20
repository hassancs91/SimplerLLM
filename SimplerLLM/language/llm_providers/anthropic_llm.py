from typing import Dict, Optional
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import time
import requests
from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv(override=True)

# Constants
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


def generate_response(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    prompt_caching: bool = False,
    cached_input: str = "",
    cache_control_type : str = "ephemeral",
    api_key= None,
    json_mode=False,
) -> Optional[Dict]:
    """
    Makes a POST request to the Anthropic API to generate content based on the provided text
    with specified generation configuration settings.
    """
    start_time = time.time()  # Record the start time
    retry_attempts = 3
    retry_delay = 1  # initial delay between retries in seconds

    # Define the URL and headers
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    if prompt_caching:
        headers["anthropic-beta"] = "prompt-caching-2024-07-31"


    # Base payload
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    if prompt_caching:
        payload["system"] = [
            {
                "type": "text",
                "text": system_prompt
            },
            {
                "type": "text",
                "text": cached_input,
                "cache_control": {"type": cache_control_type}
            }
        ]
    else:
        payload["system"] = system_prompt

    for attempt in range(retry_attempts):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)

            if full_response:
                response_json = response.json()
                return LLMFullResponse(
                    generated_text=response_json["content"][0]["text"],
                    model=model_name,
                    process_time=time.time() - start_time,
                    input_token_count=response_json["usage"]["input_tokens"],
                    output_token_count=response_json["usage"]["output_tokens"],
                    llm_provider_response=response_json,
                )

            else:
                return response.json()["content"][0]["text"]

        except Exception as e:
            if attempt < retry_attempts - 1:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Double the delay each retry
            else:
                error_msg = f"Failed after {retry_attempts} attempts due to: {e}"
                raise Exception(error_msg)

async def generate_response_async(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    prompt_caching: bool = False,
    cached_input: str = "",
    cache_control_type: str = "ephemeral",
    api_key= None,
    json_mode=False,
) -> Optional[Dict]:
    """
    Makes an asynchronous POST request to the Anthropic API to generate content based on the provided text
    with specified generation configuration settings using asyncio.
    """
    start_time = time.time()  # Record the start time
    retry_attempts = 3
    retry_delay = 1  # initial delay between retries in seconds

    # Define the URL and headers
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    if prompt_caching:
        headers["anthropic-beta"] = "prompt-caching-2024-07-31"


    # Base payload
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    if prompt_caching:
        payload["system"] = [
            {
                "type": "text",
                "text": system_prompt
            },
            {
                "type": "text",
                "text": cached_input,
                "cache_control": {"type": cache_control_type}
            }
        ]
    else:
        payload["system"] = system_prompt

    async with aiohttp.ClientSession() as session:
        for attempt in range(retry_attempts):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    if full_response:
                        return LLMFullResponse(
                            generated_text=json_response["content"][0]["text"],
                            model=model_name,
                            process_time=time.time() - start_time,
                            input_token_count=json_response["usage"]["input_tokens"],
                            output_token_count=json_response["usage"]["output_tokens"],
                            llm_provider_response=json_response,
                        )
                    else:
                        return json_response["content"][0]["text"]

            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay each retry
                else:
                    error_msg = f"Failed after {retry_attempts} attempts due to: {e}"
                    raise Exception(error_msg)
