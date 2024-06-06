from typing import Dict, Optional
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import time
import requests
from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv()

# Constants
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
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
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Create the data payload
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system_prompt,  # <-- system prompt
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    for attempt in range(retry_attempts):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)

            if full_response:
                return LLMFullResponse(
                    generated_text=response.json()["content"][0]["text"],
                    model=model_name,
                    process_time=time.time() - start_time,
                    llm_provider_response=response.json(),
                )

            else:
                return response.json()["content"][0]["text"]

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Double the delay each retry

    print("All retry attempts failed.")
    return None


async def generate_response_async(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
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
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Create the data payload
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(retry_attempts):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)
                    data = await response.json()
                    if full_response:
                        return LLMFullResponse(
                            generated_text=data["content"][0]["text"],
                            model=model_name,
                            process_time=time.time() - start_time,
                            llm_provider_response=data,
                        )
                    else:
                        return data["content"][0]["text"]

            except aiohttp.ClientError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Double the delay each retry

    print("All retry attempts failed.")
    return None
