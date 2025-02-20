from typing import Dict, Optional
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import time
import json
import requests
from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv(override=True)


MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
OLLAMA_URL = str(os.getenv("OLLAMA_URL", "http://localhost:11434/")) + "api/chat"

def generate_response(
    model_name: str,
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
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

    for attempt in range(retry_attempts):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)

            if full_response:
                response_json = response.json()
                return LLMFullResponse(
                    generated_text=response_json["message"]["content"],
                    model=model_name,
                    process_time=time.time() - start_time,
                    input_token_count=response_json["prompt_eval_count"],
                    output_token_count=response_json["eval_count"],
                    llm_provider_response=response_json,
                )

            else:
                return response.json()["message"]["content"]

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
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
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


    async with aiohttp.ClientSession() as session:
        for attempt in range(retry_attempts):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)
                    data = await response.json()
                    if full_response:
                        return LLMFullResponse(
                            generated_text=data["message"]["content"],
                            model=model_name,
                            process_time=time.time() - start_time,
                            input_token_count=data["prompt_eval_count"],
                            output_token_count=data["eval_count"],
                            llm_provider_response=data,
                        )

                    else:
                        return data["message"]["content"]

            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay each retry
                else:
                    error_msg = f"Failed after {retry_attempts} attempts due to: {e}"
                    raise Exception(error_msg)
