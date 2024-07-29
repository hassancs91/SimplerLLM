from typing import Dict, Optional
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import time
import requests
from .llm_response_models import LLMFullResponse
from typing import Optional, Dict, List

# Load environment variables
load_dotenv()

# Constants
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


def generate_response(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key = None
) -> Optional[Dict]:
    start_time = time.time()  # Record the start time
    """
    Sends a POST request to the generativelanguage API to generate content based on the provided text
    or a list of messages with specified generation configuration settings.

    Parameters:
    - model_name (str): The name of the model to use for content generation.
    - prompt (Optional[str]): The single text prompt for content generation. Required if 'messages' is not provided.
    - system_prompt (str): The default system-generated prompt used if 'prompt' is provided.
    - messages (Optional[List[Dict]]): A structured list of message parts for complex conversations. Required if 'prompt' is not provided.
    - temperature (float): Controls randomness in generation. Higher values result in more random completions.
    - max_tokens (int): The maximum number of tokens to generate.
    - top_p (float): Nucleus sampling parameter controlling the size of the probability mass to consider for token generation.
    - full_response (bool): If True, returns the full response from the API instead of just the generated text.

    Returns:
    - Optional[Dict]: The generated text or full response depending on the 'full_response' flag.
    """

    retry_attempts = 3
    retry_delay = 1  # initial delay between retries in seconds

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}


    payload = {
        "contents": messages,
        # "system_instruction":
        #     {
        #         "parts": [
        #             {
        #                 "text": system_prompt
        #             }
        #         ]
        #     },
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": top_p,
        },
    }

    for attempt in range(retry_attempts):
        try:
            response = requests.post(
                url, headers=headers, json=payload, params={"key": api_key}
            )
            response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)

            if full_response:
                return LLMFullResponse(
                    generated_text=response.json()["candidates"][0]["content"]["parts"][
                        0
                    ]["text"],
                    model=model_name,
                    process_time=time.time() - start_time,
                    llm_provider_response=response.json(),
                )

            else:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]

        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Double the delay each retry

    print("All retry attempts failed.")
    return None

async def generate_response_async(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key = None
) -> Optional[Dict]:
    """
    Sends a POST request to the generativelanguage API to generate content based on the provided text
    or a list of messages with specified generation configuration settings.

    Parameters:
    - model_name (str): The name of the model to use for content generation.
    - prompt (Optional[str]): The single text prompt for content generation. Required if 'messages' is not provided.
    - system_prompt (str): The default system-generated prompt used if 'prompt' is provided.
    - messages (Optional[List[Dict]]): A structured list of message parts for complex conversations. Required if 'prompt' is not provided.
    - temperature (float): Controls randomness in generation. Higher values result in more random completions.
    - max_tokens (int): The maximum number of tokens to generate.
    - top_p (float): Nucleus sampling parameter controlling the size of the probability mass to consider for token generation.
    - full_response (bool): If True, returns the full response from the API instead of just the generated text.

    Returns:
    - Optional[Dict]: The generated text or full response depending on the 'full_response' flag.
    """

    start_time = time.time()  # Record the start time

    retry_attempts = 3
    retry_delay = 1  # initial delay between retries in seconds

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}

    
    payload = {
        "contents": messages,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": top_p,
        },
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(retry_attempts):
            try:
                async with session.post(
                    url, headers=headers, json=payload, params={"key": api_key}
                ) as response:

                    data = await response.json()

                    if full_response:
                        return LLMFullResponse(
                            generated_text=data["candidates"][0]["content"]["parts"][0][
                                "text"
                            ],
                            model=model_name,
                            process_time=time.time() - start_time,
                            llm_provider_response=data,
                        )

                    else:
                        return data["candidates"][0]["content"]["parts"][0]["text"]

            except aiohttp.ClientError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Double the delay each retry

    print("All retry attempts failed.")
    return None
