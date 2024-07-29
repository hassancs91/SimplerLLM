from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import aiohttp
import json
import asyncio
import time
import requests
from .llm_response_models import LLMFullResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatMessagesRequest(BaseModel):
    last_prompt: str
    conversation_history: List[ChatMessage]  # List of chat messages


def generate_response(
    model_name: str,
    system_prompt: str = "You are a helpful AI Assistant",
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key= None,
    user_id = None
) -> Optional[Dict]:
    """
    Makes a POST request to the LWH API to generate content based on the provided text
    with specified generation configuration settings.
    """
    start_time = time.time()  # Record the start time
    retry_delay = 1  # initial delay between retries in seconds

    # Define the URL and headers
    url = "https://learnwithhasan.com/wp-json/lwh-user-api/llm-playground/v1/chat"
    headers = {
        "X-User-ID": user_id,
        "X-Auth-Key": api_key,
        "Content-Type": "application/json",
    }

    last_prompt = messages[-1]['content']
    conversation_history = messages[:-1]


    # Create the data payload
    payload = {
        "last_prompt": last_prompt,
        "conversation_history":  conversation_history,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
        "top_p": top_p,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)

            if full_response:
                return {
                    "generated_text": str(json.loads(response.text)["result"]).strip('"'),
                    "model": model_name,
                    "process_time": time.time() - start_time,
                    "llm_provider_response": response.json(),
                }

            else:
                
                return str(json.loads(response.text)["result"]).strip('"')

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
    api_key= None,
    user_id = None
) -> Optional[Dict]:
    """
    Makes an asynchronous POST request to the Anthropic API to generate content based on the provided text
    with specified generation configuration settings using asyncio.
    """
    start_time = time.time()  # Record the start time
    retry_delay = 1  # initial delay between retries in seconds

    # Define the URL and headers
    url = "https://learnwithhasan.com/wp-json/lwh-user-api/llm-playground/v1/chat"
    headers = {
        "X-User-ID": user_id,
        "X-Auth-Key": api_key,
        "Content-Type": "application/json",
    }

    last_prompt = messages[-1]['content']
    conversation_history = messages[:-1]


    # Create the data payload
    payload = {
        "last_prompt": last_prompt,
        "conversation_history":  conversation_history,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
        "top_p": top_p,
    }


    async with aiohttp.ClientSession() as session:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)
                    if full_response:
                        return {
                            "generated_text": str(json.loads(response.text)["result"]).replace('"', ''),
                            "model": model_name,
                            "process_time": time.time() - start_time,
                            "llm_provider_response": response.json(),
                        }

                    else:
                        
                        return str(json.loads(response.text)["result"]).replace('"', '')

            except aiohttp.ClientError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Double the delay each retry

    print("All retry attempts failed.")
    return None
