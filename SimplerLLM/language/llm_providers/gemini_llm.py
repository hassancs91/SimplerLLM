from typing import Dict, Optional
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import time
import requests
from .llm_response_models import LLMFullResponse
from typing import Optional, Dict, List
import json

# Load environment variables
load_dotenv(override=True)

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
    prompt_caching: bool = False,
    cache_id: str = None,
    api_key = None,
    json_mode=False,
) -> Optional[Dict]:

    start_time = time.time()  
    retry_attempts = 3
    retry_delay = 1  
    headers = {"Content-Type": "application/json"}

    if prompt_caching:
        # Use the cached payload if caching is enabled
        payload = {
            "contents": messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            },
            "cachedContent": cache_id
        }

    else:
        # Use the normal payload if caching is disabled
        payload = {
            "contents": messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            },
        }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    for attempt in range(retry_attempts):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            if full_response:
                response_json = response.json()
                return LLMFullResponse(
                    generated_text=response_json["candidates"][0]["content"]["parts"][0]["text"],
                    model=model_name,
                    process_time=time.time() - start_time,
                    input_token_count=response_json["usageMetadata"]["promptTokenCount"],
                    output_token_count=response_json["usageMetadata"]["candidatesTokenCount"],
                    llm_provider_response=response_json,
                )
            else:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]

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
    messages: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    prompt_caching: bool = False,
    cache_id: str = None,
    api_key=None,
    json_mode=False,
) -> Optional[Dict]:
    
    start_time = time.time()
    retry_attempts = 3
    retry_delay = 1  # Initial retry delay in seconds
    headers = {"Content-Type": "application/json"}
    
    if prompt_caching:
        # Use the cached payload if caching is enabled
        payload = {
            "contents": messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            },
            "cachedContent": cache_id
        }

    else:
        # Use the normal payload if caching is disabled
        payload = {
            "contents": messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            },
        }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    for attempt in range(retry_attempts):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    json_response = await response.json()

                    if full_response:
                        return LLMFullResponse(
                            generated_text=json_response["candidates"][0]["content"]["parts"][0]["text"],
                            model=model_name,
                            process_time=time.time() - start_time,
                            input_token_count=json_response["usageMetadata"]["promptTokenCount"],
                            output_token_count=json_response["usageMetadata"]["candidatesTokenCount"],
                            llm_provider_response=json_response,
                        )
                    else:
                        return json_response["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            if attempt < retry_attempts - 1:
                print(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Double the delay each retry
            else:
                error_msg = f"Failed after {retry_attempts} attempts due to: {e}"
                raise Exception(error_msg)
