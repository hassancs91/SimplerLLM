from dotenv import load_dotenv
import asyncio
import os
import time
import requests
import aiohttp
from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

def generate_response(
    model_name,
    messages=None,
    temperature=0.7,
    max_tokens=300,
    top_p=1.0,
    full_response=False,
    api_key=None,
    json_mode=False,
):
    start_time = time.time() if full_response else None
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False
    }
    
    if json_mode:
        data["response_format"] = {"type": "json_object"}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=result["usage"]["prompt_tokens"],
                    output_token_count=result["usage"]["completion_tokens"],
                    llm_provider_response=result,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)

async def generate_response_async(
    model_name,
    messages=None,
    temperature=0.7,
    max_tokens=300,
    top_p=1.0,
    full_response=False,
    api_key=None,
    json_mode=False,
):
    start_time = time.time() if full_response else None
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False
    }
    
    if json_mode:
        data["response_format"] = {"type": "json_object"}
    
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepseek.com/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    generated_text = result["choices"][0]["message"]["content"]

                    if full_response:
                        end_time = time.time()
                        process_time = end_time - start_time
                        return LLMFullResponse(
                            generated_text=generated_text,
                            model=model_name,
                            process_time=process_time,
                            input_token_count=result["usage"]["prompt_tokens"],
                            output_token_count=result["usage"]["completion_tokens"],
                            llm_provider_response=result,
                        )
                    return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
