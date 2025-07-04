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
    api_key=None,
    json_mode=False,
) -> Optional[Dict]:
    """
    Makes a POST request to the Cohere API to generate content based on the provided text
    with specified generation configuration settings.
    """
    start_time = time.time()  # Record the start time
    retry_attempts = 3
    retry_delay = 1  # initial delay between retries in seconds

    # Define the URL and headers
    url = "https://api.cohere.com/v1/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Convert messages to Cohere format
    chat_history = []
    user_message = ""
    
    if messages:
        for message in messages:
            if message["role"] == "user":
                user_message = message["content"]
            elif message["role"] == "assistant":
                chat_history.append({
                    "role": "CHATBOT",
                    "message": message["content"]
                })
            elif message["role"] == "system":
                # System messages are handled via preamble
                system_prompt = message["content"]

    # Base payload
    payload = {
        "model": model_name,
        "message": user_message,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "p": top_p,
        "preamble": system_prompt,
    }

    if chat_history:
        payload["chat_history"] = chat_history

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(retry_attempts):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)

            response_json = response.json()
            
            if full_response:
                return LLMFullResponse(
                    generated_text=response_json["text"],
                    model=model_name,
                    process_time=time.time() - start_time,
                    input_token_count=response_json.get("meta", {}).get("tokens", {}).get("input_tokens"),
                    output_token_count=response_json.get("meta", {}).get("tokens", {}).get("output_tokens"),
                    llm_provider_response=response_json,
                )
            else:
                return response_json["text"]

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
    api_key=None,
    json_mode=False,
) -> Optional[Dict]:
    """
    Makes an asynchronous POST request to the Cohere API to generate content based on the provided text
    with specified generation configuration settings using asyncio.
    """
    start_time = time.time()  # Record the start time
    retry_attempts = 3
    retry_delay = 1  # initial delay between retries in seconds

    # Define the URL and headers
    url = "https://api.cohere.com/v1/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Convert messages to Cohere format
    chat_history = []
    user_message = ""
    
    if messages:
        for message in messages:
            if message["role"] == "user":
                user_message = message["content"]
            elif message["role"] == "assistant":
                chat_history.append({
                    "role": "CHATBOT",
                    "message": message["content"]
                })
            elif message["role"] == "system":
                # System messages are handled via preamble
                system_prompt = message["content"]

    # Base payload
    payload = {
        "model": model_name,
        "message": user_message,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "p": top_p,
        "preamble": system_prompt,
    }

    if chat_history:
        payload["chat_history"] = chat_history

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with aiohttp.ClientSession() as session:
        for attempt in range(retry_attempts):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    
                    if full_response:
                        return LLMFullResponse(
                            generated_text=json_response["text"],
                            model=model_name,
                            process_time=time.time() - start_time,
                            input_token_count=json_response.get("meta", {}).get("tokens", {}).get("input_tokens"),
                            output_token_count=json_response.get("meta", {}).get("tokens", {}).get("output_tokens"),
                            llm_provider_response=json_response,
                        )
                    else:
                        return json_response["text"]

            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay each retry
                else:
                    error_msg = f"Failed after {retry_attempts} attempts due to: {e}"
                    raise Exception(error_msg)


def generate_embeddings(
    user_input,
    model_name: str = "embed-english-v3.0",
    full_response: bool = False,
    api_key=None,
    input_type="search_document",
    embedding_types=None,
    truncate="END",
) -> any:
    """
    Generate embeddings using Cohere API.
    
    Args:
        user_input (str or list): Text(s) to embed
        model_name (str): Model name to use for embeddings
        full_response (bool): Whether to return full response object
        api_key (str): Cohere API key
        input_type (str): "search_document", "search_query", "classification", "clustering"
        embedding_types (list): List of embedding types to return
        truncate (str): "START", "END", or "NONE"
    """
    from .llm_response_models import LLMEmbeddingsResponse
    
    start_time = time.time()
    retry_attempts = 3
    retry_delay = 1

    # Define the URL and headers
    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prepare texts - ensure it's a list
    if isinstance(user_input, str):
        texts = [user_input]
    else:
        texts = user_input

    # Base payload
    payload = {
        "model": model_name,
        "texts": texts,
        "input_type": input_type,
        "truncate": truncate,
    }

    if embedding_types:
        payload["embedding_types"] = embedding_types

    for attempt in range(retry_attempts):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            response_json = response.json()
            embeddings = response_json["embeddings"]
            
            # Return single embedding if single input was provided
            if isinstance(user_input, str):
                embeddings = embeddings[0]

            if full_response:
                return LLMEmbeddingsResponse(
                    generated_embedding=embeddings,
                    model=model_name,
                    process_time=time.time() - start_time,
                    llm_provider_response=response_json,
                )
            else:
                return embeddings

        except Exception as e:
            if attempt < retry_attempts - 1:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                error_msg = f"Failed after {retry_attempts} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_embeddings_async(
    user_input,
    model_name: str = "embed-english-v3.0",
    full_response: bool = False,
    api_key=None,
    input_type="search_document",
    embedding_types=None,
    truncate="END",
) -> any:
    """
    Asynchronously generate embeddings using Cohere API.
    
    Args:
        user_input (str or list): Text(s) to embed
        model_name (str): Model name to use for embeddings
        full_response (bool): Whether to return full response object
        api_key (str): Cohere API key
        input_type (str): "search_document", "search_query", "classification", "clustering"
        embedding_types (list): List of embedding types to return
        truncate (str): "START", "END", or "NONE"
    """
    from .llm_response_models import LLMEmbeddingsResponse
    
    start_time = time.time()
    retry_attempts = 3
    retry_delay = 1

    # Define the URL and headers
    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prepare texts - ensure it's a list
    if isinstance(user_input, str):
        texts = [user_input]
    else:
        texts = user_input

    # Base payload
    payload = {
        "model": model_name,
        "texts": texts,
        "input_type": input_type,
        "truncate": truncate,
    }

    if embedding_types:
        payload["embedding_types"] = embedding_types

    async with aiohttp.ClientSession() as session:
        for attempt in range(retry_attempts):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    
                    embeddings = json_response["embeddings"]
                    
                    # Return single embedding if single input was provided
                    if isinstance(user_input, str):
                        embeddings = embeddings[0]

                    if full_response:
                        return LLMEmbeddingsResponse(
                            generated_embedding=embeddings,
                            model=model_name,
                            process_time=time.time() - start_time,
                            llm_provider_response=json_response,
                        )
                    else:
                        return embeddings

            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    error_msg = f"Failed after {retry_attempts} attempts due to: {e}"
                    raise Exception(error_msg)