from typing import Dict, Optional
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import time
import requests

# Load environment variables
load_dotenv()

# Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


def __call_gemeni(
    model_name: str,
    user_prompt: str,
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2024,
    top_p: float = 0.8,
) -> Optional[Dict]:
    """
    Makes a POST request to the generativelanguage API to generate content based on the provided text
    with specified generation configuration settings.

    Parameters:
    - model_name (str): The name of the model to use for generation.
    - user_prompt (str): The text prompt for content generation.
    - system_prompt (str): The system-generated text prompt.
    - temperature (float): Controls randomness in generation. Higher values mean more random completions.
    - max_tokens (int): The maximum number of tokens to generate.
    - top_p (float): Nucleus sampling parameter controlling the size of the probability mass to consider for token generation.

    Returns:
    - dict: The response from the API.
    """
    # Define the URL and headers
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}

    # Create the data payload
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "model", "parts": [{"text": "ok"}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        "generationConfig": {
            "stopSequences": ["Title"],
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": top_p,
        },
    }

    try:
        response = requests.post(
            url, headers=headers, json=payload, params={"key": GEMINI_API_KEY}
        )
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def __generate_response(
    model_name,
    user_prompt,
    system_prompt,
    temperature,
    max_tokens,
    top_p,
    full_response=False,
):
    """
    Generates a response from the generative language model.

    Parameters:
    model_name (str): Name of the model to use for generation.
    user_prompt (str): The text prompt for content generation.
    system_prompt (str): The system-generated text prompt.
    temperature (float): Controls randomness in generation.
    max_tokens (int): Maximum number of tokens to generate.
    top_p (float): Nucleus sampling parameter.
    full_response (bool): If True, returns the full API response, else returns the generated text.

    Returns:
    str or dict: The generated text or the full response from the API, based on the 'full_response' flag.
    """
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")

    if not system_prompt:
        system_prompt = "You are a helpful AI Assitant"

    if not model_name or not isinstance(model_name, str):
        raise ValueError("model must be a non-empty string.")

    for attempt in range(MAX_RETRIES):
        try:
            response = __call_gemeni(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            if full_response:
                return response
            else:
                return response["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * 2**attempt
                time.sleep(delay)
            else:
                # Log the error or inform the user
                print(
                    f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}"
                )
                return None


def generate_text(
    model_name,
    user_prompt,
    system_prompt="You are a helpful AI Assistant",
    temperature=0.7,
    max_tokens=2024,
    top_p=0.8,
):
    """
    Generates using Gemini and returns only the text.
    """
    return __generate_response(
        model_name,
        user_prompt,
        system_prompt,
        temperature,
        max_tokens,
        top_p,
        full_response=False,
    )


def generate_full_response(
    model_name,
    user_prompt,
    system_prompt="You are a helpful AI Assistant",
    max_tokens=2000,
    top_p=1.0,
    temperature=0.7,
):
    """
    Generates the full response from Gemini.
    """
    return __generate_response(
        model_name,
        user_prompt,
        system_prompt,
        temperature,
        max_tokens,
        top_p,
        full_response=True,
    )


async def __call_gemeni_async(
    model_name,
    user_prompt,
    system_prompt,
    temperature=0.7,
    max_tokens=2024,
    top_p=0.8,
):
    """
    Asynchronously makes a POST request to the generativelanguage API to generate content based on the provided text
    with specified generation configuration settings.

    Parameters:
    user_prompt (str): The text prompt for content generation.
    temperature (float): Controls randomness in generation. Higher values mean more random completions.
    max_output_tokens (int): The maximum number of tokens to generate.
    top_p (float): Nucleus sampling parameter controlling the size of the probability mass to consider for token generation.
    top_k (int): Truncates the number of tokens considered based on their probability.

    Returns:
    dict: The response from the API.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": system_prompt}],
            },
            {
                "role": "model",
                "parts": [{"text": "ok"}],
            },
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            },
        ],
        "generationConfig": {
            "stopSequences": ["Title"],
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": top_p,
        },
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url, headers=headers, json=payload, params={"key": GEMINI_API_KEY}
            ) as response:
                response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"An error occurred: {e}")
            return None


async def __generate_response_async(
    model_name,
    user_prompt,
    system_prompt,
    temperature,
    max_tokens,
    top_p,
    full_response=False,
):
    """
    Generates a response from the generative language model.

    Parameters:
    model_name (str): Name of the model to use for generation.
    user_prompt (str): The text prompt for content generation.
    system_prompt (str): The system-generated text prompt.
    temperature (float): Controls randomness in generation.
    max_tokens (int): Maximum number of tokens to generate.
    top_p (float): Nucleus sampling parameter.
    full_response (bool): If True, returns the full API response, else returns the generated text.

    Returns:
    str or dict: The generated text or the full response from the API, based on the 'full_response' flag.
    """
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")

    if not system_prompt:
        system_prompt = "You are a helpful AI Assitant"

    if not model_name or not isinstance(model_name, str):
        raise ValueError("model must be a non-empty string.")

    for attempt in range(MAX_RETRIES):
        try:
            response = await __call_gemeni_async(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            if full_response:
                return response
            else:
                return response["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(
                    f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}"
                )
                return None


async def generate_text_async(
    model_name,
    user_prompt,
    system_prompt="You are a helpful AI Assistant",
    temperature=0.7,
    max_tokens=2024,
    top_p=0.8,
):
    """
    Generates using Gemini and returns only the text.
    """
    result = await __generate_response_async(
        model_name,
        user_prompt,
        system_prompt,
        temperature,
        max_tokens,
        top_p,
        full_response=False,
    )
    return result


async def generate_full_response_async(
    model_name,
    user_prompt,
    system_prompt="You are a helpful AI Assistant",
    max_tokens=2000,
    top_p=1.0,
    temperature=0.7,
):
    """
    Generates the full response from Gemini.
    """
    result = await __generate_response_async(
        model_name,
        user_prompt,
        system_prompt,
        temperature,
        max_tokens,
        top_p,
        full_response=True,
    )
    return result
