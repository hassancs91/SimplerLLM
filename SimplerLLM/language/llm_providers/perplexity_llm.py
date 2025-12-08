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
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def generate_response(
    model_name: str,
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key=None,
    json_mode: bool = False,
    search_domain_filter: list = None,
    search_recency_filter: str = None,
    return_images: bool = False,
    return_related_questions: bool = False,
) -> Optional[Dict]:
    """
    Generate a response using Perplexity's Chat Completions API.

    Perplexity has built-in web search by default - no extra parameter needed.
    Citations are returned in the search_results field.

    Args:
        model_name: The model to use (e.g., 'sonar', 'sonar-pro', 'sonar-reasoning')
        messages: List of message dictionaries
        temperature: Controls randomness (0-2, default 0.7)
        max_tokens: Maximum tokens for the response
        top_p: Nucleus sampling threshold
        full_response: If True, returns LLMFullResponse with web_sources
        api_key: Perplexity API key
        json_mode: If True, enables JSON output mode
        search_domain_filter: List of domains to include/exclude (prefix with "-" to exclude)
        search_recency_filter: Filter by time ("day", "week", "month")
        return_images: Include images in results
        return_related_questions: Return related queries

    Returns:
        str or LLMFullResponse: Generated text or full response with web sources
    """
    start_time = time.time() if full_response else None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    # Add optional Perplexity-specific parameters
    if search_domain_filter:
        payload["search_domain_filter"] = search_domain_filter
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter
    if return_images:
        payload["return_images"] = return_images
    if return_related_questions:
        payload["return_related_questions"] = return_related_questions

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                PERPLEXITY_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            generated_text = result["choices"][0]["message"]["content"]

            # Extract web sources from search_results
            web_sources = []
            if "search_results" in result:
                for source in result["search_results"]:
                    web_sources.append({
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "date": source.get("date", ""),
                    })

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=result.get("usage", {}).get("prompt_tokens", 0),
                    output_token_count=result.get("usage", {}).get("completion_tokens", 0),
                    llm_provider_response=result,
                    web_sources=web_sources if web_sources else None,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_response_async(
    model_name: str,
    messages=None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    top_p: float = 1.0,
    full_response: bool = False,
    api_key=None,
    json_mode: bool = False,
    search_domain_filter: list = None,
    search_recency_filter: str = None,
    return_images: bool = False,
    return_related_questions: bool = False,
) -> Optional[Dict]:
    """
    Asynchronously generate a response using Perplexity's Chat Completions API.

    Perplexity has built-in web search by default - no extra parameter needed.
    Citations are returned in the search_results field.

    Args:
        model_name: The model to use (e.g., 'sonar', 'sonar-pro', 'sonar-reasoning')
        messages: List of message dictionaries
        temperature: Controls randomness (0-2, default 0.7)
        max_tokens: Maximum tokens for the response
        top_p: Nucleus sampling threshold
        full_response: If True, returns LLMFullResponse with web_sources
        api_key: Perplexity API key
        json_mode: If True, enables JSON output mode
        search_domain_filter: List of domains to include/exclude (prefix with "-" to exclude)
        search_recency_filter: Filter by time ("day", "week", "month")
        return_images: Include images in results
        return_related_questions: Return related queries

    Returns:
        str or LLMFullResponse: Generated text or full response with web sources
    """
    start_time = time.time() if full_response else None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    # Add optional Perplexity-specific parameters
    if search_domain_filter:
        payload["search_domain_filter"] = search_domain_filter
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter
    if return_images:
        payload["return_images"] = return_images
    if return_related_questions:
        payload["return_related_questions"] = return_related_questions

    async with aiohttp.ClientSession() as session:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    PERPLEXITY_API_URL,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    generated_text = result["choices"][0]["message"]["content"]

                    # Extract web sources from search_results
                    web_sources = []
                    if "search_results" in result:
                        for source in result["search_results"]:
                            web_sources.append({
                                "title": source.get("title", ""),
                                "url": source.get("url", ""),
                                "date": source.get("date", ""),
                            })

                    if full_response:
                        end_time = time.time()
                        process_time = end_time - start_time
                        return LLMFullResponse(
                            generated_text=generated_text,
                            model=model_name,
                            process_time=process_time,
                            input_token_count=result.get("usage", {}).get("prompt_tokens", 0),
                            output_token_count=result.get("usage", {}).get("completion_tokens", 0),
                            llm_provider_response=result,
                            web_sources=web_sources if web_sources else None,
                        )
                    return generated_text

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))
                else:
                    error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                    raise Exception(error_msg)
