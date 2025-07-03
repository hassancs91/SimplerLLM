from openai import AsyncOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os
import time
from .llm_response_models import LLMFullResponse,LLMEmbeddingsResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def generate_response(
    model_name,
    messages=None,
    temperature=0.7,
    max_tokens=300,
    top_p=1.0,
    full_response=False,
    api_key=None,
    json_mode=False,
    site_url=None,
    site_name=None,
):
    start_time = time.time() if full_response else None
    
    # Configure OpenAI client with OpenRouter base URL
    openrouter_client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            
            if json_mode:
                params["response_format"] = {"type": "json_object"}
                
            # Add OpenRouter-specific headers for site tracking
            extra_headers = {}
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if site_name:
                extra_headers["X-Title"] = site_name
                
            completion = openrouter_client.chat.completions.create(
                extra_headers=extra_headers,
                **params
            )
            generated_text = completion.choices[0].message.content

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=completion.usage.prompt_tokens,
                    output_token_count=completion.usage.completion_tokens,
                    llm_provider_response=completion,
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
    site_url=None,
    site_name=None,
):
    start_time = time.time() if full_response else None
    
    # Configure async OpenAI client with OpenRouter base URL
    async_openrouter_client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )
   
    for attempt in range(MAX_RETRIES):
        try:
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            
            if json_mode:
                params["response_format"] = {"type": "json_object"}
                
            # Add OpenRouter-specific headers for site tracking
            extra_headers = {}
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if site_name:
                extra_headers["X-Title"] = site_name
                
            completion = await async_openrouter_client.chat.completions.create(
                extra_headers=extra_headers,
                **params
            )
            generated_text = completion.choices[0].message.content

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=completion.usage.prompt_tokens,
                    output_token_count=completion.usage.completion_tokens,
                    llm_provider_response=completion,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


def generate_embeddings(
    model_name,
    user_input=None,
    full_response=False,
    api_key=None
):
    
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None

    # Configure OpenAI client with OpenRouter base URL
    openrouter_client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )

    for attempt in range(MAX_RETRIES):
        try:
            
            response = openrouter_client.embeddings.create(
                model=model_name,
                input=user_input
            )
            
            # Extract actual embedding vectors from the response
            embeddings = [item.embedding for item in response.data]
            
            # For single input, return single embedding; for multiple inputs, return list
            if isinstance(user_input, str):
                generate_embeddings = embeddings[0] if embeddings else []
            else:
                generate_embeddings = embeddings

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=generate_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=response,
                )
            return generate_embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_embeddings_async(
    model_name,
    user_input=None,
    full_response=False,
    api_key=None,
):
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None
    
    # Configure async OpenAI client with OpenRouter base URL
    async_openrouter_client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            result = await async_openrouter_client.embeddings.create(
                model=model_name,
                input=user_input,
            )
            
            # Extract actual embedding vectors from the response
            embeddings = [item.embedding for item in result.data]
            
            # For single input, return single embedding; for multiple inputs, return list
            if isinstance(user_input, str):
                generate_embeddings = embeddings[0] if embeddings else []
            else:
                generate_embeddings = embeddings

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=generate_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=result,
                )
            return generate_embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)