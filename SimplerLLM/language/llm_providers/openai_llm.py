# add streaming
from openai import AsyncOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os
import time
from .llm_response_models import LLMFullResponse

# Load environment variables
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


# Initialize the OpenAI clients
def initialize_openai_clients():
    async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return async_openai_client, openai_client


async_openai_client, openai_client = initialize_openai_clients()


def generate_response(
    model_name,
    prompt=None,
    messages=None,
    system_prompt="You are a helpful AI Assistant",
    temperature=0.7,
    max_tokens=300,
    top_p=1.0,
    full_response=False,
):
    start_time = time.time() if full_response else None

    # Validate inputs
    if prompt and messages:
        raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
    if not prompt and not messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided.")

    # Prepare messages based on input type
    if prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    for attempt in range(MAX_RETRIES):
        try:
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            generated_text = completion.choices[0].message.content

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=completion,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts"
                if full_response:
                    end_time = time.time()
                    process_time = end_time - start_time
                    error_msg += f" and {process_time} seconds"
                error_msg += f" due to: {e}"
                print(error_msg)
                return None


async def generate_response_async(
    model_name,
    prompt=None,
    messages=None,
    system_prompt="You are a helpful AI Assistant",
    temperature=0.7,
    max_tokens=300,
    top_p=1.0,
    full_response=False,
):
    start_time = time.time() if full_response else None

    # Validate inputs
    if prompt and messages:
        raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
    if not prompt and not messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided.")

    # Prepare messages based on input type
    if prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            generated_text = completion.choices[0].message.content

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=completion,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts"
                if full_response:
                    end_time = time.time()
                    process_time = end_time - start_time
                    error_msg += f" and {process_time} seconds"
                error_msg += f" due to: {e}"
                print(error_msg)
                return None


def generate_embeddings(
    model_name,
    user_input=None,
):

    if not user_input:
        raise ValueError("user_input must be provided.")

    # Prepare messages based on input type
    if prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    for attempt in range(MAX_RETRIES):
        try:
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            generated_text = completion.choices[0].message.content

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=completion,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts"
                if full_response:
                    end_time = time.time()
                    process_time = end_time - start_time
                    error_msg += f" and {process_time} seconds"
                error_msg += f" due to: {e}"
                print(error_msg)
                return None
