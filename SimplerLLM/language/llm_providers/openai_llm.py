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


def generate_text(
    model_name,
    user_prompt,
    system_prompt="You are a helpful AI Assistant",
    temperature=0.7,
    max_tokens=2024,
    top_p=0.8,
) -> str:
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return [completion.choices[0].message.content][0]

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
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
) -> str:
    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return [completion.choices[0].message.content][0]

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                # Log the error or inform the user
                print(
                    f"Failed to generate response after {MAX_RETRIES} attempts due to: {e}"
                )
                return None


def generate_full_response(
    model_name,
    user_prompt,
    system_prompt="You are a helpful AI Assistant",
    temperature=0.7,
    max_tokens=2024,
    top_p=0.8,
) -> LLMFullResponse:

    start_time = time.time()  # Record the start time

    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            end_time = time.time()  # Record the end time before returning
            process_time = end_time - start_time

            full_reponse = LLMFullResponse(
                generated_text=[completion.choices[0].message.content][0],
                model=model_name,
                process_time=process_time,
                llm_provider_response=completion,
            )

            return full_reponse

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:

                end_time = time.time()  # Record the end time in case of failure
                process_time = end_time - start_time
                # Log the error or inform the user
                print(
                    f"Failed to generate response after {MAX_RETRIES} attempts and {process_time} seconds due to: {e}"
                )
                return None


async def generate_full_response_async(
    model_name,
    user_prompt,
    system_prompt="You are a helpful AI Assistant",
    temperature=0.7,
    max_tokens=2024,
    top_p=0.8,
) -> LLMFullResponse:
    start_time = time.time()  # Record the start time

    if not user_prompt or not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a non-empty string.")
    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            end_time = time.time()  # Record the end time before returning
            process_time = end_time - start_time

            full_reponse = LLMFullResponse(
                generated_text=[completion.choices[0].message.content][0],
                model=model_name,
                process_time=process_time,
                llm_provider_response=completion,
            )

            return full_reponse

        except Exception as e:  # Consider catching more specific exceptions
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                end_time = time.time()  # Record the end time in case of failure
                process_time = end_time - start_time
                # Log the error or inform the user
                print(
                    f"Failed to generate response after {MAX_RETRIES} attempts and {process_time} seconds due to: {e}"
                )
                return None
