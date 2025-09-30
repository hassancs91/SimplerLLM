# add streaming
from openai import AsyncOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os
import time
import json
from .llm_response_models import LLMFullResponse,LLMEmbeddingsResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))
DEBUG_GPT5 = os.getenv("DEBUG_GPT5", "false").lower() == "true"


def generate_response(
    model_name,
    messages=None,
    temperature=0.7,
    max_tokens=300,
    top_p=1.0,
    full_response=False,
    api_key = None,
    json_mode=False,
):
    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key)

    # Check if it's a GPT-5 model
    is_gpt5 = "gpt-5" in model_name.lower()

    for attempt in range(MAX_RETRIES):
        try:
            params = {
                "model": model_name,
                "messages": messages,
                "top_p": top_p,
            }

            # Check if it's a GPT-5 model and apply specific settings
            if is_gpt5:
                # Smart default: Use 4000 tokens for GPT-5 if user hasn't changed from default
                # This ensures enough tokens for both reasoning and output
                if max_tokens == 300:  # Function's default parameter value
                    actual_max_tokens = 4000
                else:
                    actual_max_tokens = max_tokens  # Respect user's explicit choice

                params["max_completion_tokens"] = actual_max_tokens

                # GPT-5 models only support temperature=1 (default)
                if temperature != 1:
                    params["temperature"] = 1
                else:
                    params["temperature"] = temperature
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            if json_mode:
                params["response_format"] = {"type": "json_object"}

            completion = openai_client.chat.completions.create(**params)

            # Extract text from response
            generated_text = completion.choices[0].message.content

            # Handle empty GPT-5 responses with reasoning information
            if is_gpt5 and (generated_text is None or generated_text == ""):
                # Get reasoning token information
                reasoning_tokens = 0
                output_tokens = 0

                if hasattr(completion.usage, 'completion_tokens_details'):
                    details = completion.usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        reasoning_tokens = details.reasoning_tokens

                    # Total completion tokens
                    output_tokens = completion.usage.completion_tokens

                # Check if it's due to reasoning consuming all tokens
                if reasoning_tokens > 0 and completion.choices[0].finish_reason == "length":
                    # Return informative message about reasoning token consumption
                    info_msg = f"[GPT-5 Notice] Model used all {reasoning_tokens} tokens for internal reasoning. "
                    info_msg += f"No tokens remained for output text (finish_reason: length). "
                    info_msg += f"To get a response, increase max_tokens beyond {reasoning_tokens}. "
                    info_msg += f"Suggested minimum: {reasoning_tokens + 200} tokens."

                    # If DEBUG mode is on, show more details
                    if DEBUG_GPT5:
                        print(f"\n[DEBUG] GPT-5 Token Usage Details:")
                        print(f"  - Reasoning tokens: {reasoning_tokens}")
                        print(f"  - Total completion tokens: {output_tokens}")
                        print(f"  - Max tokens allowed: {params['max_completion_tokens']}")
                        print(f"  - Finish reason: {completion.choices[0].finish_reason}")

                    # Return the informative message instead of empty content
                    generated_text = info_msg
                else:
                    # Some other issue caused empty content
                    generated_text = f"[GPT-5 Notice] Received empty response. Finish reason: {completion.choices[0].finish_reason}"

            # If we got a response (or reasoning info for GPT-5), proceed
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=completion.usage.prompt_tokens if completion.usage else 0,
                    output_token_count=completion.usage.completion_tokens if completion.usage else 0,
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
    api_key = None,
    json_mode=False,
):
    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key)

    # Check if it's a GPT-5 model
    is_gpt5 = "gpt-5" in model_name.lower()

    for attempt in range(MAX_RETRIES):
        try:
            params = {
                "model": model_name,
                "messages": messages,
                "top_p": top_p,
            }

            # Check if it's a GPT-5 model and apply specific settings
            if is_gpt5:
                # Smart default: Use 4000 tokens for GPT-5 if user hasn't changed from default
                # This ensures enough tokens for both reasoning and output
                if max_tokens == 300:  # Function's default parameter value
                    actual_max_tokens = 4000
                else:
                    actual_max_tokens = max_tokens  # Respect user's explicit choice

                params["max_completion_tokens"] = actual_max_tokens

                # GPT-5 models only support temperature=1 (default)
                if temperature != 1:
                    params["temperature"] = 1
                else:
                    params["temperature"] = temperature
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            if json_mode:
                params["response_format"] = {"type": "json_object"}

            completion = await async_openai_client.chat.completions.create(**params)

            # Extract text from response
            generated_text = completion.choices[0].message.content

            # Handle empty GPT-5 responses with reasoning information
            if is_gpt5 and (generated_text is None or generated_text == ""):
                # Get reasoning token information
                reasoning_tokens = 0
                output_tokens = 0

                if hasattr(completion.usage, 'completion_tokens_details'):
                    details = completion.usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        reasoning_tokens = details.reasoning_tokens

                    # Total completion tokens
                    output_tokens = completion.usage.completion_tokens

                # Check if it's due to reasoning consuming all tokens
                if reasoning_tokens > 0 and completion.choices[0].finish_reason == "length":
                    # Return informative message about reasoning token consumption
                    info_msg = f"[GPT-5 Notice] Model used all {reasoning_tokens} tokens for internal reasoning. "
                    info_msg += f"No tokens remained for output text (finish_reason: length). "
                    info_msg += f"To get a response, increase max_tokens beyond {reasoning_tokens}. "
                    info_msg += f"Suggested minimum: {reasoning_tokens + 200} tokens."

                    # If DEBUG mode is on, show more details
                    if DEBUG_GPT5:
                        print(f"\n[DEBUG] GPT-5 Token Usage Details:")
                        print(f"  - Reasoning tokens: {reasoning_tokens}")
                        print(f"  - Total completion tokens: {output_tokens}")
                        print(f"  - Max tokens allowed: {params['max_completion_tokens']}")
                        print(f"  - Finish reason: {completion.choices[0].finish_reason}")

                    # Return the informative message instead of empty content
                    generated_text = info_msg
                else:
                    # Some other issue caused empty content
                    generated_text = f"[GPT-5 Notice] Received empty response. Finish reason: {completion.choices[0].finish_reason}"

            # If we got a response (or reasoning info for GPT-5), proceed
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    input_token_count=completion.usage.prompt_tokens if completion.usage else 0,
                    output_token_count=completion.usage.completion_tokens if completion.usage else 0,
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
    full_response = False,
    api_key = None
):
    
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None

    openai_client = OpenAI(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            
            response = openai_client.embeddings.create(
                model= model_name,
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
    full_response = False,
    api_key = None,
):
    async_openai_client = AsyncOpenAI(api_key=api_key)
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None
    for attempt in range(MAX_RETRIES):
        try:
            result = await async_openai_client.embeddings.create(
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
