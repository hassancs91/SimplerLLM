import time
from typing import Type, Union, Tuple
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_providers.llm_response_models import LLMFullResponse
import asyncio
import tiktoken

from SimplerLLM.tools.json_helpers import (
    extract_json_from_text,
    convert_json_to_pydantic_model,
    validate_json_with_pydantic_model,
    generate_json_example_from_pydantic,
)


def calculate_text_generation_costs(input: str, response: str, cost_per_million_input_tokens: float, cost_per_million_output_tokens: float, approximate: bool = True):
    def count_tokens(text: str) -> int:
        if approximate:
            return len(text) // 4
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    
    input_tokens = count_tokens(input)
    output_tokens = count_tokens(response)
    
    input_cost = (input_tokens / 1_000_000) * cost_per_million_input_tokens
    output_cost = (output_tokens / 1_000_000) * cost_per_million_output_tokens
    
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def create_optimized_prompt(prompt: str, model_class: Type[BaseModel], custom_prompt_suffix: str = None) -> str:
    """
    Creates an optimized prompt by combining the base prompt with JSON model example.
    
    :param prompt: The base prompt text
    :param model_class: The Pydantic model class to generate JSON example from
    :param custom_prompt_suffix: Optional custom suffix to override default format
    :return: The optimized prompt string
    """
    json_model = generate_json_example_from_pydantic(model_class)
    return custom_prompt_suffix or (prompt + f"\n\nThe response should be in a structured JSON format for machine processing that matches the following JSON: {json_model}\n\nOUTPUT: {{")

def generate_pydantic_json_model(
    model_class: Type[BaseModel],
    prompt: str,
    llm_instance: LLM,
    max_retries: int = 3,
    max_tokens: int = 4096,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False,
) -> Union[BaseModel, Tuple[BaseModel, LLMFullResponse]]:
    """
    Generates a model instance based on a given prompt, retrying on validation errors.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param llm_instance: Instance of a large language model.
    :param max_retries: Maximum number of retries on validation errors.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.

    :return: BaseModel object if successful, otherwise error message.
    """
    # Create optimized prompt and calculate exponential backoff before the loop
    optimized_prompt = create_optimized_prompt(prompt, model_class, custom_prompt_suffix)
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            ai_response = llm_instance.generate_response(
                prompt=optimized_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                json_mode=True,
                full_response=full_response
            )

            response_text = ai_response.generated_text if full_response else ai_response

            if response_text:
                json_object = extract_json_from_text(response_text)

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    return (model_object, ai_response) if full_response else model_object

        except Exception as e:  # Replace SpecificException with the appropriate exception
            return f"Exception occurred: {e}"

        # Retry logic for empty AI response or validation errors
        if not ai_response or (errors and attempt < max_retries):
            time.sleep(delay)
        elif errors:
            return f"Validation failed after {max_retries} retries: {errors}"

    return "Maximum retries exceeded without successful validation."

def generate_pydantic_json_model_reliable(
    model_class: Type[BaseModel],
    prompt: str,
    reliable_llm: ReliableLLM,
    max_retries: int = 3,
    max_tokens: int = 4096,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False,
) -> Union[Tuple[BaseModel, LLMProvider], Tuple[BaseModel, LLMFullResponse, LLMProvider]]:
    """
    Generates a model instance using ReliableLLM with fallback capability.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers.
    :param max_retries: Maximum number of retries on validation errors.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.

    :return: BaseModel object if successful, otherwise error message.
    """
    optimized_prompt = create_optimized_prompt(prompt, model_class, custom_prompt_suffix)
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            result = reliable_llm.generate_response(
                return_provider=True,
                prompt=optimized_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                json_mode=True,
                full_response=full_response
            )
            
            if full_response:
                ai_response, provider = result
                response_text = ai_response.generated_text
            else:
                response_text, provider = result

            if response_text:
                json_object = extract_json_from_text(response_text)

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    return (model_object, ai_response, provider) if full_response else (model_object, provider)

        except Exception as e:
            return f"Exception occurred: {e}"

        if not ai_response or (errors and attempt < max_retries):
            time.sleep(delay)
        elif errors:
            return f"Validation failed after {max_retries} retries: {errors}"

    return "Maximum retries exceeded without successful validation."

async def generate_pydantic_json_model_reliable_async(
    model_class: Type[BaseModel],
    prompt: str,
    reliable_llm: ReliableLLM,
    max_retries: int = 3,
    max_tokens: int = 4096,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False,
) -> Union[Tuple[BaseModel, LLMProvider], Tuple[BaseModel, LLMFullResponse, LLMProvider]]:
    """
    Asynchronously generates a model instance using ReliableLLM with fallback capability.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers.
    :param max_retries: Maximum number of retries on validation errors.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.

    :return: BaseModel object if successful, otherwise error message.
    """
    optimized_prompt = create_optimized_prompt(prompt, model_class, custom_prompt_suffix)
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            result = await reliable_llm.generate_response_async(
                return_provider=True,
                prompt=optimized_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                json_mode=True,
                full_response=full_response
            )
            
            if full_response:
                ai_response, provider = result
                response_text = ai_response.generated_text
            else:
                response_text, provider = result

            if response_text:
                json_object = extract_json_from_text(response_text)

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    return (model_object, ai_response, provider) if full_response else (model_object, provider)

        except Exception as e:
            return f"Exception occurred: {e}"

        if not ai_response or (errors and attempt < max_retries):
            await asyncio.sleep(delay)
        elif errors:
            return f"Validation failed after {max_retries} retries: {errors}"

    return "Maximum retries exceeded without successful validation."

async def generate_pydantic_json_model_async(
    model_class: Type[BaseModel],
    prompt: str,
    llm_instance: LLM,
    max_retries: int = 3,
    max_tokens: int = 4096,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False,
) -> Union[BaseModel, Tuple[BaseModel, LLMFullResponse]]:
    """
    Generates a model instance based on a given prompt, retrying on validation errors.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param llm_instance: Instance of a large language model.
    :param max_retries: Maximum number of retries on validation errors.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.

    :return: BaseModel object if successful, otherwise error message.
    """
    # Create optimized prompt and calculate exponential backoff before the loop
    optimized_prompt = create_optimized_prompt(prompt, model_class, custom_prompt_suffix)
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            ai_response = await llm_instance.generate_response_async(
                prompt=optimized_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                json_mode=True,
                full_response=full_response
            )

            response_text = ai_response.generated_text if full_response else ai_response

            if response_text:
                json_object = extract_json_from_text(response_text)

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    return (model_object, ai_response) if full_response else model_object

        except Exception as e:  # Replace SpecificException with the appropriate exception
            return f"Exception occurred: {e}"

        # Retry logic for empty AI response or validation errors
        if not ai_response or (errors and attempt < max_retries):
            await asyncio.sleep(delay)
        elif errors:
            return f"Validation failed after {max_retries} retries: {errors}"

    return "Maximum retries exceeded without successful validation."
