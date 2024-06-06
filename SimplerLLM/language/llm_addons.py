import time
from typing import Type
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM
import asyncio

from SimplerLLM.tools.json_helpers import (
    extract_json_from_text,
    convert_json_to_pydantic_model,
    validate_json_with_pydantic_model,
    generate_json_example_from_pydantic,
)



def generate_pydantic_json_model(
    model_class: Type[BaseModel],
    prompt: str,
    llm_instance: LLM,
    max_retries: int = 3,
    max_tokens: int = 4096,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
) -> BaseModel:
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
    # Concatenate prompt and JSON model outside the loop
    json_model = generate_json_example_from_pydantic(model_class)
    optimized_prompt = custom_prompt_suffix or (prompt + f"\n\nThe response should be in a structured JSON format that matches the following JSON: {json_model}")
    
    # Calculate exponential backoff before the loop
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            ai_response = llm_instance.generate_response(prompt=optimized_prompt, max_tokens = max_tokens)

            if ai_response:
                json_object = extract_json_from_text(ai_response)

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    return model_object

        except Exception as e:  # Replace SpecificException with the appropriate exception
            return f"Exception occurred: {e}"

        # Retry logic for empty AI response or validation errors
        if not ai_response or (errors and attempt < max_retries):
            time.sleep(delay)
        elif errors:
            return f"Validation failed after {max_retries} retries: {errors}"

    return "Maximum retries exceeded without successful validation."

async def generate_pydantic_json_model_async(
    model_class: Type[BaseModel],
    prompt: str,
    llm_instance: LLM,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
) -> BaseModel:
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
    # Concatenate prompt and JSON model outside the loop
    json_model = generate_json_example_from_pydantic(model_class)
    optimized_prompt = custom_prompt_suffix or (prompt + f"\n\nThe response should be in a structured JSON format that matches the following JSON: {json_model}")
    
    # Calculate exponential backoff before the loop
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            ai_response = await llm_instance.generate_response_async(prompt=optimized_prompt)

            if ai_response:
                json_object = extract_json_from_text(ai_response)

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    return model_object

        except Exception as e:  # Replace SpecificException with the appropriate exception
            return f"Exception occurred: {e}"

        # Retry logic for empty AI response or validation errors
        if not ai_response or (errors and attempt < max_retries):
            await asyncio.sleep(delay)
        elif errors:
            return f"Validation failed after {max_retries} retries: {errors}"

    return "Maximum retries exceeded without successful validation."







