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
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False,
) -> Union[BaseModel, LLMFullResponse, str]:
    """
    Generates a model instance based on a given prompt, retrying on validation errors.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param llm_instance: Instance of a large language model.
    :param max_retries: Maximum number of retries on validation errors.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output. Lower values make output more deterministic.
    :param top_p: Controls diversity of output. Lower values make output more focused.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.
    :param system_prompt: System prompt to set the context for the LLM.
    :param full_response: If True, returns the full API response including token counts.

    :return: 
        - If full_response=False: BaseModel object
        - If full_response=True: LLMFullResponse object with model_object attribute and input_token_count and output_token_count
        - Error message string if unsuccessful
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
                temperature=temperature,
                top_p=top_p,
                json_mode=True,
                full_response=full_response
            )

            response_text = ai_response.generated_text if full_response else ai_response

            if response_text:
                json_object = extract_json_from_text(response_text)
                
                # Check if json_object is None (no valid JSON found)
                if json_object is None:
                    if attempt < max_retries:
                        continue  # Try again
                    else:
                        return f"No valid JSON found in response after {max_retries} attempts"

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    if full_response:
                        ai_response.model_object = model_object
                        return ai_response
                    else:
                        return model_object

        except Exception as e:  # Replace SpecificException with the appropriate exception
            return f"Exception occurred: {e}"

        # Retry logic for empty AI response or validation errors
        if (not response_text or errors) and attempt < max_retries:
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
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False,
) -> Union[Tuple[BaseModel, LLMProvider, str], LLMFullResponse, str]:
    """
    Generates a model instance using ReliableLLM with fallback capability.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers.
    :param max_retries: Maximum number of retries on validation errors.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output. Lower values make output more deterministic.
    :param top_p: Controls diversity of output. Lower values make output more focused.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.
    :param system_prompt: System prompt to set the context for the LLM.
    :param full_response: If True, returns the full API response including token counts.

    :return: 
        - If full_response=False: Tuple of (model_object, provider, model_name)
        - If full_response=True: LLMFullResponse object with model_object, provider, and model_name attributes, and input_token_count and output_token_count
        - Error message string if unsuccessful
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
                temperature=temperature,
                top_p=top_p,
                json_mode=True,
                full_response=full_response
            )
            
            if full_response:
                ai_response, provider, model_name = result
                response_text = ai_response.generated_text
            else:
                response_text, provider, model_name = result

            if response_text:
                json_object = extract_json_from_text(response_text)
                
                # Check if json_object is None (no valid JSON found)
                if json_object is None:
                    if attempt < max_retries:
                        continue  # Try again
                    else:
                        return f"No valid JSON found in response after {max_retries} attempts"

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    if full_response:
                        ai_response.model_object = model_object
                        ai_response.provider = provider
                        ai_response.model_name = model_name
                        return ai_response
                    else:
                        return (model_object, provider, model_name)

        except Exception as e:
            return f"Exception occurred: {e}"

        if (not response_text or errors) and attempt < max_retries:
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
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False,
) -> Union[Tuple[BaseModel, LLMProvider, str], LLMFullResponse, str]:
    """
    Asynchronously generates a model instance using ReliableLLM with fallback capability.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers.
    :param max_retries: Maximum number of retries on validation errors.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output. Lower values make output more deterministic.
    :param top_p: Controls diversity of output. Lower values make output more focused.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.
    :param system_prompt: System prompt to set the context for the LLM.
    :param full_response: If True, returns the full API response including token counts.

    :return: 
        - If full_response=False: Tuple of (model_object, provider, model_name)
        - If full_response=True: LLMFullResponse object with model_object, provider, and model_name attributes, and input_token_count and output_token_count
        - Error message string if unsuccessful
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
                temperature=temperature,
                top_p=top_p,
                json_mode=True,
                full_response=full_response
            )
            
            if full_response:
                ai_response, provider, model_name = result
                response_text = ai_response.generated_text
            else:
                response_text, provider, model_name = result

            if response_text:
                json_object = extract_json_from_text(response_text)
                
                # Check if json_object is None (no valid JSON found)
                if json_object is None:
                    if attempt < max_retries:
                        continue  # Try again
                    else:
                        return f"No valid JSON found in response after {max_retries} attempts"

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    if full_response:
                        ai_response.model_object = model_object
                        ai_response.provider = provider
                        ai_response.model_name = model_name
                        return ai_response
                    else:
                        return (model_object, provider, model_name)

        except Exception as e:
            return f"Exception occurred: {e}"

        if (not response_text or errors) and attempt < max_retries:
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
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = "The Output is a VALID Structured JSON",
    full_response: bool = False,
) -> Union[BaseModel, LLMFullResponse, str]:
    """
    Asynchronously generates a model instance based on a given prompt, retrying on validation errors.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param llm_instance: Instance of a large language model.
    :param max_retries: Maximum number of retries on validation errors.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output. Lower values make output more deterministic.
    :param top_p: Controls diversity of output. Lower values make output more focused.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.
    :param system_prompt: System prompt to set the context for the LLM.
    :param full_response: If True, returns the full API response including token counts.

    :return: 
        - If full_response=False: BaseModel object
        - If full_response=True: LLMFullResponse object with model_object attribute and input_token_count and output_token_count
        - Error message string if unsuccessful
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
                temperature=temperature,
                top_p=top_p,
                json_mode=True,
                full_response=full_response
            )

            response_text = ai_response.generated_text if full_response else ai_response

            if response_text:
                json_object = extract_json_from_text(response_text)
                
                # Check if json_object is None (no valid JSON found)
                if json_object is None:
                    if attempt < max_retries:
                        continue  # Try again
                    else:
                        return f"No valid JSON found in response after {max_retries} attempts"

                validated, errors = validate_json_with_pydantic_model(
                    model_class, json_object
                )

                if not errors:
                    model_object = convert_json_to_pydantic_model(
                        model_class, json_object[0]
                    )
                    if full_response:
                        ai_response.model_object = model_object
                        return ai_response
                    else:
                        return model_object

        except Exception as e:  # Replace SpecificException with the appropriate exception
            return f"Exception occurred: {e}"

        # Retry logic for empty AI response or validation errors
        if (not response_text or errors) and attempt < max_retries:
            await asyncio.sleep(delay)
        elif errors:
            return f"Validation failed after {max_retries} retries: {errors}"

    return "Maximum retries exceeded without successful validation."
