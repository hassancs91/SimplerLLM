"""
JSON/Pydantic model generation from LLM responses.

This module provides functions for generating structured JSON output from LLMs
with automatic Pydantic model validation, retry logic, and fallback support.

Features:
    - Automatic JSON extraction from LLM responses
    - Pydantic model validation with retries
    - Exponential backoff for reliability
    - Support for vision (images) and web search
    - Sync and async variants
    - ReliableLLM fallback support

Example:
    >>> from pydantic import BaseModel
    >>> from SimplerLLM.language import LLM, LLMProvider
    >>> from SimplerLLM.language.llm_addons import generate_pydantic_json_model
    >>>
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    ...     city: str
    >>>
    >>> llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    >>> result = generate_pydantic_json_model(
    ...     model_class=Person,
    ...     prompt="Generate a fictional person",
    ...     llm_instance=llm
    ... )
    >>> print(result.name, result.age, result.city)
"""

import time
import asyncio
from typing import Type, Union, Tuple, Optional, Literal

from pydantic import BaseModel

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_providers.llm_response_models import LLMFullResponse

from SimplerLLM.tools.json_helpers import (
    extract_json_from_text,
    convert_json_to_pydantic_model,
    validate_json_with_pydantic_model,
    generate_json_example_from_pydantic,
    extract_schema_constraints,
    HAS_ROOT_MODEL,
    RootModel,
)


def _unwrap_rootmodel_list(json_data: list, model_class: Type[BaseModel]) -> list:
    """
    Handle RootModel list unwrapping.

    When a model is a RootModel expecting a list, but the LLM returns an object
    with a single key containing the list, extract and use that list.

    Args:
        json_data: The extracted JSON data (list of dicts)
        model_class: The Pydantic model class

    Returns:
        The potentially unwrapped JSON data
    """
    if not HAS_ROOT_MODEL or RootModel is None:
        return json_data

    try:
        if not issubclass(model_class, RootModel):
            return json_data
    except TypeError:
        return json_data

    # If we have a RootModel, check if the JSON is an object with a single list value
    result = []
    for item in json_data:
        if isinstance(item, dict) and len(item) == 1:
            # Get the single value
            single_value = list(item.values())[0]
            if isinstance(single_value, list):
                # The LLM wrapped the list in an object, unwrap it
                result.append(single_value)
            else:
                result.append(item)
        elif isinstance(item, list):
            # Already a list, use as-is
            result.append(item)
        else:
            result.append(item)

    return result


def _validate_reasoning_params(
    thinking_budget: Optional[int],
    max_tokens: int,
) -> None:
    """
    Validate reasoning/thinking parameters.

    Args:
        thinking_budget: Token budget for extended thinking (Anthropic/Gemini)
        max_tokens: Maximum tokens for response

    Raises:
        ValueError: If thinking_budget is invalid (< 1024 or >= max_tokens)
    """
    if thinking_budget is not None:
        if thinking_budget < 1024:
            raise ValueError(
                f"thinking_budget must be at least 1024 tokens (got {thinking_budget})"
            )
        if thinking_budget >= max_tokens:
            raise ValueError(
                f"thinking_budget ({thinking_budget}) must be less than max_tokens ({max_tokens}). "
                f"Set max_tokens to at least {thinking_budget + 2000} for adequate output space."
            )


def create_optimized_prompt(
    prompt: str,
    model_class: Type[BaseModel],
    custom_prompt_suffix: str = None
) -> str:
    """
    Creates an optimized prompt by combining the base prompt with JSON model example.

    :param prompt: The base prompt text.
    :param model_class: The Pydantic model class to generate JSON example from.
    :param custom_prompt_suffix: Optional custom suffix to override default format.

    :return: The optimized prompt string with JSON format instructions.

    Example:
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> prompt = create_optimized_prompt("Generate a person", Person)
        >>> print(prompt)  # Includes JSON format instructions
    """
    if custom_prompt_suffix:
        return custom_prompt_suffix

    json_model = generate_json_example_from_pydantic(model_class)
    constraints = extract_schema_constraints(model_class)

    constraints_text = ""
    if constraints:
        constraints_text = "\n\nIMPORTANT - Field constraints (use EXACT values):\n" + "\n".join(constraints)

    # Determine the appropriate output prefix based on the JSON structure
    # RootModel with list type will produce a JSON array, not an object
    is_array = json_model.strip().startswith("[")
    output_prefix = "[" if is_array else "{"

    # Add extra instruction for array outputs to prevent LLM from wrapping in object
    array_instruction = ""
    if is_array:
        array_instruction = " IMPORTANT: Return ONLY the JSON array directly, do NOT wrap it in an object."

    return (
        prompt +
        constraints_text +
        f"\n\nThe response should be in a structured JSON format for machine "
        f"processing that matches the following JSON: {json_model}{array_instruction}\n\nOUTPUT: {output_prefix}"
    )


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
    images: list = None,
    detail: str = "auto",
    web_search: bool = False,
    # Reasoning/Thinking parameters
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
    thinking: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Union[BaseModel, LLMFullResponse, str]:
    """
    Generates a Pydantic model instance based on a given prompt, retrying on validation errors.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param llm_instance: Instance of a large language model.
    :param max_retries: Maximum number of retries on validation errors.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output. Lower values make output more deterministic.
        Note: Ignored by reasoning models (OpenAI o1/o3, DeepSeek reasoner).
    :param top_p: Controls diversity of output. Lower values make output more focused.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.
    :param system_prompt: System prompt to set the context for the LLM.
    :param full_response: If True, returns the full API response including token counts.
    :param images: A list of image URLs or file paths for vision tasks.
    :param detail: Level of detail for image analysis ("low", "high", "auto"). Defaults to "auto".
    :param web_search: If True, enables web search before generating response. Defaults to False.
    :param reasoning_effort: Reasoning depth for OpenAI/OpenRouter models ("low", "medium", "high").
        Only applies to o1, o3, and GPT-5 series models. Defaults to None.
    :param thinking_budget: Token budget for extended thinking (Anthropic/Gemini).
        Must be at least 1024 and less than max_tokens. Defaults to None.
    :param thinking_level: Thinking level preset for Gemini 3.x models
        ("minimal", "low", "medium", "high"). Defaults to None.
    :param thinking: Enable chain-of-thought for DeepSeek reasoner. Defaults to None.
    :param timeout: Request timeout in seconds. Defaults to None.

    :return:
        - If full_response=False: BaseModel object
        - If full_response=True: LLMFullResponse object with model_object attribute
        - Error message string if unsuccessful

    Example:
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: float
        ...     in_stock: bool
        >>>
        >>> llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
        >>> product = generate_pydantic_json_model(
        ...     model_class=Product,
        ...     prompt="Generate a laptop product",
        ...     llm_instance=llm,
        ...     temperature=0.5
        ... )
        >>> print(product.name, product.price)
    """
    # Validate reasoning parameters
    _validate_reasoning_params(thinking_budget, max_tokens)

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
                full_response=full_response,
                images=images,
                detail=detail,
                web_search=web_search,
                reasoning_effort=reasoning_effort,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
                thinking=thinking,
                timeout=timeout,
            )

            response_text = ai_response.generated_text if full_response else ai_response

            if response_text:
                json_object = extract_json_from_text(response_text)

                if json_object is None:
                    if attempt < max_retries:
                        continue
                    else:
                        return f"No valid JSON found in response after {max_retries} attempts"

                # Handle RootModel list unwrapping (when LLM wraps list in object)
                json_object = _unwrap_rootmodel_list(json_object, model_class)

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

        except Exception as e:
            return f"Exception occurred: {e}"

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
    images: list = None,
    detail: str = "auto",
    web_search: bool = False,
    # Reasoning/Thinking parameters
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
    thinking: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Union[Tuple[BaseModel, LLMProvider, str], LLMFullResponse, str]:
    """
    Generates a Pydantic model instance using ReliableLLM with fallback capability.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers.
    :param max_retries: Maximum number of retries on validation errors.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output. Lower values make output more deterministic.
        Note: Ignored by reasoning models (OpenAI o1/o3, DeepSeek reasoner).
    :param top_p: Controls diversity of output. Lower values make output more focused.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.
    :param system_prompt: System prompt to set the context for the LLM.
    :param full_response: If True, returns the full API response including token counts.
    :param images: A list of image URLs or file paths for vision tasks.
    :param detail: Level of detail for image analysis ("low", "high", "auto"). Defaults to "auto".
    :param web_search: If True, enables web search before generating response. Defaults to False.
    :param reasoning_effort: Reasoning depth for OpenAI/OpenRouter models ("low", "medium", "high").
        Only applies to o1, o3, and GPT-5 series models. Defaults to None.
    :param thinking_budget: Token budget for extended thinking (Anthropic/Gemini).
        Must be at least 1024 and less than max_tokens. Defaults to None.
    :param thinking_level: Thinking level preset for Gemini 3.x models
        ("minimal", "low", "medium", "high"). Defaults to None.
    :param thinking: Enable chain-of-thought for DeepSeek reasoner. Defaults to None.
    :param timeout: Request timeout in seconds. Defaults to None.

    :return:
        - If full_response=False: Tuple of (model_object, provider, model_name)
        - If full_response=True: LLMFullResponse object with model_object, provider, and model_name attributes
        - Error message string if unsuccessful

    Example:
        >>> reliable = ReliableLLM(primary_llm=openai_llm, secondary_llm=anthropic_llm)
        >>> result, provider, model = generate_pydantic_json_model_reliable(
        ...     model_class=Product,
        ...     prompt="Generate a product",
        ...     reliable_llm=reliable
        ... )
    """
    # Validate reasoning parameters
    _validate_reasoning_params(thinking_budget, max_tokens)

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
                full_response=full_response,
                images=images,
                detail=detail,
                web_search=web_search,
                reasoning_effort=reasoning_effort,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
                thinking=thinking,
                timeout=timeout,
            )

            if full_response:
                ai_response, provider, model_name = result
                response_text = ai_response.generated_text
            else:
                response_text, provider, model_name = result

            if response_text:
                json_object = extract_json_from_text(response_text)

                if json_object is None:
                    if attempt < max_retries:
                        continue
                    else:
                        return f"No valid JSON found in response after {max_retries} attempts"

                # Handle RootModel list unwrapping (when LLM wraps list in object)
                json_object = _unwrap_rootmodel_list(json_object, model_class)

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
    images: list = None,
    detail: str = "auto",
    web_search: bool = False,
    # Reasoning/Thinking parameters
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
    thinking: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Union[BaseModel, LLMFullResponse, str]:
    """
    Asynchronously generates a Pydantic model instance based on a given prompt.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param llm_instance: Instance of a large language model.
    :param max_retries: Maximum number of retries on validation errors.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output. Lower values make output more deterministic.
        Note: Ignored by reasoning models (OpenAI o1/o3, DeepSeek reasoner).
    :param top_p: Controls diversity of output. Lower values make output more focused.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.
    :param system_prompt: System prompt to set the context for the LLM.
    :param full_response: If True, returns the full API response including token counts.
    :param images: A list of image URLs or file paths for vision tasks.
    :param detail: Level of detail for image analysis ("low", "high", "auto"). Defaults to "auto".
    :param web_search: If True, enables web search before generating response. Defaults to False.
    :param reasoning_effort: Reasoning depth for OpenAI/OpenRouter models ("low", "medium", "high").
        Only applies to o1, o3, and GPT-5 series models. Defaults to None.
    :param thinking_budget: Token budget for extended thinking (Anthropic/Gemini).
        Must be at least 1024 and less than max_tokens. Defaults to None.
    :param thinking_level: Thinking level preset for Gemini 3.x models
        ("minimal", "low", "medium", "high"). Defaults to None.
    :param thinking: Enable chain-of-thought for DeepSeek reasoner. Defaults to None.
    :param timeout: Request timeout in seconds. Defaults to None.

    :return:
        - If full_response=False: BaseModel object
        - If full_response=True: LLMFullResponse object with model_object attribute
        - Error message string if unsuccessful

    Example:
        >>> async def main():
        ...     product = await generate_pydantic_json_model_async(
        ...         model_class=Product,
        ...         prompt="Generate a laptop product",
        ...         llm_instance=llm
        ...     )
        ...     print(product.name)
    """
    # Validate reasoning parameters
    _validate_reasoning_params(thinking_budget, max_tokens)

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
                full_response=full_response,
                images=images,
                detail=detail,
                web_search=web_search,
                reasoning_effort=reasoning_effort,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
                thinking=thinking,
                timeout=timeout,
            )

            response_text = ai_response.generated_text if full_response else ai_response

            if response_text:
                json_object = extract_json_from_text(response_text)

                if json_object is None:
                    if attempt < max_retries:
                        continue
                    else:
                        return f"No valid JSON found in response after {max_retries} attempts"

                # Handle RootModel list unwrapping (when LLM wraps list in object)
                json_object = _unwrap_rootmodel_list(json_object, model_class)

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

        except Exception as e:
            return f"Exception occurred: {e}"

        if (not response_text or errors) and attempt < max_retries:
            await asyncio.sleep(delay)
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
    images: list = None,
    detail: str = "auto",
    web_search: bool = False,
    # Reasoning/Thinking parameters
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
    thinking: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Union[Tuple[BaseModel, LLMProvider, str], LLMFullResponse, str]:
    """
    Asynchronously generates a Pydantic model instance using ReliableLLM with fallback capability.

    :param model_class: The Pydantic model class to be used for validation and conversion.
    :param prompt: The fully formatted prompt including the topic.
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers.
    :param max_retries: Maximum number of retries on validation errors.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output. Lower values make output more deterministic.
        Note: Ignored by reasoning models (OpenAI o1/o3, DeepSeek reasoner).
    :param top_p: Controls diversity of output. Lower values make output more focused.
    :param initial_delay: Initial delay in seconds before the first retry.
    :param custom_prompt_suffix: Optional string to customize or override the generated prompt extension.
    :param system_prompt: System prompt to set the context for the LLM.
    :param full_response: If True, returns the full API response including token counts.
    :param images: A list of image URLs or file paths for vision tasks.
    :param detail: Level of detail for image analysis ("low", "high", "auto"). Defaults to "auto".
    :param web_search: If True, enables web search before generating response. Defaults to False.
    :param reasoning_effort: Reasoning depth for OpenAI/OpenRouter models ("low", "medium", "high").
        Only applies to o1, o3, and GPT-5 series models. Defaults to None.
    :param thinking_budget: Token budget for extended thinking (Anthropic/Gemini).
        Must be at least 1024 and less than max_tokens. Defaults to None.
    :param thinking_level: Thinking level preset for Gemini 3.x models
        ("minimal", "low", "medium", "high"). Defaults to None.
    :param thinking: Enable chain-of-thought for DeepSeek reasoner. Defaults to None.
    :param timeout: Request timeout in seconds. Defaults to None.

    :return:
        - If full_response=False: Tuple of (model_object, provider, model_name)
        - If full_response=True: LLMFullResponse object with model_object, provider, and model_name attributes
        - Error message string if unsuccessful

    Example:
        >>> async def main():
        ...     reliable = ReliableLLM(primary_llm=openai_llm, secondary_llm=anthropic_llm)
        ...     result, provider, model = await generate_pydantic_json_model_reliable_async(
        ...         model_class=Product,
        ...         prompt="Generate a product",
        ...         reliable_llm=reliable
        ...     )
    """
    # Validate reasoning parameters
    _validate_reasoning_params(thinking_budget, max_tokens)

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
                full_response=full_response,
                images=images,
                detail=detail,
                web_search=web_search,
                reasoning_effort=reasoning_effort,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
                thinking=thinking,
                timeout=timeout,
            )

            if full_response:
                ai_response, provider, model_name = result
                response_text = ai_response.generated_text
            else:
                response_text, provider, model_name = result

            if response_text:
                json_object = extract_json_from_text(response_text)

                if json_object is None:
                    if attempt < max_retries:
                        continue
                    else:
                        return f"No valid JSON found in response after {max_retries} attempts"

                # Handle RootModel list unwrapping (when LLM wraps list in object)
                json_object = _unwrap_rootmodel_list(json_object, model_class)

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
