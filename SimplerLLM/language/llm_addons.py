import time
from typing import Type, Union, Tuple, Dict, List
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_providers.llm_response_models import (
    LLMFullResponse,
    PatternMatch,
    PatternExtractionResult
)
import asyncio
import tiktoken
from datetime import datetime

from SimplerLLM.tools.json_helpers import (
    extract_json_from_text,
    convert_json_to_pydantic_model,
    validate_json_with_pydantic_model,
    generate_json_example_from_pydantic,
)

from SimplerLLM.tools.pattern_helpers import (
    get_predefined_pattern,
    extract_pattern_from_text,
    create_pattern_extraction_prompt,
    get_validation_function,
    get_normalization_function,
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


# ==============================================================================
# PATTERN EXTRACTION FUNCTIONS
# ==============================================================================

def generate_structured_pattern(
    pattern: Union[str, Dict[str, str]],
    prompt: str,
    llm_instance: LLM,
    extract_all: bool = False,
    validate: bool = True,
    normalize: bool = False,
    max_retries: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = None,
    full_response: bool = False,
) -> Union[PatternExtractionResult, LLMFullResponse, str]:
    """
    Generates a pattern extraction result from LLM output using regex patterns.

    :param pattern: Pattern name (e.g., 'email', 'phone') or custom regex dict {'custom': 'regex_pattern'}
    :param prompt: The user's prompt for the LLM
    :param llm_instance: Instance of a large language model
    :param extract_all: If True, extract all matches; if False, extract only first match
    :param validate: If True, validate matches beyond regex (e.g., email domain validation)
    :param normalize: If True, normalize extracted values (e.g., lowercase email, format phone)
    :param max_retries: Maximum number of retries if no matches found
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in output
    :param top_p: Controls diversity of output
    :param initial_delay: Initial delay in seconds before first retry
    :param custom_prompt_suffix: Optional custom suffix to override prompt enhancement
    :param system_prompt: System prompt to set context for the LLM
    :param full_response: If True, returns full API response including token counts

    :return:
        - If full_response=False: PatternExtractionResult object
        - If full_response=True: LLMFullResponse object with extraction_result attribute
        - Error message string if unsuccessful

    Example:
        >>> llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")
        >>> result = generate_structured_pattern(
        ...     pattern="email",
        ...     prompt="What is the contact email?",
        ...     llm_instance=llm,
        ...     validate=True,
        ...     normalize=True
        ... )
        >>> print(result.matches[0].value)
    """
    # Determine pattern type and regex
    if isinstance(pattern, dict) and 'custom' in pattern:
        pattern_type = 'custom'
        regex_pattern = pattern['custom']
    elif isinstance(pattern, str):
        pattern_type = pattern.lower()
        regex_pattern = get_predefined_pattern(pattern_type)
        if not regex_pattern:
            return f"Unknown pattern type: {pattern}. Use a predefined pattern or provide a custom regex."
    else:
        return "Invalid pattern format. Provide a pattern name string or dict with 'custom' key."

    # Create optimized prompt
    if custom_prompt_suffix:
        optimized_prompt = custom_prompt_suffix
    else:
        optimized_prompt = create_pattern_extraction_prompt(prompt, pattern_type, extract_all)

    # Default system prompt for pattern extraction
    if system_prompt is None:
        system_prompt = f"You are a helpful assistant that provides accurate {pattern_type} information."

    # Calculate exponential backoff delays
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            # Generate LLM response
            ai_response = llm_instance.generate_response(
                prompt=optimized_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                full_response=full_response
            )

            response_text = ai_response.generated_text if full_response else ai_response

            if response_text:
                # Extract pattern matches from response
                raw_matches = extract_pattern_from_text(
                    text=response_text,
                    pattern=regex_pattern,
                    extract_all=extract_all
                )

                # If no matches found, retry
                if not raw_matches:
                    if attempt < max_retries:
                        continue
                    else:
                        return f"No {pattern_type} pattern found in response after {max_retries} attempts"

                # Process matches with validation and normalization
                processed_matches = []
                for match_data in raw_matches:
                    match_value = match_data['value']
                    match_position = match_data['position']

                    # Validate if requested
                    is_valid = True
                    validation_msg = "Match found"
                    if validate:
                        validator = get_validation_function(pattern_type)
                        if validator:
                            is_valid, validation_msg = validator(match_value)
                        else:
                            validation_msg = "No validator available for this pattern"

                    # Normalize if requested
                    normalized_value = None
                    if normalize:
                        normalizer = get_normalization_function(pattern_type)
                        if normalizer:
                            try:
                                normalized_value = normalizer(match_value)
                            except Exception as e:
                                normalized_value = match_value  # Fallback to original

                    # Create PatternMatch object
                    pattern_match = PatternMatch(
                        value=match_value,
                        normalized_value=normalized_value,
                        pattern_type=pattern_type,
                        position=match_position,
                        is_valid=is_valid,
                        validation_message=validation_msg,
                        confidence=1.0 if is_valid else 0.5
                    )
                    processed_matches.append(pattern_match)

                # Create result object
                extraction_result = PatternExtractionResult(
                    matches=processed_matches,
                    total_matches=len(processed_matches),
                    pattern_used=regex_pattern,
                    original_text=response_text,
                    extraction_timestamp=datetime.now()
                )

                # Return based on full_response flag
                if full_response:
                    ai_response.extraction_result = extraction_result
                    return ai_response
                else:
                    return extraction_result

        except Exception as e:
            return f"Exception occurred: {e}"

        # Retry logic if no matches found
        if attempt < max_retries:
            time.sleep(delay)

    return f"Maximum retries exceeded without finding {pattern_type} pattern."


async def generate_structured_pattern_async(
    pattern: Union[str, Dict[str, str]],
    prompt: str,
    llm_instance: LLM,
    extract_all: bool = False,
    validate: bool = True,
    normalize: bool = False,
    max_retries: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = None,
    full_response: bool = False,
) -> Union[PatternExtractionResult, LLMFullResponse, str]:
    """
    Asynchronously generates a pattern extraction result from LLM output using regex patterns.

    :param pattern: Pattern name (e.g., 'email', 'phone') or custom regex dict {'custom': 'regex_pattern'}
    :param prompt: The user's prompt for the LLM
    :param llm_instance: Instance of a large language model
    :param extract_all: If True, extract all matches; if False, extract only first match
    :param validate: If True, validate matches beyond regex (e.g., email domain validation)
    :param normalize: If True, normalize extracted values (e.g., lowercase email, format phone)
    :param max_retries: Maximum number of retries if no matches found
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in output
    :param top_p: Controls diversity of output
    :param initial_delay: Initial delay in seconds before first retry
    :param custom_prompt_suffix: Optional custom suffix to override prompt enhancement
    :param system_prompt: System prompt to set context for the LLM
    :param full_response: If True, returns full API response including token counts

    :return:
        - If full_response=False: PatternExtractionResult object
        - If full_response=True: LLMFullResponse object with extraction_result attribute
        - Error message string if unsuccessful

    Example:
        >>> llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")
        >>> result = await generate_structured_pattern_async(
        ...     pattern="email",
        ...     prompt="What is the contact email?",
        ...     llm_instance=llm,
        ...     validate=True,
        ...     normalize=True
        ... )
        >>> print(result.matches[0].value)
    """
    # Determine pattern type and regex
    if isinstance(pattern, dict) and 'custom' in pattern:
        pattern_type = 'custom'
        regex_pattern = pattern['custom']
    elif isinstance(pattern, str):
        pattern_type = pattern.lower()
        regex_pattern = get_predefined_pattern(pattern_type)
        if not regex_pattern:
            return f"Unknown pattern type: {pattern}. Use a predefined pattern or provide a custom regex."
    else:
        return "Invalid pattern format. Provide a pattern name string or dict with 'custom' key."

    # Create optimized prompt
    if custom_prompt_suffix:
        optimized_prompt = custom_prompt_suffix
    else:
        optimized_prompt = create_pattern_extraction_prompt(prompt, pattern_type, extract_all)

    # Default system prompt for pattern extraction
    if system_prompt is None:
        system_prompt = f"You are a helpful assistant that provides accurate {pattern_type} information."

    # Calculate exponential backoff delays
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            # Generate LLM response asynchronously
            ai_response = await llm_instance.generate_response_async(
                prompt=optimized_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                full_response=full_response
            )

            response_text = ai_response.generated_text if full_response else ai_response

            if response_text:
                # Extract pattern matches from response
                raw_matches = extract_pattern_from_text(
                    text=response_text,
                    pattern=regex_pattern,
                    extract_all=extract_all
                )

                # If no matches found, retry
                if not raw_matches:
                    if attempt < max_retries:
                        continue
                    else:
                        return f"No {pattern_type} pattern found in response after {max_retries} attempts"

                # Process matches with validation and normalization
                processed_matches = []
                for match_data in raw_matches:
                    match_value = match_data['value']
                    match_position = match_data['position']

                    # Validate if requested
                    is_valid = True
                    validation_msg = "Match found"
                    if validate:
                        validator = get_validation_function(pattern_type)
                        if validator:
                            is_valid, validation_msg = validator(match_value)
                        else:
                            validation_msg = "No validator available for this pattern"

                    # Normalize if requested
                    normalized_value = None
                    if normalize:
                        normalizer = get_normalization_function(pattern_type)
                        if normalizer:
                            try:
                                normalized_value = normalizer(match_value)
                            except Exception as e:
                                normalized_value = match_value  # Fallback to original

                    # Create PatternMatch object
                    pattern_match = PatternMatch(
                        value=match_value,
                        normalized_value=normalized_value,
                        pattern_type=pattern_type,
                        position=match_position,
                        is_valid=is_valid,
                        validation_message=validation_msg,
                        confidence=1.0 if is_valid else 0.5
                    )
                    processed_matches.append(pattern_match)

                # Create result object
                extraction_result = PatternExtractionResult(
                    matches=processed_matches,
                    total_matches=len(processed_matches),
                    pattern_used=regex_pattern,
                    original_text=response_text,
                    extraction_timestamp=datetime.now()
                )

                # Return based on full_response flag
                if full_response:
                    ai_response.extraction_result = extraction_result
                    return ai_response
                else:
                    return extraction_result

        except Exception as e:
            return f"Exception occurred: {e}"

        # Retry logic if no matches found
        if attempt < max_retries:
            await asyncio.sleep(delay)

    return f"Maximum retries exceeded without finding {pattern_type} pattern."


def generate_structured_pattern_reliable(
    pattern: Union[str, Dict[str, str]],
    prompt: str,
    reliable_llm: ReliableLLM,
    extract_all: bool = False,
    validate: bool = True,
    normalize: bool = False,
    max_retries: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = None,
    full_response: bool = False,
) -> Union[Tuple[PatternExtractionResult, LLMProvider, str], LLMFullResponse, str]:
    """
    Generates a pattern extraction result using ReliableLLM with fallback capability.

    :param pattern: Pattern name (e.g., 'email', 'phone') or custom regex dict {'custom': 'regex_pattern'}
    :param prompt: The user's prompt for the LLM
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers
    :param extract_all: If True, extract all matches; if False, extract only first match
    :param validate: If True, validate matches beyond regex (e.g., email domain validation)
    :param normalize: If True, normalize extracted values (e.g., lowercase email, format phone)
    :param max_retries: Maximum number of retries if no matches found
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in output
    :param top_p: Controls diversity of output
    :param initial_delay: Initial delay in seconds before first retry
    :param custom_prompt_suffix: Optional custom suffix to override prompt enhancement
    :param system_prompt: System prompt to set context for the LLM
    :param full_response: If True, returns full API response including token counts

    :return:
        - If full_response=False: Tuple of (PatternExtractionResult, provider, model_name)
        - If full_response=True: LLMFullResponse object with extraction_result, provider, and model_name attributes
        - Error message string if unsuccessful

    Example:
        >>> reliable_llm = ReliableLLM(primary_llm=llm1, secondary_llm=llm2)
        >>> result, provider, model = generate_structured_pattern_reliable(
        ...     pattern="email",
        ...     prompt="What is the contact email?",
        ...     reliable_llm=reliable_llm,
        ...     validate=True,
        ...     normalize=True
        ... )
        >>> print(result.matches[0].value)
    """
    # Determine pattern type and regex
    if isinstance(pattern, dict) and 'custom' in pattern:
        pattern_type = 'custom'
        regex_pattern = pattern['custom']
    elif isinstance(pattern, str):
        pattern_type = pattern.lower()
        regex_pattern = get_predefined_pattern(pattern_type)
        if not regex_pattern:
            return f"Unknown pattern type: {pattern}. Use a predefined pattern or provide a custom regex."
    else:
        return "Invalid pattern format. Provide a pattern name string or dict with 'custom' key."

    # Create optimized prompt
    if custom_prompt_suffix:
        optimized_prompt = custom_prompt_suffix
    else:
        optimized_prompt = create_pattern_extraction_prompt(prompt, pattern_type, extract_all)

    # Default system prompt for pattern extraction
    if system_prompt is None:
        system_prompt = f"You are a helpful assistant that provides accurate {pattern_type} information."

    # Calculate exponential backoff delays
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            # Generate LLM response with fallback support
            result = reliable_llm.generate_response(
                return_provider=True,
                prompt=optimized_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                full_response=full_response
            )

            if full_response:
                ai_response, provider, model_name = result
                response_text = ai_response.generated_text
            else:
                response_text, provider, model_name = result

            if response_text:
                # Extract pattern matches from response
                raw_matches = extract_pattern_from_text(
                    text=response_text,
                    pattern=regex_pattern,
                    extract_all=extract_all
                )

                # If no matches found, retry
                if not raw_matches:
                    if attempt < max_retries:
                        continue
                    else:
                        return f"No {pattern_type} pattern found in response after {max_retries} attempts"

                # Process matches with validation and normalization
                processed_matches = []
                for match_data in raw_matches:
                    match_value = match_data['value']
                    match_position = match_data['position']

                    # Validate if requested
                    is_valid = True
                    validation_msg = "Match found"
                    if validate:
                        validator = get_validation_function(pattern_type)
                        if validator:
                            is_valid, validation_msg = validator(match_value)
                        else:
                            validation_msg = "No validator available for this pattern"

                    # Normalize if requested
                    normalized_value = None
                    if normalize:
                        normalizer = get_normalization_function(pattern_type)
                        if normalizer:
                            try:
                                normalized_value = normalizer(match_value)
                            except Exception as e:
                                normalized_value = match_value  # Fallback to original

                    # Create PatternMatch object
                    pattern_match = PatternMatch(
                        value=match_value,
                        normalized_value=normalized_value,
                        pattern_type=pattern_type,
                        position=match_position,
                        is_valid=is_valid,
                        validation_message=validation_msg,
                        confidence=1.0 if is_valid else 0.5
                    )
                    processed_matches.append(pattern_match)

                # Create result object
                extraction_result = PatternExtractionResult(
                    matches=processed_matches,
                    total_matches=len(processed_matches),
                    pattern_used=regex_pattern,
                    original_text=response_text,
                    extraction_timestamp=datetime.now()
                )

                # Return based on full_response flag
                if full_response:
                    ai_response.extraction_result = extraction_result
                    ai_response.provider = provider
                    ai_response.model_name = model_name
                    return ai_response
                else:
                    return (extraction_result, provider, model_name)

        except Exception as e:
            return f"Exception occurred: {e}"

        # Retry logic if no matches found
        if attempt < max_retries:
            time.sleep(delay)

    return f"Maximum retries exceeded without finding {pattern_type} pattern."


async def generate_structured_pattern_reliable_async(
    pattern: Union[str, Dict[str, str]],
    prompt: str,
    reliable_llm: ReliableLLM,
    extract_all: bool = False,
    validate: bool = True,
    normalize: bool = False,
    max_retries: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 1.0,
    initial_delay: float = 1.0,
    custom_prompt_suffix: str = None,
    system_prompt: str = None,
    full_response: bool = False,
) -> Union[Tuple[PatternExtractionResult, LLMProvider, str], LLMFullResponse, str]:
    """
    Asynchronously generates a pattern extraction result using ReliableLLM with fallback capability.

    :param pattern: Pattern name (e.g., 'email', 'phone') or custom regex dict {'custom': 'regex_pattern'}
    :param prompt: The user's prompt for the LLM
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers
    :param extract_all: If True, extract all matches; if False, extract only first match
    :param validate: If True, validate matches beyond regex (e.g., email domain validation)
    :param normalize: If True, normalize extracted values (e.g., lowercase email, format phone)
    :param max_retries: Maximum number of retries if no matches found
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in output
    :param top_p: Controls diversity of output
    :param initial_delay: Initial delay in seconds before first retry
    :param custom_prompt_suffix: Optional custom suffix to override prompt enhancement
    :param system_prompt: System prompt to set context for the LLM
    :param full_response: If True, returns full API response including token counts

    :return:
        - If full_response=False: Tuple of (PatternExtractionResult, provider, model_name)
        - If full_response=True: LLMFullResponse object with extraction_result, provider, and model_name attributes
        - Error message string if unsuccessful

    Example:
        >>> reliable_llm = ReliableLLM(primary_llm=llm1, secondary_llm=llm2)
        >>> result, provider, model = await generate_structured_pattern_reliable_async(
        ...     pattern="email",
        ...     prompt="What is the contact email?",
        ...     reliable_llm=reliable_llm,
        ...     validate=True,
        ...     normalize=True
        ... )
        >>> print(result.matches[0].value)
    """
    # Determine pattern type and regex
    if isinstance(pattern, dict) and 'custom' in pattern:
        pattern_type = 'custom'
        regex_pattern = pattern['custom']
    elif isinstance(pattern, str):
        pattern_type = pattern.lower()
        regex_pattern = get_predefined_pattern(pattern_type)
        if not regex_pattern:
            return f"Unknown pattern type: {pattern}. Use a predefined pattern or provide a custom regex."
    else:
        return "Invalid pattern format. Provide a pattern name string or dict with 'custom' key."

    # Create optimized prompt
    if custom_prompt_suffix:
        optimized_prompt = custom_prompt_suffix
    else:
        optimized_prompt = create_pattern_extraction_prompt(prompt, pattern_type, extract_all)

    # Default system prompt for pattern extraction
    if system_prompt is None:
        system_prompt = f"You are a helpful assistant that provides accurate {pattern_type} information."

    # Calculate exponential backoff delays
    backoff_delays = [initial_delay * (2**attempt) for attempt in range(max_retries + 1)]

    for attempt, delay in enumerate(backoff_delays):
        try:
            # Generate LLM response asynchronously with fallback support
            result = await reliable_llm.generate_response_async(
                return_provider=True,
                prompt=optimized_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                full_response=full_response
            )

            if full_response:
                ai_response, provider, model_name = result
                response_text = ai_response.generated_text
            else:
                response_text, provider, model_name = result

            if response_text:
                # Extract pattern matches from response
                raw_matches = extract_pattern_from_text(
                    text=response_text,
                    pattern=regex_pattern,
                    extract_all=extract_all
                )

                # If no matches found, retry
                if not raw_matches:
                    if attempt < max_retries:
                        continue
                    else:
                        return f"No {pattern_type} pattern found in response after {max_retries} attempts"

                # Process matches with validation and normalization
                processed_matches = []
                for match_data in raw_matches:
                    match_value = match_data['value']
                    match_position = match_data['position']

                    # Validate if requested
                    is_valid = True
                    validation_msg = "Match found"
                    if validate:
                        validator = get_validation_function(pattern_type)
                        if validator:
                            is_valid, validation_msg = validator(match_value)
                        else:
                            validation_msg = "No validator available for this pattern"

                    # Normalize if requested
                    normalized_value = None
                    if normalize:
                        normalizer = get_normalization_function(pattern_type)
                        if normalizer:
                            try:
                                normalized_value = normalizer(match_value)
                            except Exception as e:
                                normalized_value = match_value  # Fallback to original

                    # Create PatternMatch object
                    pattern_match = PatternMatch(
                        value=match_value,
                        normalized_value=normalized_value,
                        pattern_type=pattern_type,
                        position=match_position,
                        is_valid=is_valid,
                        validation_message=validation_msg,
                        confidence=1.0 if is_valid else 0.5
                    )
                    processed_matches.append(pattern_match)

                # Create result object
                extraction_result = PatternExtractionResult(
                    matches=processed_matches,
                    total_matches=len(processed_matches),
                    pattern_used=regex_pattern,
                    original_text=response_text,
                    extraction_timestamp=datetime.now()
                )

                # Return based on full_response flag
                if full_response:
                    ai_response.extraction_result = extraction_result
                    ai_response.provider = provider
                    ai_response.model_name = model_name
                    return ai_response
                else:
                    return (extraction_result, provider, model_name)

        except Exception as e:
            return f"Exception occurred: {e}"

        # Retry logic if no matches found
        if attempt < max_retries:
            await asyncio.sleep(delay)

    return f"Maximum retries exceeded without finding {pattern_type} pattern."
