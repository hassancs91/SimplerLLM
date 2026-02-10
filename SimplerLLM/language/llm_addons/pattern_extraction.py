"""
Pattern extraction from LLM responses.

This module provides functions for extracting structured patterns (emails, phones,
URLs, etc.) from LLM responses with validation, normalization, and fallback support.

Supported Patterns:
    - email: Email addresses
    - phone: Phone numbers
    - url: Web URLs
    - date: Various date formats
    - time: Time formats
    - ssn: Social Security Numbers
    - credit_card: Credit card numbers
    - zip_code: ZIP codes
    - ipv4: IPv4 addresses
    - ipv6: IPv6 addresses
    - currency: Currency amounts
    - hex_color: Hex color codes
    - username: Usernames
    - hashtag: Hashtags
    - filepath: File paths
    - custom: Custom regex patterns

Example:
    >>> from SimplerLLM.language import LLM, LLMProvider
    >>> from SimplerLLM.language.llm_addons import generate_structured_pattern
    >>>
    >>> llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    >>> result = generate_structured_pattern(
    ...     pattern="email",
    ...     prompt="What is the contact email for support?",
    ...     llm_instance=llm,
    ...     validate=True,
    ...     normalize=True
    ... )
    >>> print(result.matches[0].value)
"""

import time
import asyncio
from typing import Union, Dict, Tuple
from datetime import datetime

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_providers.llm_response_models import (
    LLMFullResponse,
    PatternMatch,
    PatternExtractionResult,
)

from SimplerLLM.tools.pattern_helpers import (
    get_predefined_pattern,
    extract_pattern_from_text,
    create_pattern_extraction_prompt,
    get_validation_function,
    get_normalization_function,
)


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

    :param pattern: Pattern name (e.g., 'email', 'phone') or custom regex dict {'custom': 'regex_pattern'}.
    :param prompt: The user's prompt for the LLM.
    :param llm_instance: Instance of a large language model.
    :param extract_all: If True, extract all matches; if False, extract only first match.
    :param validate: If True, validate matches beyond regex (e.g., email domain validation).
    :param normalize: If True, normalize extracted values (e.g., lowercase email, format phone).
    :param max_retries: Maximum number of retries if no matches found.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output.
    :param top_p: Controls diversity of output.
    :param initial_delay: Initial delay in seconds before first retry.
    :param custom_prompt_suffix: Optional custom suffix to override prompt enhancement.
    :param system_prompt: System prompt to set context for the LLM.
    :param full_response: If True, returns full API response including token counts.

    :return:
        - If full_response=False: PatternExtractionResult object
        - If full_response=True: LLMFullResponse object with extraction_result attribute
        - Error message string if unsuccessful

    Example:
        >>> result = generate_structured_pattern(
        ...     pattern="email",
        ...     prompt="What is the contact email?",
        ...     llm_instance=llm,
        ...     validate=True,
        ...     normalize=True
        ... )
        >>> print(result.matches[0].value)
        >>> print(result.matches[0].is_valid)
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
                            except Exception:
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

    :param pattern: Pattern name (e.g., 'email', 'phone') or custom regex dict {'custom': 'regex_pattern'}.
    :param prompt: The user's prompt for the LLM.
    :param llm_instance: Instance of a large language model.
    :param extract_all: If True, extract all matches; if False, extract only first match.
    :param validate: If True, validate matches beyond regex (e.g., email domain validation).
    :param normalize: If True, normalize extracted values (e.g., lowercase email, format phone).
    :param max_retries: Maximum number of retries if no matches found.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output.
    :param top_p: Controls diversity of output.
    :param initial_delay: Initial delay in seconds before first retry.
    :param custom_prompt_suffix: Optional custom suffix to override prompt enhancement.
    :param system_prompt: System prompt to set context for the LLM.
    :param full_response: If True, returns full API response including token counts.

    :return:
        - If full_response=False: PatternExtractionResult object
        - If full_response=True: LLMFullResponse object with extraction_result attribute
        - Error message string if unsuccessful

    Example:
        >>> async def main():
        ...     result = await generate_structured_pattern_async(
        ...         pattern="phone",
        ...         prompt="What is the customer service number?",
        ...         llm_instance=llm
        ...     )
        ...     print(result.matches[0].value)
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
                            except Exception:
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

    :param pattern: Pattern name (e.g., 'email', 'phone') or custom regex dict {'custom': 'regex_pattern'}.
    :param prompt: The user's prompt for the LLM.
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers.
    :param extract_all: If True, extract all matches; if False, extract only first match.
    :param validate: If True, validate matches beyond regex (e.g., email domain validation).
    :param normalize: If True, normalize extracted values (e.g., lowercase email, format phone).
    :param max_retries: Maximum number of retries if no matches found.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output.
    :param top_p: Controls diversity of output.
    :param initial_delay: Initial delay in seconds before first retry.
    :param custom_prompt_suffix: Optional custom suffix to override prompt enhancement.
    :param system_prompt: System prompt to set context for the LLM.
    :param full_response: If True, returns full API response including token counts.

    :return:
        - If full_response=False: Tuple of (PatternExtractionResult, provider, model_name)
        - If full_response=True: LLMFullResponse object with extraction_result, provider, and model_name attributes
        - Error message string if unsuccessful

    Example:
        >>> reliable = ReliableLLM(primary_llm=openai_llm, secondary_llm=anthropic_llm)
        >>> result, provider, model = generate_structured_pattern_reliable(
        ...     pattern="email",
        ...     prompt="What is the contact email?",
        ...     reliable_llm=reliable
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
                            except Exception:
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

    :param pattern: Pattern name (e.g., 'email', 'phone') or custom regex dict {'custom': 'regex_pattern'}.
    :param prompt: The user's prompt for the LLM.
    :param reliable_llm: Instance of ReliableLLM with primary and secondary providers.
    :param extract_all: If True, extract all matches; if False, extract only first match.
    :param validate: If True, validate matches beyond regex (e.g., email domain validation).
    :param normalize: If True, normalize extracted values (e.g., lowercase email, format phone).
    :param max_retries: Maximum number of retries if no matches found.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: Controls randomness in output.
    :param top_p: Controls diversity of output.
    :param initial_delay: Initial delay in seconds before first retry.
    :param custom_prompt_suffix: Optional custom suffix to override prompt enhancement.
    :param system_prompt: System prompt to set context for the LLM.
    :param full_response: If True, returns full API response including token counts.

    :return:
        - If full_response=False: Tuple of (PatternExtractionResult, provider, model_name)
        - If full_response=True: LLMFullResponse object with extraction_result, provider, and model_name attributes
        - Error message string if unsuccessful

    Example:
        >>> async def main():
        ...     reliable = ReliableLLM(primary_llm=openai_llm, secondary_llm=anthropic_llm)
        ...     result, provider, model = await generate_structured_pattern_reliable_async(
        ...         pattern="url",
        ...         prompt="What is the company website?",
        ...         reliable_llm=reliable
        ...     )
        ...     print(result.matches[0].value)
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
                            except Exception:
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
