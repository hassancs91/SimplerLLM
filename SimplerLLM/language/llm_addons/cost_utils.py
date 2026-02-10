"""
Cost calculation utilities for LLM text generation.

This module provides utilities for calculating the cost of LLM API calls
based on token counts and pricing.
"""

import tiktoken


def calculate_text_generation_costs(
    input: str,
    response: str,
    cost_per_million_input_tokens: float,
    cost_per_million_output_tokens: float,
    approximate: bool = True
) -> dict:
    """
    Calculate the cost of an LLM text generation request.

    :param input: The input text/prompt sent to the LLM.
    :param response: The response text received from the LLM.
    :param cost_per_million_input_tokens: Cost per million input tokens (e.g., 0.50 for $0.50/M).
    :param cost_per_million_output_tokens: Cost per million output tokens.
    :param approximate: If True, uses fast approximation (len/4). If False, uses tiktoken for exact count.

    :return: Dictionary with token counts and costs:
        - input_tokens: Number of input tokens
        - output_tokens: Number of output tokens
        - input_cost: Cost for input tokens
        - output_cost: Cost for output tokens
        - total_cost: Total cost (input + output)

    Example:
        >>> costs = calculate_text_generation_costs(
        ...     input="Hello, how are you?",
        ...     response="I'm doing well, thank you!",
        ...     cost_per_million_input_tokens=0.50,
        ...     cost_per_million_output_tokens=1.50,
        ...     approximate=True
        ... )
        >>> print(f"Total cost: ${costs['total_cost']:.6f}")
    """
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
