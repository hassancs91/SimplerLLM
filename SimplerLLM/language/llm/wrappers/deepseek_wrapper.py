"""
DeepSeek LLM Wrapper - High-level interface for DeepSeek models.

This module provides a user-friendly wrapper around DeepSeek's API,
supporting both standard chat and reasoning models.

Supported Models:
    - deepseek-chat: Standard chat model (DeepSeek V3.2)
    - deepseek-reasoner: Reasoning model with chain-of-thought (R1-based)

Features:
    - Basic text generation with temperature/top_p control
    - JSON mode for structured output
    - Reasoning model support with thinking content extraction
    - Async support for non-blocking operations

Not Supported (DeepSeek API limitations):
    - Vision/Image input (no multimodal support)
    - Web search integration

Example:
    >>> from SimplerLLM.language import LLM, LLMProvider
    >>>
    >>> # Create DeepSeek instance
    >>> llm = LLM.create(provider=LLMProvider.DEEPSEEK, model_name="deepseek-chat")
    >>>
    >>> # Basic generation
    >>> response = llm.generate_response(
    ...     prompt="What is the capital of France?",
    ...     max_tokens=100
    ... )
    >>>
    >>> # Reasoning model
    >>> llm = LLM.create(provider=LLMProvider.DEEPSEEK, model_name="deepseek-reasoner")
    >>> response = llm.generate_response(
    ...     prompt="What is 15! / 13!?",
    ...     max_tokens=8000,
    ...     thinking=True,
    ...     full_response=True
    ... )
    >>> print(f"Answer: {response.generated_text}")
    >>> print(f"Reasoning: {response.thinking_content[:200]}...")
"""

import SimplerLLM.language.llm_providers.deepseek_llm as deepseek_llm
import os
import warnings
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print


class DeepSeekLLM(LLM):
    """
    High-level wrapper for DeepSeek language models.

    This class provides a clean interface for interacting with DeepSeek's
    API, handling message formatting, parameter validation, and response
    processing.

    Attributes:
        api_key: The DeepSeek API key for authentication.
        model_name: The default model to use for generation.
        temperature: Default sampling temperature.
        top_p: Default nucleus sampling parameter.
        verbose: Whether to print debug information.

    Example:
        >>> llm = DeepSeekLLM(
        ...     provider=LLMProvider.DEEPSEEK,
        ...     model_name="deepseek-chat",
        ...     temperature=0.7,
        ...     top_p=1.0,
        ...     api_key="your-api-key"
        ... )
        >>> response = llm.generate_response(prompt="Hello!")
    """

    def __init__(self, provider, model_name, temperature, top_p, api_key, verbose=False):
        """
        Initialize the DeepSeek LLM wrapper.

        Args:
            provider: The LLMProvider enum value (should be DEEPSEEK).
            model_name: The model to use (e.g., 'deepseek-chat', 'deepseek-reasoner').
            temperature: Default sampling temperature (0-2).
            top_p: Default nucleus sampling parameter (0-1).
            api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
            verbose: If True, print debug information during operations.
        """
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")

    def append_messages(self, system_prompt: str, messages: list):
        """
        Prepend system prompt to the message list.

        Args:
            system_prompt: The system instructions to prepend.
            messages: List of message dictionaries.

        Returns:
            List of messages with system prompt prepended.
        """
        model_messages = [{"role": "system", "content": system_prompt}]
        if messages:
            model_messages.extend(messages)
        return model_messages

    def generate_response(
        self,
        model_name: str = None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        json_mode: bool = False,
        thinking: bool = None,
        images: list = None,
        detail: str = "auto",
        web_search: bool = False,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ):
        """
        Generate a response using DeepSeek.

        Supports both standard chat (deepseek-chat) and reasoning (deepseek-reasoner)
        models. For reasoning models, use the `thinking` parameter to control
        chain-of-thought behavior.

        Args:
            model_name: Model to use. Defaults to instance's model_name.
                Options: 'deepseek-chat', 'deepseek-reasoner'
            prompt: A single prompt string for simple requests.
            messages: List of message dicts for multi-turn conversations.
                Cannot be used with `prompt`.
            system_prompt: System instructions. Default: "You are a helpful AI Assistant"
            temperature: Sampling temperature (0-2). Higher = more creative. Default: 0.7
            max_tokens: Maximum tokens to generate. For reasoning models,
                recommend 4000-8000 to allow space for thinking. Default: 300
            top_p: Nucleus sampling (0-1). Default: 1.0
            full_response: If True, returns LLMFullResponse with metadata
                including reasoning_tokens and thinking_content. Default: False
            json_mode: If True, forces JSON output format. Default: False
            thinking: For deepseek-reasoner, enable (True) or disable (False)
                chain-of-thought reasoning. Default: None (model decides)
            images: NOT SUPPORTED - DeepSeek API doesn't support vision.
                Will log a warning if provided.
            detail: NOT SUPPORTED - Ignored (vision not supported).
            web_search: NOT SUPPORTED - DeepSeek API doesn't support web search.
                Will log a warning if True.

        Returns:
            str: Generated text (if full_response=False)
            LLMFullResponse: Full response object with metadata (if full_response=True)
                - generated_text: The model's response
                - reasoning_tokens: Tokens used for reasoning (reasoner model)
                - thinking_content: Chain-of-thought content (reasoner model)
                - is_reasoning_model: True for deepseek-reasoner

        Raises:
            ValueError: If both prompt and messages are provided, or neither.

        Example:
            >>> # Basic usage
            >>> response = llm.generate_response(
            ...     prompt="What is Python?",
            ...     max_tokens=200
            ... )

            >>> # Reasoning model
            >>> response = llm.generate_response(
            ...     model_name="deepseek-reasoner",
            ...     prompt="Solve: What is 15 factorial divided by 13 factorial?",
            ...     thinking=True,
            ...     max_tokens=8000,
            ...     full_response=True
            ... )
            >>> print(f"Reasoning tokens: {response.reasoning_tokens}")
        """
        params = self.prepare_params(model_name, temperature, top_p)

        # Warn about unsupported features
        if images:
            warning_msg = "DeepSeek API does not support vision/images. The 'images' parameter will be ignored."
            warnings.warn(warning_msg, UserWarning)
            if self.verbose:
                verbose_print(warning_msg, "warning")

        if web_search:
            warning_msg = "DeepSeek API does not support web search. The 'web_search' parameter will be ignored."
            warnings.warn(warning_msg, UserWarning)
            if self.verbose:
                verbose_print(warning_msg, "warning")

        # Validate inputs
        if prompt and messages:
            if self.verbose:
                verbose_print("Error: Both prompt and messages provided", "error")
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            if self.verbose:
                verbose_print("Error: Neither prompt nor messages provided", "error")
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
                verbose_print(f"System prompt: {system_prompt}", "debug")
                verbose_print(f"User prompt: {prompt[:100]}...", "debug")
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = self.append_messages(system_prompt, messages)

        params.update({
            "api_key": self.api_key,
            "messages": model_messages,
            "max_tokens": max_tokens,
            "full_response": full_response,
            "json_mode": json_mode,
            "thinking": thinking,
        })

        if self.verbose:
            verbose_print(f"Generating response with DeepSeek ({params.get('model_name', self.model_name)})...", "info")

        try:
            response = deepseek_llm.generate_response(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
                if full_response and hasattr(response, 'reasoning_tokens') and response.reasoning_tokens:
                    verbose_print(f"Reasoning tokens used: {response.reasoning_tokens}", "debug")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise

    async def generate_response_async(
        self,
        model_name: str = None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        json_mode: bool = False,
        thinking: bool = None,
        images: list = None,
        detail: str = "auto",
        web_search: bool = False,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ):
        """
        Asynchronously generate a response using DeepSeek.

        This is the async version of generate_response(). See generate_response()
        for full parameter documentation.

        Args:
            model_name: Model to use. Defaults to instance's model_name.
            prompt: A single prompt string for simple requests.
            messages: List of message dicts for multi-turn conversations.
            system_prompt: System instructions.
            temperature: Sampling temperature (0-2). Default: 0.7
            max_tokens: Maximum tokens to generate. Default: 300
            top_p: Nucleus sampling (0-1). Default: 1.0
            full_response: If True, returns LLMFullResponse. Default: False
            json_mode: If True, forces JSON output. Default: False
            thinking: For deepseek-reasoner, enable/disable thinking mode.
            images: NOT SUPPORTED - Will log warning if provided.
            detail: NOT SUPPORTED - Ignored.
            web_search: NOT SUPPORTED - Will log warning if True.

        Returns:
            str or LLMFullResponse: The generated response.

        Raises:
            ValueError: If both prompt and messages are provided, or neither.
        """
        params = self.prepare_params(model_name, temperature, top_p)

        # Warn about unsupported features
        if images:
            warning_msg = "DeepSeek API does not support vision/images. The 'images' parameter will be ignored."
            warnings.warn(warning_msg, UserWarning)
            if self.verbose:
                verbose_print(warning_msg, "warning")

        if web_search:
            warning_msg = "DeepSeek API does not support web search. The 'web_search' parameter will be ignored."
            warnings.warn(warning_msg, UserWarning)
            if self.verbose:
                verbose_print(warning_msg, "warning")

        # Validate inputs
        if prompt and messages:
            if self.verbose:
                verbose_print("Error: Both prompt and messages provided", "error")
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            if self.verbose:
                verbose_print("Error: Neither prompt nor messages provided", "error")
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
                verbose_print(f"System prompt: {system_prompt}", "debug")
                verbose_print(f"User prompt: {prompt[:100]}...", "debug")
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = self.append_messages(system_prompt, messages)

        params.update({
            "api_key": self.api_key,
            "messages": model_messages,
            "max_tokens": max_tokens,
            "full_response": full_response,
            "json_mode": json_mode,
            "thinking": thinking,
        })

        if self.verbose:
            verbose_print(f"Generating response with DeepSeek (async) ({params.get('model_name', self.model_name)})...", "info")

        try:
            response = await deepseek_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
                if full_response and hasattr(response, 'reasoning_tokens') and response.reasoning_tokens:
                    verbose_print(f"Reasoning tokens used: {response.reasoning_tokens}", "debug")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
