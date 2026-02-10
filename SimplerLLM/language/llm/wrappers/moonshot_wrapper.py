"""
Moonshot LLM Wrapper - High-level interface for Moonshot AI models.

This wrapper provides a convenient, unified interface for interacting with
all Moonshot/Kimi language models, including support for:

- V1 models (moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k)
- Vision models (moonshot-v1-*-vision-preview)
- K2 models (kimi-k2-0905-preview, kimi-k2-thinking)
- Latest models (kimi-latest-8k, kimi-latest-32k, kimi-latest-128k)

The wrapper automatically handles Kimi-specific constraints:
- Temperature is clamped to [0, 1] range (Kimi doesn't support 0-2)
- Thinking models return reasoning content
- Vision uses OpenAI-compatible format

Note: Web search is NOT supported by Kimi API.

Example:
    >>> from SimplerLLM.language.llm import LLM, LLMProvider
    >>>
    >>> # Create instance
    >>> llm = LLM.create(provider=LLMProvider.MOONSHOT, model_name="kimi-k2-thinking")
    >>>
    >>> # Simple text generation
    >>> response = llm.generate_response(prompt="Explain quantum computing")
    >>>
    >>> # With vision (vision-preview models)
    >>> llm = LLM.create(provider=LLMProvider.MOONSHOT, model_name="moonshot-v1-128k-vision-preview")
    >>> response = llm.generate_response(
    ...     prompt="What's in this image?",
    ...     images=["photo.jpg"],
    ...     detail="high"
    ... )
    >>>
    >>> # With thinking model
    >>> llm = LLM.create(provider=LLMProvider.MOONSHOT, model_name="kimi-k2-thinking")
    >>> response = llm.generate_response(
    ...     prompt="Solve this complex math problem...",
    ...     max_tokens=8000,
    ...     full_response=True
    ... )
    >>> print(f"Thinking: {response.thinking_content}")
    >>> print(f"Answer: {response.generated_text}")
"""

import SimplerLLM.language.llm_providers.moonshot_llm as moonshot_llm
import os
from typing import Optional, List, Literal, Union
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print
from SimplerLLM.tools.image_helpers import prepare_vision_content


class MoonshotLLM(LLM):
    """
    Moonshot LLM wrapper supporting all Moonshot/Kimi models.

    This class provides a high-level interface for Moonshot AI's language models,
    handling parameter formatting, vision content preparation, and model-specific
    constraints automatically.

    Attributes:
        provider: The LLM provider enum value (LLMProvider.MOONSHOT).
        model_name: Default model to use for generation.
        temperature: Default temperature setting (0-1).
        top_p: Default top_p setting (0-1).
        api_key: Moonshot API key.
        verbose: Enable verbose logging.

    Example:
        >>> llm = MoonshotLLM(
        ...     provider=LLMProvider.MOONSHOT,
        ...     model_name="kimi-k2-instruct",
        ...     temperature=0.6,
        ...     top_p=0.95,
        ...     api_key="your-api-key"
        ... )
        >>> response = llm.generate_response(prompt="Hello!")
    """

    def __init__(
        self,
        provider,
        model_name: str,
        temperature: float,
        top_p: float,
        api_key: Optional[str],
        verbose: bool = False
    ):
        """
        Initialize the Kimi LLM wrapper.

        Args:
            provider: LLMProvider enum value.
            model_name: Default model name (e.g., 'kimi-k2-instruct', 'kimi-k2-thinking').
            temperature: Default temperature (0-1). Values > 1 are auto-clamped.
            top_p: Default top_p value (0-1). Recommended: 0.95.
            api_key: Moonshot API key. Falls back to MOONSHOT_API_KEY env var.
            verbose: Enable verbose logging.
        """
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY", "")

    def append_messages(self, system_prompt: str, messages: List[dict]) -> List[dict]:
        """
        Append system prompt to messages list.

        Args:
            system_prompt: The system message content.
            messages: List of message dictionaries.

        Returns:
            List[dict]: Messages with system prompt prepended.
        """
        model_messages = [{"role": "system", "content": system_prompt}]
        if messages:
            model_messages.extend(messages)
        return model_messages

    def generate_response(
        self,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.6,
        max_tokens: int = 300,
        top_p: float = 0.95,
        full_response: bool = False,
        json_mode: bool = False,
        images: Optional[List[str]] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
        web_search: bool = False,
        **kwargs,
    ) -> Union[str, "LLMFullResponse"]:
        """
        Generate a response using a Kimi/Moonshot model.

        This method supports all Kimi models including thinking/reasoning models
        (K2-Thinking, K2.5-Thinking) and vision models.

        Args:
            model_name: Override the default model. Defaults to instance model.
            prompt: Simple text prompt. Cannot be used with messages.
            messages: List of message dicts for multi-turn conversations.
                Cannot be used with prompt.
            system_prompt: System message to set context. Default:
                "You are a helpful AI Assistant"
            temperature: Sampling temperature (0-1). Values > 1 are auto-clamped.
                Recommended: 0.6 for standard, 1.0 for thinking. Default: 0.6
            max_tokens: Maximum tokens to generate. Default: 300 (recommend
                8000+ for thinking models).
            top_p: Nucleus sampling parameter. Recommended: 0.95. Default: 0.95
            full_response: Return LLMFullResponse with metadata including
                token counts, timing, and thinking information. Default: False
            json_mode: Force JSON output format. Default: False
            images: List of image URLs or file paths for vision models.
                Supports PNG, JPEG, WebP. Requires vision-capable model.
            detail: Image detail level. Default: "auto"
                - "low": Fast processing
                - "high": Detailed analysis
                - "auto": Let API decide based on image size
            thinking: For thinking models, explicitly enable/disable thinking.
                None uses model default. Set False to disable for faster responses.
            timeout: Request timeout in seconds. Default: None (no timeout)
            web_search: NOT SUPPORTED by Kimi. Will log a warning if True.

        Returns:
            str: Generated text if full_response=False
            LLMFullResponse: Full response with metadata if full_response=True
                - generated_text: The model's response
                - model: Model name used
                - process_time: Time taken in seconds
                - input_token_count: Prompt tokens used
                - output_token_count: Completion tokens used
                - thinking_content: Reasoning process (thinking models)
                - reasoning_tokens: Tokens used for reasoning
                - finish_reason: Why generation stopped
                - is_reasoning_model: True if thinking model was used

        Raises:
            ValueError: If both prompt and messages provided, or neither

        Example:
            >>> # Simple text generation
            >>> response = llm.generate_response(prompt="Hello!")

            >>> # With vision (K2.5 or vision models)
            >>> response = llm.generate_response(
            ...     prompt="Describe this image",
            ...     images=["path/to/image.jpg", "https://example.com/img.png"],
            ...     detail="high"
            ... )

            >>> # Thinking model with full response
            >>> response = llm.generate_response(
            ...     model_name="kimi-k2-thinking",
            ...     prompt="Prove that there are infinitely many primes",
            ...     max_tokens=10000,
            ...     full_response=True
            ... )
            >>> print(f"Thinking: {response.thinking_content}")
        """
        # Warn about unsupported web_search
        if web_search:
            if self.verbose:
                verbose_print("Warning: web_search is not supported by Kimi API and will be ignored", "warning")

        params = self.prepare_params(model_name, temperature, top_p)

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
                verbose_print(f"User prompt: {prompt}", "debug")
                if images:
                    verbose_print(f"Images provided: {len(images)} image(s)", "debug")

            # Handle vision content if images are provided
            # Kimi uses OpenAI-compatible format
            if images:
                user_content = prepare_vision_content(prompt, images, detail)
            else:
                user_content = prompt

            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
                if images:
                    verbose_print("Warning: Images parameter is ignored when using messages format", "warning")
            model_messages = self.append_messages(system_prompt, messages)

        params.update(
            {
                "api_key": self.api_key,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "json_mode": json_mode,
                "thinking": thinking,
                "timeout": timeout,
            }
        )

        if self.verbose:
            verbose_print("Generating response with Kimi...", "info")
            if thinking is not None:
                verbose_print(f"Thinking mode: {'enabled' if thinking else 'disabled'}", "debug")

        try:
            response = moonshot_llm.generate_response(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise

    async def generate_response_async(
        self,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.6,
        max_tokens: int = 300,
        top_p: float = 0.95,
        full_response: bool = False,
        json_mode: bool = False,
        images: Optional[List[str]] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
        web_search: bool = False,
        **kwargs,
    ) -> Union[str, "LLMFullResponse"]:
        """
        Asynchronously generate a response using a Kimi/Moonshot model.

        This is the async version of generate_response(). It supports all
        Kimi models including thinking/reasoning models and vision.

        Args:
            model_name: Override the default model. Defaults to instance model.
            prompt: Simple text prompt. Cannot be used with messages.
            messages: List of message dicts for multi-turn conversations.
            system_prompt: System message to set context. Default:
                "You are a helpful AI Assistant"
            temperature: Sampling temperature (0-1). Default: 0.6
            max_tokens: Maximum tokens to generate. Default: 300
            top_p: Nucleus sampling parameter. Default: 0.95
            full_response: Return LLMFullResponse with metadata. Default: False
            json_mode: Force JSON output format. Default: False
            images: List of image URLs or file paths for vision models.
            detail: Image detail level ("low", "high", "auto"). Default: "auto"
            thinking: Enable/disable thinking for thinking models. Default: None
            timeout: Request timeout in seconds. Default: None
            web_search: NOT SUPPORTED by Kimi. Will log a warning if True.

        Returns:
            str: Generated text if full_response=False
            LLMFullResponse: Full response with metadata if full_response=True

        Raises:
            ValueError: If both prompt and messages provided, or neither

        Example:
            >>> # Async usage
            >>> response = await llm.generate_response_async(
            ...     prompt="Hello!",
            ...     max_tokens=100
            ... )

        See Also:
            generate_response: Synchronous version of this method.
        """
        # Warn about unsupported web_search
        if web_search:
            if self.verbose:
                verbose_print("Warning: web_search is not supported by Kimi API and will be ignored", "warning")

        params = self.prepare_params(model_name, temperature, top_p)

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
                verbose_print(f"User prompt: {prompt}", "debug")
                if images:
                    verbose_print(f"Images provided: {len(images)} image(s)", "debug")

            # Handle vision content if images are provided
            # Kimi uses OpenAI-compatible format
            if images:
                user_content = prepare_vision_content(prompt, images, detail)
            else:
                user_content = prompt

            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
                if images:
                    verbose_print("Warning: Images parameter is ignored when using messages format", "warning")
            model_messages = self.append_messages(system_prompt, messages)

        params.update(
            {
                "api_key": self.api_key,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "json_mode": json_mode,
                "thinking": thinking,
                "timeout": timeout,
            }
        )

        if self.verbose:
            verbose_print("Generating response with Kimi (async)...", "info")
            if thinking is not None:
                verbose_print(f"Thinking mode: {'enabled' if thinking else 'disabled'}", "debug")

        try:
            response = await moonshot_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
