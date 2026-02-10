"""
Cohere LLM Wrapper - High-level interface for Cohere models.

This wrapper provides a convenient, unified interface for interacting with
all Cohere language models, including support for:

- All Command models (Command A, Command R+, Command R)
- Reasoning models (Command-A-Reasoning)
- Vision capabilities (Command-A-Vision, Aya Vision)
- JSON mode for structured outputs

Example:
    >>> from SimplerLLM.language.llm import LLM, LLMProvider
    >>>
    >>> # Create instance
    >>> llm = LLM.create(provider=LLMProvider.COHERE, model_name="command-a-03-2025")
    >>>
    >>> # Simple text generation
    >>> response = llm.generate_response(prompt="Explain quantum computing")
    >>>
    >>> # With vision
    >>> response = llm.generate_response(
    ...     model_name="command-a-vision-07-2025",
    ...     prompt="What's in this image?",
    ...     images=["photo.jpg"],
    ... )
"""

import SimplerLLM.language.llm_providers.cohere_llm as cohere_llm
import os
from typing import Optional, List, Literal, Union
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print
from SimplerLLM.tools.image_helpers import prepare_vision_content_cohere


class CohereLLM(LLM):
    """
    Cohere LLM wrapper supporting all Cohere Command models.

    This class provides a high-level interface for Cohere's language models,
    handling parameter formatting, vision content preparation, and model-specific
    constraints automatically.

    Attributes:
        provider: The LLM provider enum value (LLMProvider.COHERE).
        model_name: Default model to use for generation.
        temperature: Default temperature setting (0-2).
        top_p: Default top_p setting (0-1).
        api_key: Cohere API key.
        verbose: Enable verbose logging.
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
        Initialize the Cohere LLM wrapper.

        Args:
            provider: LLMProvider enum value.
            model_name: Default model name (e.g., 'command-a-03-2025').
            temperature: Default temperature (0-2).
            top_p: Default top_p value (0-1).
            api_key: Cohere API key. Falls back to COHERE_API_KEY env var.
            verbose: Enable verbose logging.
        """
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("COHERE_API_KEY", "")

    def append_messages(self, messages: list) -> list:
        """
        Process and return messages list.

        Args:
            messages: List of message dictionaries.

        Returns:
            List[dict]: Processed messages.
        """
        model_messages = []
        if messages:
            model_messages.extend(messages)
        return model_messages

    def generate_response(
        self,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        json_mode: bool = False,
        images: Optional[List[str]] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        web_search: bool = False,
        timeout: Optional[float] = None,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ) -> Union[str, "cohere_llm.LLMFullResponse"]:
        """
        Generate a response using a Cohere model.

        This method supports all Cohere models including reasoning models
        and vision models.

        Args:
            model_name: Override the default model. Defaults to instance model.
            prompt: Simple text prompt. Cannot be used with messages.
            messages: List of message dicts for multi-turn conversations.
            system_prompt: System message to set context. Default:
                "You are a helpful AI Assistant"
            temperature: Sampling temperature (0-2). Default: 0.7
            max_tokens: Maximum tokens to generate. Default: 300
            top_p: Nucleus sampling parameter. Default: 1.0
            full_response: Return LLMFullResponse with metadata. Default: False
            json_mode: Force JSON output format. Default: False
            images: List of image URLs or file paths for vision models.
                Requires a vision-capable model like 'command-a-vision-07-2025'.
            detail: Image detail level (kept for API compatibility). Default: "auto"
            web_search: Not supported by Cohere API. Included for interface
                compatibility. Will be ignored with a warning.
            timeout: Request timeout in seconds. Default: None (no timeout)

        Returns:
            str: Generated text if full_response=False
            LLMFullResponse: Full response with metadata if full_response=True

        Raises:
            ValueError: If both prompt and messages provided, or neither

        Example:
            >>> # Simple text generation
            >>> response = llm.generate_response(prompt="Hello!")

            >>> # With vision
            >>> response = llm.generate_response(
            ...     model_name="command-a-vision-07-2025",
            ...     prompt="Describe this image",
            ...     images=["path/to/image.jpg"],
            ...     max_tokens=200
            ... )
        """
        params = self.prepare_params(model_name, temperature, top_p)

        # Warn about unsupported web_search
        if web_search:
            if self.verbose:
                verbose_print(
                    "Warning: web_search is not supported by Cohere API. "
                    "Use Cohere's connectors for search functionality.",
                    "warning"
                )

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
                verbose_print(f"User prompt: {prompt}", "debug")
                if images:
                    verbose_print(f"Images provided: {len(images)} image(s)", "debug")

            # Handle vision content if images are provided
            if images:
                user_content = prepare_vision_content_cohere(prompt, images, detail)
            else:
                user_content = prompt

            model_messages = [
                {"role": "user", "content": user_content},
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
                if images:
                    verbose_print(
                        "Warning: Images parameter is ignored when using messages format. "
                        "Include images directly in the message content.",
                        "warning"
                    )
            model_messages = self.append_messages(messages)

        params.update(
            {
                "api_key": self.api_key,
                "system_prompt": system_prompt,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "json_mode": json_mode,
                "timeout": timeout,
            }
        )

        if self.verbose:
            verbose_print("Generating response with Cohere...", "info")

        try:
            response = cohere_llm.generate_response(**params)
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
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        json_mode: bool = False,
        images: Optional[List[str]] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        web_search: bool = False,
        timeout: Optional[float] = None,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ) -> Union[str, "cohere_llm.LLMFullResponse"]:
        """
        Asynchronously generate a response using a Cohere model.

        See generate_response() for full parameter documentation.

        Example:
            >>> import asyncio
            >>> response = asyncio.run(llm.generate_response_async(
            ...     prompt="Hello!",
            ...     max_tokens=100
            ... ))
        """
        params = self.prepare_params(model_name, temperature, top_p)

        # Warn about unsupported web_search
        if web_search:
            if self.verbose:
                verbose_print(
                    "Warning: web_search is not supported by Cohere API",
                    "warning"
                )

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
                verbose_print("Preparing single prompt message (async)", "debug")
                if images:
                    verbose_print(f"Images provided: {len(images)} image(s)", "debug")

            if images:
                user_content = prepare_vision_content_cohere(prompt, images, detail)
            else:
                user_content = prompt

            model_messages = [
                {"role": "user", "content": user_content},
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages (async)", "debug")
                if images:
                    verbose_print(
                        "Warning: Images parameter is ignored when using messages format",
                        "warning"
                    )
            model_messages = self.append_messages(messages)

        params.update(
            {
                "api_key": self.api_key,
                "system_prompt": system_prompt,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "json_mode": json_mode,
                "timeout": timeout,
            }
        )

        if self.verbose:
            verbose_print("Generating response with Cohere (async)...", "info")

        try:
            response = await cohere_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully (async)", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response (async): {str(e)}", "error")
            raise
