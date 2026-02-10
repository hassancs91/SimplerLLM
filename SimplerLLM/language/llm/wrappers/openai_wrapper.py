"""
OpenAI LLM Wrapper - High-level interface for OpenAI models.

This wrapper provides a convenient, unified interface for interacting with
all OpenAI language models, including support for:

- All GPT models (GPT-4, GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Reasoning models (o1, o1-mini, o1-preview, o3, o3-mini, GPT-5 series)
- Vision capabilities (image analysis)
- Web search via OpenAI's Responses API
- JSON mode for structured outputs

The wrapper automatically handles model-specific constraints:
- Reasoning models: Uses max_completion_tokens, adjusts temperature
- o1 models: Converts system messages to user messages automatically
- Vision models: Formats images for the API

Example:
    >>> from SimplerLLM.language.llm import LLM, LLMProvider
    >>>
    >>> # Create instance
    >>> llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    >>>
    >>> # Simple text generation
    >>> response = llm.generate_response(prompt="Explain quantum computing")
    >>>
    >>> # With vision
    >>> response = llm.generate_response(
    ...     prompt="What's in this image?",
    ...     images=["https://example.com/image.jpg"],
    ...     detail="high"
    ... )
    >>>
    >>> # With reasoning model
    >>> llm = LLM.create(provider=LLMProvider.OPENAI, model_name="o1-mini")
    >>> response = llm.generate_response(
    ...     prompt="Solve this complex math problem...",
    ...     max_tokens=8000,
    ...     reasoning_effort="high",
    ...     full_response=True
    ... )
    >>> print(f"Used {response.reasoning_tokens} reasoning tokens")
"""

import SimplerLLM.language.llm_providers.openai_llm as openai_llm
import os
from typing import Optional, List, Literal, Union
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print
from SimplerLLM.tools.image_helpers import prepare_vision_content


class OpenAILLM(LLM):
    """
    OpenAI LLM wrapper supporting all OpenAI models.

    This class provides a high-level interface for OpenAI's language models,
    handling parameter formatting, vision content preparation, and model-specific
    constraints automatically.

    Attributes:
        provider: The LLM provider enum value (LLMProvider.OPENAI).
        model_name: Default model to use for generation.
        temperature: Default temperature setting (0-2).
        top_p: Default top_p setting (0-1).
        api_key: OpenAI API key.
        verbose: Enable verbose logging.

    Example:
        >>> llm = OpenAILLM(
        ...     provider=LLMProvider.OPENAI,
        ...     model_name="gpt-4o-mini",
        ...     temperature=0.7,
        ...     top_p=1.0,
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
        Initialize the OpenAI LLM wrapper.

        Args:
            provider: LLMProvider enum value.
            model_name: Default model name (e.g., 'gpt-4o', 'o1-mini').
            temperature: Default temperature (0-2).
            top_p: Default top_p value (0-1).
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            verbose: Enable verbose logging.
        """
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

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
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        json_mode: bool = False,
        images: Optional[List[str]] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        web_search: bool = False,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        timeout: Optional[float] = None,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ) -> Union[str, "LLMFullResponse"]:
        """
        Generate a response using an OpenAI model.

        This method supports all OpenAI models including reasoning models
        (o1, o3, GPT-5) and automatically handles their special requirements.

        Args:
            model_name: Override the default model. Defaults to instance model.
            prompt: Simple text prompt. Cannot be used with messages.
            messages: List of message dicts for multi-turn conversations.
                Cannot be used with prompt.
            system_prompt: System message to set context. Note: Some models
                (o1-mini, o1-preview) don't support system messages - they'll
                be converted to user messages automatically. Default:
                "You are a helpful AI Assistant"
            temperature: Sampling temperature (0-2). Ignored for some
                reasoning models. Default: 0.7
            max_tokens: Maximum tokens to generate. For reasoning models,
                this includes reasoning tokens. Default: 300 (recommend
                4000+ for reasoning models - applied automatically when
                using default).
            top_p: Nucleus sampling parameter. Default: 1.0
            full_response: Return LLMFullResponse with metadata including
                token counts, timing, and reasoning information. Default: False
            json_mode: Force JSON output format. Default: False
            images: List of image URLs or file paths for vision models.
                Supports PNG, JPEG, GIF, WebP. Max 20MB per image.
            detail: Image detail level. Default: "auto"
                - "low": Fast processing, ~85 tokens per image
                - "high": Detailed analysis, ~765+ tokens per image
                - "auto": Let API decide based on image size
            web_search: Enable web search via Responses API. Default: False
            reasoning_effort: Control reasoning depth for thinking models.
                Options: "low", "medium", "high". Default: None (model default)
                Only applies to reasoning models (o1, o3, GPT-5).
            timeout: Request timeout in seconds. Default: None (no timeout)

        Returns:
            str: Generated text if full_response=False
            LLMFullResponse: Full response with metadata if full_response=True
                - generated_text: The model's response
                - model: Model name used
                - process_time: Time taken in seconds
                - input_token_count: Prompt tokens used
                - output_token_count: Completion tokens used
                - reasoning_tokens: Tokens used for reasoning (reasoning models)
                - finish_reason: Why generation stopped
                - is_reasoning_model: True if reasoning model was used
                - web_sources: List of sources (if web_search=True)

        Raises:
            ValueError: If both prompt and messages provided, or neither

        Example:
            >>> # Simple text generation
            >>> response = llm.generate_response(prompt="Hello!")

            >>> # With vision
            >>> response = llm.generate_response(
            ...     prompt="Describe this image",
            ...     images=["path/to/image.jpg", "https://example.com/img.png"],
            ...     detail="high"
            ... )

            >>> # Reasoning model with full response
            >>> response = llm.generate_response(
            ...     model_name="o1-mini",
            ...     prompt="Prove that there are infinitely many primes",
            ...     max_tokens=10000,
            ...     reasoning_effort="high",
            ...     full_response=True
            ... )
            >>> print(f"Reasoning tokens: {response.reasoning_tokens}")

            >>> # With web search
            >>> response = llm.generate_response(
            ...     prompt="What are the latest AI developments?",
            ...     web_search=True,
            ...     full_response=True
            ... )
            >>> for source in response.web_sources or []:
            ...     print(f"Source: {source['title']}")
        """
        # Handle web search mode - uses Responses API
        if web_search:
            if not prompt and not messages:
                raise ValueError("Either 'prompt' or 'messages' must be provided.")

            # Combine system prompt and user input for web search
            if prompt:
                input_text = f"{system_prompt}\n\n{prompt}"
            else:
                # Convert messages to a single input string
                input_text = system_prompt + "\n\n"
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    input_text += f"{role}: {content}\n"

            effective_model = model_name if model_name else self.model_name

            if self.verbose:
                verbose_print(f"Generating response with web search using {effective_model}...", "info")

            try:
                response = openai_llm.generate_response_with_web_search(
                    model_name=effective_model,
                    input_text=input_text,
                    max_tokens=max_tokens,
                    full_response=full_response,
                    api_key=self.api_key,
                )
                if self.verbose:
                    verbose_print("Web search response received successfully", "info")
                return response
            except Exception as e:
                if self.verbose:
                    verbose_print(f"Error generating web search response: {str(e)}", "error")
                raise

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
                "reasoning_effort": reasoning_effort,
                "timeout": timeout,
            }
        )

        if self.verbose:
            verbose_print("Generating response with OpenAI...", "info")
            if reasoning_effort:
                verbose_print(f"Reasoning effort: {reasoning_effort}", "debug")

        try:
            response = openai_llm.generate_response(**params)
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
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        timeout: Optional[float] = None,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ) -> Union[str, "LLMFullResponse"]:
        """
        Asynchronously generate a response using an OpenAI model.

        This is the async version of generate_response(). It supports all
        OpenAI models including reasoning models (o1, o3, GPT-5).

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
            detail: Image detail level ("low", "high", "auto"). Default: "auto"
            web_search: Enable web search via Responses API. Default: False
            reasoning_effort: Control reasoning depth for thinking models.
                Options: "low", "medium", "high". Default: None
            timeout: Request timeout in seconds. Default: None

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
        # Handle web search mode - uses Responses API
        if web_search:
            if not prompt and not messages:
                raise ValueError("Either 'prompt' or 'messages' must be provided.")

            # Combine system prompt and user input for web search
            if prompt:
                input_text = f"{system_prompt}\n\n{prompt}"
            else:
                # Convert messages to a single input string
                input_text = system_prompt + "\n\n"
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    input_text += f"{role}: {content}\n"

            effective_model = model_name if model_name else self.model_name

            if self.verbose:
                verbose_print(f"Generating response with web search (async) using {effective_model}...", "info")

            try:
                response = await openai_llm.generate_response_with_web_search_async(
                    model_name=effective_model,
                    input_text=input_text,
                    max_tokens=max_tokens,
                    full_response=full_response,
                    api_key=self.api_key,
                )
                if self.verbose:
                    verbose_print("Web search response received successfully", "info")
                return response
            except Exception as e:
                if self.verbose:
                    verbose_print(f"Error generating web search response: {str(e)}", "error")
                raise

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
                "reasoning_effort": reasoning_effort,
                "timeout": timeout,
            }
        )

        if self.verbose:
            verbose_print("Generating response with OpenAI (async)...", "info")
            if reasoning_effort:
                verbose_print(f"Reasoning effort: {reasoning_effort}", "debug")

        try:
            response = await openai_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
