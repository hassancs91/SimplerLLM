"""
Perplexity LLM Wrapper - High-level interface for Perplexity AI models.

This wrapper provides a convenient, unified interface for interacting with
Perplexity's language models, including support for:

- All Sonar models (sonar, sonar-pro, sonar-reasoning-pro, sonar-deep-research)
- Built-in web search with citations (always enabled)
- Vision capabilities (image analysis)
- Domain and recency filtering for search
- JSON mode for structured outputs

Key Feature: Built-in Web Search
    Unlike other providers where web_search is optional, Perplexity models
    ALWAYS perform web search. Every response includes citations from
    searched sources, accessible via web_sources in full_response mode.

Example:
    >>> from SimplerLLM.language.llm import LLM, LLMProvider
    >>>
    >>> # Create instance
    >>> llm = LLM.create(provider=LLMProvider.PERPLEXITY, model_name="sonar-pro")
    >>>
    >>> # Simple generation (automatically searches the web)
    >>> response = llm.generate_response(prompt="What are the latest AI trends?")
    >>>
    >>> # With citations
    >>> response = llm.generate_response(
    ...     prompt="Recent developments in quantum computing",
    ...     search_recency_filter="month",
    ...     full_response=True
    ... )
    >>> for source in response.web_sources or []:
    ...     print(f"- {source['url']}")
    >>>
    >>> # With vision
    >>> response = llm.generate_response(
    ...     prompt="What's in this image?",
    ...     images=["https://example.com/image.jpg"],
    ...     detail="high"
    ... )
"""

import SimplerLLM.language.llm_providers.perplexity_llm as perplexity_llm
import os
from typing import Optional, List, Literal, Union
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print
from SimplerLLM.tools.image_helpers import prepare_vision_content


class PerplexityLLM(LLM):
    """
    Perplexity LLM wrapper supporting all Perplexity models.

    This class provides a high-level interface for Perplexity's language models,
    handling parameter formatting, vision content preparation, and automatic
    web search citation extraction.

    Key Difference from Other Providers:
        Perplexity has built-in web search - it's not optional. Every request
        performs a web search and returns citations. The web_search parameter
        is accepted for API compatibility but is effectively always True.

    Attributes:
        provider: The LLM provider enum value (LLMProvider.PERPLEXITY).
        model_name: Default model to use for generation.
        temperature: Default temperature setting (0-2).
        top_p: Default top_p setting (0-1).
        api_key: Perplexity API key.
        verbose: Enable verbose logging.

    Example:
        >>> llm = PerplexityLLM(
        ...     provider=LLMProvider.PERPLEXITY,
        ...     model_name="sonar-pro",
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
        Initialize the Perplexity LLM wrapper.

        Args:
            provider: LLMProvider enum value.
            model_name: Default model name (e.g., 'sonar', 'sonar-pro').
            temperature: Default temperature (0-2).
            top_p: Default top_p value (0-1).
            api_key: Perplexity API key. Falls back to PERPLEXITY_API_KEY env var.
            verbose: Enable verbose logging.
        """
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY", "")

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
        web_search: bool = True,  # Always True for Perplexity, kept for API compatibility
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[Literal["day", "week", "month", "year"]] = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        timeout: Optional[float] = None,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ) -> Union[str, "LLMFullResponse"]:
        """
        Generate a response using a Perplexity model.

        This method supports all Perplexity models including reasoning models
        (sonar-reasoning-pro). Web search is ALWAYS performed - it's a core
        Perplexity feature, not optional.

        Args:
            model_name: Override the default model. Defaults to instance model.
            prompt: Simple text prompt. Cannot be used with messages.
            messages: List of message dicts for multi-turn conversations.
                Cannot be used with prompt.
            system_prompt: System message to set context. Default:
                "You are a helpful AI Assistant"
            temperature: Sampling temperature (0-2). Default: 0.7
            max_tokens: Maximum tokens to generate. Default: 300 (recommend
                4000+ for reasoning models).
            top_p: Nucleus sampling parameter. Default: 1.0
            full_response: Return LLMFullResponse with metadata including
                web_sources (citations). Default: False
            json_mode: Accepted for API compatibility. For JSON output,
                include "Return JSON" in your prompt. Default: False
            images: List of image URLs or file paths for vision models.
                Supports PNG, JPEG, GIF, WebP. Max 50MB per image.
            detail: Image detail level. Default: "auto"
                - "low": Fast processing
                - "high": Detailed analysis
                - "auto": Let API decide based on image size
            web_search: Kept for API compatibility - always True for Perplexity.
            search_domain_filter: List of domains to include/exclude.
                Prefix with "-" to exclude (e.g., ["-example.com"]).
                Max 20 domains.
            search_recency_filter: Filter search results by time.
                Options: "day", "week", "month", "year". Default: None
            return_images: Include images in search results. Default: False
            return_related_questions: Return related query suggestions. Default: False
            timeout: Request timeout in seconds. Default: None (no timeout)

        Returns:
            str: Generated text if full_response=False
            LLMFullResponse: Full response with metadata if full_response=True
                - generated_text: The model's response
                - model: Model name used
                - process_time: Time taken in seconds
                - input_token_count: Prompt tokens used
                - output_token_count: Completion tokens used
                - web_sources: List of citations from web search
                - finish_reason: Why generation stopped
                - is_reasoning_model: True if reasoning model was used

        Raises:
            ValueError: If both prompt and messages provided, or neither

        Example:
            >>> # Simple generation with built-in web search
            >>> response = llm.generate_response(prompt="Latest news on AI")

            >>> # With domain and recency filters
            >>> response = llm.generate_response(
            ...     prompt="Recent AI developments",
            ...     search_domain_filter=["arxiv.org", "nature.com"],
            ...     search_recency_filter="month",
            ...     full_response=True
            ... )
            >>> for source in response.web_sources or []:
            ...     print(f"Source: {source['url']}")

            >>> # With vision
            >>> response = llm.generate_response(
            ...     prompt="Describe this image",
            ...     images=["photo.jpg", "https://example.com/img.png"],
            ...     detail="high"
            ... )

            >>> # Reasoning model
            >>> response = llm.generate_response(
            ...     model_name="sonar-reasoning-pro",
            ...     prompt="Analyze the implications of...",
            ...     max_tokens=4000,
            ...     full_response=True
            ... )
        """
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
        else:
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
                "search_domain_filter": search_domain_filter,
                "search_recency_filter": search_recency_filter,
                "return_images": return_images,
                "return_related_questions": return_related_questions,
                "timeout": timeout,
            }
        )

        if self.verbose:
            verbose_print(f"Generating response with Perplexity using {params['model_name']}...", "info")
            if search_recency_filter:
                verbose_print(f"Search recency filter: {search_recency_filter}", "debug")
            if search_domain_filter:
                verbose_print(f"Search domain filter: {search_domain_filter}", "debug")

        try:
            response = perplexity_llm.generate_response(**params)
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
        web_search: bool = True,  # Always True for Perplexity, kept for API compatibility
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[Literal["day", "week", "month", "year"]] = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        timeout: Optional[float] = None,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ) -> Union[str, "LLMFullResponse"]:
        """
        Asynchronously generate a response using a Perplexity model.

        This is the async version of generate_response(). It supports all
        Perplexity models including reasoning models (sonar-reasoning-pro).
        Web search is ALWAYS performed - it's a core Perplexity feature.

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
            json_mode: Accepted for API compatibility. For JSON output,
                include "Return JSON" in your prompt. Default: False
            images: List of image URLs or file paths for vision models.
            detail: Image detail level ("low", "high", "auto"). Default: "auto"
            web_search: Kept for API compatibility - always True for Perplexity.
            search_domain_filter: List of domains to include/exclude.
            search_recency_filter: Filter by time ("day", "week", "month", "year").
            return_images: Include images in search results. Default: False
            return_related_questions: Return related query suggestions. Default: False
            timeout: Request timeout in seconds. Default: None

        Returns:
            str: Generated text if full_response=False
            LLMFullResponse: Full response with metadata if full_response=True

        Raises:
            ValueError: If both prompt and messages provided, or neither

        Example:
            >>> # Async usage
            >>> response = await llm.generate_response_async(
            ...     prompt="Latest tech news",
            ...     search_recency_filter="day",
            ...     full_response=True
            ... )
            >>> print(response.web_sources)

        See Also:
            generate_response: Synchronous version of this method.
        """
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
        else:
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
                "search_domain_filter": search_domain_filter,
                "search_recency_filter": search_recency_filter,
                "return_images": return_images,
                "return_related_questions": return_related_questions,
                "timeout": timeout,
            }
        )

        if self.verbose:
            verbose_print(f"Generating response with Perplexity (async) using {params['model_name']}...", "info")
            if search_recency_filter:
                verbose_print(f"Search recency filter: {search_recency_filter}", "debug")
            if search_domain_filter:
                verbose_print(f"Search domain filter: {search_domain_filter}", "debug")

        try:
            response = await perplexity_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
