"""
Gemini LLM Wrapper - High-level interface for Google Gemini models.

This wrapper provides a convenient, unified interface for interacting with
all Gemini language models, including support for:

- All Gemini models (1.5, 2.5, 3 series)
- Thinking models with configurable levels (Gemini 2.5/3)
- Vision capabilities (image analysis)
- Web search grounding with source attribution
- JSON mode for structured outputs
- Native system instructions
- Prompt caching

Example:
    >>> from SimplerLLM.language.llm import LLM, LLMProvider
    >>>
    >>> # Create instance
    >>> llm = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-2.5-flash")
    >>>
    >>> # Simple text generation
    >>> response = llm.generate_response(prompt="Explain quantum computing")
    >>>
    >>> # With vision
    >>> response = llm.generate_response(
    ...     prompt="What's in this image?",
    ...     images=["photo.jpg"]
    ... )
    >>>
    >>> # With thinking model
    >>> llm = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-3-pro-preview")
    >>> response = llm.generate_response(
    ...     prompt="Solve this complex problem...",
    ...     max_tokens=8000,
    ...     thinking_level="high",
    ...     full_response=True
    ... )
    >>> print(f"Used {response.reasoning_tokens} thinking tokens")
    >>> print(f"Reasoning: {response.thinking_content}")
    >>>
    >>> # With web search
    >>> response = llm.generate_response(
    ...     prompt="Latest AI news",
    ...     web_search=True,
    ...     full_response=True
    ... )
    >>> for source in response.web_sources or []:
    ...     print(f"- {source['title']}")
"""

import SimplerLLM.language.llm_providers.gemini_llm as gemini_llm
import os
import base64
import requests
import json
from typing import Optional, List, Literal, Union, Dict, Any

from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print
from SimplerLLM.tools.image_helpers import prepare_vision_content_gemini


class GeminiLLM(LLM):
    """
    Gemini LLM wrapper supporting all Google Gemini models.

    This class provides a high-level interface for Gemini's language models,
    handling parameter formatting, vision content preparation, thinking
    configuration, and web search grounding automatically.

    Attributes:
        api_key: The Gemini API key (from constructor or GEMINI_API_KEY env var).
        model_name: The model identifier.
        temperature: Default temperature for generation.
        top_p: Default top_p for generation.
        verbose: Whether to print verbose debug information.

    Example:
        >>> from SimplerLLM.language.llm import LLM, LLMProvider
        >>> llm = LLM.create(
        ...     provider=LLMProvider.GEMINI,
        ...     model_name="gemini-2.5-flash",
        ...     temperature=0.7
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
        Initialize the Gemini LLM wrapper.

        Args:
            provider: The LLMProvider enum value.
            model_name: The Gemini model to use.
            temperature: Default sampling temperature.
            top_p: Default nucleus sampling parameter.
            api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
            verbose: Enable verbose logging.
        """
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def convert_messages_template(self, messages: List[Dict]) -> List[Dict]:
        """
        Convert SimplerLLM message format to Gemini format.

        Transforms messages from the standard format with 'role' and 'content'
        to Gemini's format with 'role' and 'parts'.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            List of Gemini-formatted messages with 'role' and 'parts'.

        Example:
            >>> msgs = [{"role": "user", "content": "Hello"}]
            >>> gemini_msgs = llm.convert_messages_template(msgs)
            >>> gemini_msgs[0]
            {'role': 'user', 'parts': [{'text': 'Hello'}]}
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            # Map assistant -> model for Gemini
            if role in ("assistant", "model"):
                role = "model"
            else:
                role = "user"

            content = msg.get("content", "")
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                # Already formatted as parts
                parts = content
            else:
                parts = [{"text": str(content)}]

            converted.append({"role": role, "parts": parts})
        return converted

    def create_cache(
        self,
        cached_input: str,
        ttl: int = 600,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Create a cached content entry for prompt caching.

        Gemini's caching feature allows you to cache large inputs (like
        documents or context) and reuse them across multiple requests,
        reducing costs and latency.

        Args:
            cached_input: The content to cache (e.g., system prompt, document).
            ttl: Time-to-live in seconds (default: 600 = 10 minutes).
            model_name: Model to use for caching (default: instance's model).

        Returns:
            str: The cache ID to use in subsequent requests.

        Example:
            >>> cache_id = llm.create_cache(
            ...     cached_input="Large document content here...",
            ...     ttl=3600  # 1 hour
            ... )
            >>> response = llm.generate_response(
            ...     prompt="Summarize the document",
            ...     prompt_caching=True,
            ...     cache_id=cache_id
            ... )
        """
        # Use provided model or instance's model
        model = model_name or self.model_name

        encoded_content = base64.b64encode(cached_input.encode()).decode()
        cache_url = f"https://generativelanguage.googleapis.com/v1beta/cachedContents?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        cache_payload = {
            "model": f"models/{model}",
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "text/plain",
                                "data": encoded_content
                            }
                        }
                    ],
                    "role": "user"
                }
            ],
            "ttl": f"{ttl}s"
        }

        response = requests.post(cache_url, headers=headers, data=json.dumps(cache_payload))
        response.raise_for_status()

        return response.json()["name"]

    def generate_response(
        self,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: Optional[float] = None,
        max_tokens: int = 300,
        top_p: Optional[float] = None,
        full_response: bool = False,
        prompt_caching: bool = False,
        cache_id: Optional[str] = None,
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
        detail: str = "auto",  # Kept for API compatibility with OpenAI
        media_resolution: Optional[str] = None,
        web_search: bool = False,
        # Thinking parameters
        thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
        thinking_budget: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ) -> Union[str, "gemini_llm.LLMFullResponse"]:
        """
        Generate a response using a Gemini model.

        This method supports all Gemini models including thinking models
        (2.5/3 series) and automatically handles their special requirements.

        Args:
            model_name: Override the default model.
            prompt: Simple text prompt. Cannot be used with messages.
            messages: List of message dicts for multi-turn conversations.
            system_prompt: System instruction (uses native systemInstruction).
            temperature: Sampling temperature (0-2). Uses instance default if None.
            max_tokens: Maximum output tokens. Default: 300
            top_p: Nucleus sampling parameter. Uses instance default if None.
            full_response: Return LLMFullResponse with metadata. Default: False
            prompt_caching: Enable Gemini prompt caching. Default: False
            cache_id: Cached content ID for prompt caching.
            json_mode: Force JSON output format. Default: False
            response_schema: Optional JSON schema for structured output.
            images: List of image URLs or file paths for vision.
            detail: Kept for API compatibility (use media_resolution for Gemini 3).
            media_resolution: Image resolution setting for Gemini 3.
            web_search: Enable Google Search grounding. Default: False
            thinking_level: For Gemini 3: "minimal", "low", "medium", "high".
            thinking_budget: For Gemini 2.5: Token budget (0-32768, -1 dynamic).
            timeout: Request timeout in seconds. Default: None

        Returns:
            str: Generated text if full_response=False
            LLMFullResponse: Full response with metadata if full_response=True
                - generated_text: The generated response
                - reasoning_tokens: Thinking tokens used (thinking models)
                - thinking_content: The model's reasoning process (thinking models)
                - web_sources: List of cited sources (web_search=True)
                - finish_reason: Why generation stopped

        Raises:
            ValueError: If both prompt and messages provided, or neither.

        Example:
            >>> # Simple text
            >>> response = llm.generate_response(prompt="Hello!")

            >>> # With vision
            >>> response = llm.generate_response(
            ...     prompt="Describe this image",
            ...     images=["photo.jpg"]
            ... )

            >>> # With thinking (Gemini 3)
            >>> response = llm.generate_response(
            ...     model_name="gemini-3-pro-preview",
            ...     prompt="Solve this puzzle...",
            ...     thinking_level="high",
            ...     max_tokens=10000,
            ...     full_response=True
            ... )
            >>> print(f"Thinking: {response.thinking_content}")

            >>> # With web search
            >>> response = llm.generate_response(
            ...     prompt="Latest AI news",
            ...     web_search=True,
            ...     full_response=True
            ... )
            >>> for source in response.web_sources or []:
            ...     print(f"- {source['title']}: {source['url']}")

            >>> # JSON mode with schema
            >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            >>> response = llm.generate_response(
            ...     prompt="Return info about Python",
            ...     json_mode=True,
            ...     response_schema=schema
            ... )
        """
        # Use instance defaults if not provided
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        model_name = model_name or self.model_name

        # Validate inputs
        if prompt and messages:
            if self.verbose:
                verbose_print("Error: Both prompt and messages provided", "error")
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            if self.verbose:
                verbose_print("Error: Neither prompt nor messages provided", "error")
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages
        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
                if images:
                    verbose_print(f"Images provided: {len(images)} image(s)", "debug")

            # If images provided, prepare vision content
            if images:
                vision_parts = prepare_vision_content_gemini(prompt, images)
                model_messages = [{"role": "user", "content": vision_parts}]
            else:
                model_messages = [{"role": "user", "content": prompt}]
        else:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = messages

        if self.verbose:
            verbose_print("Generating response with Gemini...", "info")
            verbose_print(f"Model: {model_name}", "debug")
            if thinking_level:
                verbose_print(f"Thinking level: {thinking_level}", "debug")
            if thinking_budget is not None:
                verbose_print(f"Thinking budget: {thinking_budget}", "debug")
            if web_search:
                verbose_print("Web search grounding enabled", "debug")
            if json_mode:
                verbose_print("JSON mode enabled", "debug")

        try:
            response = gemini_llm.generate_response(
                model_name=model_name,
                system_prompt=system_prompt,
                messages=model_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                full_response=full_response,
                prompt_caching=prompt_caching,
                cache_id=cache_id,
                api_key=self.api_key,
                json_mode=json_mode,
                response_schema=response_schema,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                web_search=web_search,
                timeout=timeout,
            )

            if self.verbose:
                verbose_print("Response received successfully", "info")
                if full_response and hasattr(response, 'reasoning_tokens') and response.reasoning_tokens:
                    verbose_print(f"Thinking tokens used: {response.reasoning_tokens}", "debug")
                if full_response and hasattr(response, 'web_sources') and response.web_sources:
                    verbose_print(f"Web sources: {len(response.web_sources)}", "debug")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise

    async def generate_response_async(
        self,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: Optional[float] = None,
        max_tokens: int = 300,
        top_p: Optional[float] = None,
        full_response: bool = False,
        prompt_caching: bool = False,
        cache_id: Optional[str] = None,
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
        detail: str = "auto",
        media_resolution: Optional[str] = None,
        web_search: bool = False,
        thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
        thinking_budget: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,  # Accept and ignore unsupported provider-specific params
    ) -> Union[str, "gemini_llm.LLMFullResponse"]:
        """
        Asynchronously generate a response using a Gemini model.

        See generate_response() for full parameter documentation.

        Example:
            >>> import asyncio
            >>> async def main():
            ...     response = await llm.generate_response_async(
            ...         prompt="Hello!",
            ...         max_tokens=100
            ...     )
            ...     print(response)
            >>> asyncio.run(main())
        """
        # Use instance defaults if not provided
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        model_name = model_name or self.model_name

        # Validate inputs
        if prompt and messages:
            if self.verbose:
                verbose_print("Error: Both prompt and messages provided", "error")
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            if self.verbose:
                verbose_print("Error: Neither prompt nor messages provided", "error")
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages
        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message (async)", "debug")

            if images:
                vision_parts = prepare_vision_content_gemini(prompt, images)
                model_messages = [{"role": "user", "content": vision_parts}]
            else:
                model_messages = [{"role": "user", "content": prompt}]
        else:
            if self.verbose:
                verbose_print("Preparing chat messages (async)", "debug")
            model_messages = messages

        if self.verbose:
            verbose_print("Generating response with Gemini (async)...", "info")

        try:
            response = await gemini_llm.generate_response_async(
                model_name=model_name,
                system_prompt=system_prompt,
                messages=model_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                full_response=full_response,
                prompt_caching=prompt_caching,
                cache_id=cache_id,
                api_key=self.api_key,
                json_mode=json_mode,
                response_schema=response_schema,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                web_search=web_search,
                timeout=timeout,
            )

            if self.verbose:
                verbose_print("Response received successfully (async)", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response (async): {str(e)}", "error")
            raise
