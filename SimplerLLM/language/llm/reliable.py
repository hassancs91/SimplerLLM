from typing import Union, Tuple, Optional, Any, Dict, List, Set, Literal
from .base import LLM, LLMProvider
from SimplerLLM.utils.custom_verbose import verbose_print
from SimplerLLM.language.llm_providers.llm_response_models import LLMFullResponse

# Provider-specific parameter capabilities mapping
PROVIDER_CAPABILITIES: Dict[LLMProvider, Set[str]] = {
    LLMProvider.OPENAI: {
        "reasoning_effort", "timeout", "web_search", "images", "detail"
    },
    LLMProvider.ANTHROPIC: {
        "thinking_budget", "prompt_caching", "cached_input", "web_search", "images"
    },
    LLMProvider.GEMINI: {
        "thinking_budget", "thinking_level", "prompt_caching", "cache_id",
        "response_schema", "media_resolution", "timeout", "web_search", "images"
    },
    LLMProvider.DEEPSEEK: {"thinking", "images"},
    LLMProvider.PERPLEXITY: {
        "timeout", "search_domain_filter", "search_recency_filter",
        "return_images", "return_related_questions", "images"
    },
    LLMProvider.OPENROUTER: {
        "reasoning_effort", "timeout", "site_url", "site_name", "images", "detail"
    },
    LLMProvider.OLLAMA: {"images"},
    LLMProvider.COHERE: set(),
    LLMProvider.HUGGING_FACE_LOCAL: set(),
}

# Universal parameters supported by all providers
UNIVERSAL_PARAMS: Set[str] = {
    "model_name", "prompt", "messages", "system_prompt", "temperature",
    "max_tokens", "top_p", "full_response", "json_mode"
}

class ReliableLLM:
    def __init__(
        self,
        primary_llm: LLM,
        secondary_llm: LLM,
        verbose: bool = False,
        validation_max_tokens: int = 4000,
        skip_validation: bool = False,
        lazy_validation: bool = False,
    ):
        """
        Initialize ReliableLLM with primary and secondary LLM providers.

        Args:
            primary_llm (LLM): The primary LLM provider to use first
            secondary_llm (LLM): The secondary LLM provider to use as fallback
            verbose (bool): Enable verbose logging
            validation_max_tokens (int): Max tokens for provider validation test (default: 4000)
            skip_validation (bool): If True, skip provider validation entirely (default: False)
            lazy_validation (bool): If True, validate on first request instead of initialization (default: False)
        """
        self.primary_llm = primary_llm
        self.secondary_llm = secondary_llm
        self.verbose = verbose
        self.validation_max_tokens = validation_max_tokens
        self.skip_validation = skip_validation
        self.lazy_validation = lazy_validation
        self._validation_done = False

        # Initialize validity flags
        self.primary_valid = True
        self.secondary_valid = True

        if self.verbose:
            verbose_print("Initializing ReliableLLM with fallback support", "info")
            verbose_print(f"Primary provider: {primary_llm.provider.name}", "debug")
            verbose_print(f"Secondary provider: {secondary_llm.provider.name}", "debug")

        # Perform validation based on settings
        if not skip_validation and not lazy_validation:
            self._validate_providers()
            self._validation_done = True
        elif skip_validation and self.verbose:
            verbose_print("Provider validation skipped", "info")

    def _ensure_validated(self):
        """Perform lazy validation on first request if lazy_validation is enabled."""
        if self.skip_validation:
            return
        if not self._validation_done:
            self._validate_providers()
            self._validation_done = True

    def _validate_providers(self):
        """
        Validate both providers during initialization.
        Sets internal flags for which providers are valid.
        """
        # Reset validity flags
        self.primary_valid = True
        self.secondary_valid = True

        # Test primary provider
        try:
            if self.verbose:
                verbose_print("Validating primary provider...", "info")
            response = self.primary_llm.generate_response(
                prompt="test",
                max_tokens=self.validation_max_tokens
            )
            if response is None:
                self.primary_valid = False
                if self.verbose:
                    verbose_print("Primary provider returned None response", "warning")
        except Exception as e:
            self.primary_valid = False
            if self.verbose:
                verbose_print(f"Primary provider validation failed: {str(e)}", "error")

        # Test secondary provider
        try:
            if self.verbose:
                verbose_print("Validating secondary provider...", "info")
            response = self.secondary_llm.generate_response(
                prompt="test",
                max_tokens=self.validation_max_tokens
            )
            if response is None:
                self.secondary_valid = False
                if self.verbose:
                    verbose_print("Secondary provider returned None response", "warning")
        except Exception as e:
            self.secondary_valid = False
            if self.verbose:
                verbose_print(f"Secondary provider validation failed: {str(e)}", "error")

        if not self.primary_valid and not self.secondary_valid:
            if self.verbose:
                verbose_print("Critical: Both providers have invalid configurations", "critical")
            raise ValueError("Both providers have invalid configurations")
        
        if self.verbose:
            if self.primary_valid and self.secondary_valid:
                verbose_print("Both providers validated successfully", "info")
            elif self.primary_valid:
                verbose_print("Only primary provider validated successfully", "warning")
            else:
                verbose_print("Only secondary provider validated successfully", "warning")

    def _filter_params_for_provider(
        self,
        llm: LLM,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter parameters to only include those supported by the provider.
        Logs warnings for ignored parameters when verbose is enabled.

        Args:
            llm: The LLM instance to filter parameters for
            params: Dictionary of all parameters to filter

        Returns:
            Filtered dictionary containing only supported parameters
        """
        provider = llm.provider
        supported = PROVIDER_CAPABILITIES.get(provider, set())

        filtered = {}
        for key, value in params.items():
            if value is None:
                continue  # Skip None values

            # Universal parameters always pass through
            if key in UNIVERSAL_PARAMS:
                filtered[key] = value
            elif key in supported:
                filtered[key] = value
            else:
                # Only warn for non-None values that are being ignored
                if self.verbose and value not in (False, "", [], {}):
                    verbose_print(
                        f"Parameter '{key}' not supported by {provider.name}, ignoring",
                        "warning"
                    )

        return filtered

    def generate_response(
        self,
        # Universal parameters
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        return_provider: bool = False,
        json_mode: bool = False,
        images: Optional[List[str]] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        web_search: bool = False,
        # Cross-provider parameters (reasoning/thinking)
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        timeout: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
        thinking: Optional[bool] = None,
        # Caching parameters
        prompt_caching: bool = False,
        cached_input: str = "",
        cache_id: Optional[str] = None,
        # Gemini-specific
        response_schema: Optional[Dict[str, Any]] = None,
        media_resolution: Optional[str] = None,
        # Perplexity-specific
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[Literal["day", "week", "month", "year"]] = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        # OpenRouter-specific
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ) -> Union[str, LLMFullResponse, Tuple[Union[str, LLMFullResponse], LLMProvider, str]]:
        """
        Generate a response using the primary LLM, falling back to secondary if primary fails.

        Args:
            model_name (str, optional): The name of the model to use.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt to set the context.
            temperature (float, optional): Controls randomness in output.
            max_tokens (int, optional): The maximum number of tokens to generate.
            top_p (float, optional): Controls diversity of output.
            full_response (bool, optional): If True, returns the full API response.
            return_provider (bool, optional): If True, returns a tuple of (response, provider, model_name).
            json_mode (bool, optional): If True, enables JSON mode for structured output.
            images (list, optional): A list of image URLs or file paths for vision tasks.
            detail (str, optional): Level of detail for image analysis ("low", "high", "auto"). Defaults to "auto".
            web_search (bool, optional): If True, enables web search before generating response.
            reasoning_effort (str, optional): Reasoning effort level for OpenAI/OpenRouter ("low", "medium", "high").
            timeout (float, optional): Request timeout in seconds.
            thinking_budget (int, optional): Token budget for extended thinking (Anthropic/Gemini).
            thinking_level (str, optional): Thinking level for Gemini ("minimal", "low", "medium", "high").
            thinking (bool, optional): Enable chain-of-thought for DeepSeek.
            prompt_caching (bool, optional): Enable prompt caching (Anthropic/Gemini).
            cached_input (str, optional): Cached input content for Anthropic.
            cache_id (str, optional): Cache ID for Gemini.
            response_schema (dict, optional): Response schema for structured output (Gemini).
            media_resolution (str, optional): Media resolution for Gemini.
            search_domain_filter (list, optional): Domain filter for Perplexity web search.
            search_recency_filter (str, optional): Recency filter for Perplexity ("day", "week", "month", "year").
            return_images (bool, optional): Return images in Perplexity results.
            return_related_questions (bool, optional): Return related questions in Perplexity results.
            site_url (str, optional): Site URL for OpenRouter tracking.
            site_name (str, optional): Site name for OpenRouter tracking.

        Returns:
            Union[str, LLMFullResponse, Tuple[Union[str, LLMFullResponse], LLMProvider, str]]:
                - If return_provider is False: The generated response from either primary or secondary LLM
                - If return_provider is True: A tuple of (response, provider, model_name)

        Raises:
            ValueError: If both primary and secondary LLMs fail
        """
        # Ensure validation has been performed (for lazy validation)
        self._ensure_validated()

        # Build complete params dict
        all_params = {
            "model_name": model_name,
            "prompt": prompt,
            "messages": messages,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "full_response": full_response,
            "json_mode": json_mode,
            "images": images,
            "detail": detail,
            "web_search": web_search,
            "reasoning_effort": reasoning_effort,
            "timeout": timeout,
            "thinking_budget": thinking_budget,
            "thinking_level": thinking_level,
            "thinking": thinking,
            "prompt_caching": prompt_caching,
            "cached_input": cached_input,
            "cache_id": cache_id,
            "response_schema": response_schema,
            "media_resolution": media_resolution,
            "search_domain_filter": search_domain_filter,
            "search_recency_filter": search_recency_filter,
            "return_images": return_images,
            "return_related_questions": return_related_questions,
            "site_url": site_url,
            "site_name": site_name,
        }

        if self.primary_valid:
            if self.verbose:
                verbose_print("Attempting to generate response with primary provider...", "info")
            try:
                # Filter params for primary provider
                primary_params = self._filter_params_for_provider(self.primary_llm, all_params)
                response = self.primary_llm.generate_response(**primary_params)
                if self.verbose:
                    verbose_print("Primary provider generated response successfully", "info")
                return (response, self.primary_llm.provider, self.primary_llm.model_name) if return_provider else response
            except Exception as e:
                if self.verbose:
                    verbose_print(f"Primary provider failed: {str(e)}", "warning")
                    verbose_print("Falling back to secondary provider...", "info")
                if self.secondary_valid:
                    # Filter params for secondary provider
                    secondary_params = self._filter_params_for_provider(self.secondary_llm, all_params)
                    response = self.secondary_llm.generate_response(**secondary_params)
                    if self.verbose:
                        verbose_print("Secondary provider generated response successfully", "info")
                    return (response, self.secondary_llm.provider, self.secondary_llm.model_name) if return_provider else response
                if self.verbose:
                    verbose_print("Critical: Both providers failed to generate response", "critical")
                raise ValueError("Both providers failed")
        elif self.secondary_valid:
            if self.verbose:
                verbose_print("Primary provider invalid, using secondary provider...", "warning")
            # Filter params for secondary provider
            secondary_params = self._filter_params_for_provider(self.secondary_llm, all_params)
            response = self.secondary_llm.generate_response(**secondary_params)
            if self.verbose:
                verbose_print("Secondary provider generated response successfully", "info")
            return (response, self.secondary_llm.provider, self.secondary_llm.model_name) if return_provider else response
        if self.verbose:
            verbose_print("Critical: No valid providers available", "critical")
        raise ValueError("No valid providers available")

    async def generate_response_async(
        self,
        # Universal parameters
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        return_provider: bool = False,
        json_mode: bool = False,
        images: Optional[List[str]] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        web_search: bool = False,
        # Cross-provider parameters (reasoning/thinking)
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        timeout: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = None,
        thinking: Optional[bool] = None,
        # Caching parameters
        prompt_caching: bool = False,
        cached_input: str = "",
        cache_id: Optional[str] = None,
        # Gemini-specific
        response_schema: Optional[Dict[str, Any]] = None,
        media_resolution: Optional[str] = None,
        # Perplexity-specific
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[Literal["day", "week", "month", "year"]] = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        # OpenRouter-specific
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ) -> Union[str, LLMFullResponse, Tuple[Union[str, LLMFullResponse], LLMProvider, str]]:
        """
        Asynchronously generate a response using the primary LLM, falling back to secondary if primary fails.

        Args:
            model_name (str, optional): The name of the model to use.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt to set the context.
            temperature (float, optional): Controls randomness in output.
            max_tokens (int, optional): The maximum number of tokens to generate.
            top_p (float, optional): Controls diversity of output.
            full_response (bool, optional): If True, returns the full API response.
            return_provider (bool, optional): If True, returns a tuple of (response, provider, model_name).
            json_mode (bool, optional): If True, enables JSON mode for structured output.
            images (list, optional): A list of image URLs or file paths for vision tasks.
            detail (str, optional): Level of detail for image analysis ("low", "high", "auto"). Defaults to "auto".
            web_search (bool, optional): If True, enables web search before generating response.
            reasoning_effort (str, optional): Reasoning effort level for OpenAI/OpenRouter ("low", "medium", "high").
            timeout (float, optional): Request timeout in seconds.
            thinking_budget (int, optional): Token budget for extended thinking (Anthropic/Gemini).
            thinking_level (str, optional): Thinking level for Gemini ("minimal", "low", "medium", "high").
            thinking (bool, optional): Enable chain-of-thought for DeepSeek.
            prompt_caching (bool, optional): Enable prompt caching (Anthropic/Gemini).
            cached_input (str, optional): Cached input content for Anthropic.
            cache_id (str, optional): Cache ID for Gemini.
            response_schema (dict, optional): Response schema for structured output (Gemini).
            media_resolution (str, optional): Media resolution for Gemini.
            search_domain_filter (list, optional): Domain filter for Perplexity web search.
            search_recency_filter (str, optional): Recency filter for Perplexity ("day", "week", "month", "year").
            return_images (bool, optional): Return images in Perplexity results.
            return_related_questions (bool, optional): Return related questions in Perplexity results.
            site_url (str, optional): Site URL for OpenRouter tracking.
            site_name (str, optional): Site name for OpenRouter tracking.

        Returns:
            Union[str, LLMFullResponse, Tuple[Union[str, LLMFullResponse], LLMProvider, str]]:
                - If return_provider is False: The generated response from either primary or secondary LLM
                - If return_provider is True: A tuple of (response, provider, model_name)

        Raises:
            ValueError: If both primary and secondary LLMs fail
        """
        # Ensure validation has been performed (for lazy validation)
        self._ensure_validated()

        # Build complete params dict
        all_params = {
            "model_name": model_name,
            "prompt": prompt,
            "messages": messages,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "full_response": full_response,
            "json_mode": json_mode,
            "images": images,
            "detail": detail,
            "web_search": web_search,
            "reasoning_effort": reasoning_effort,
            "timeout": timeout,
            "thinking_budget": thinking_budget,
            "thinking_level": thinking_level,
            "thinking": thinking,
            "prompt_caching": prompt_caching,
            "cached_input": cached_input,
            "cache_id": cache_id,
            "response_schema": response_schema,
            "media_resolution": media_resolution,
            "search_domain_filter": search_domain_filter,
            "search_recency_filter": search_recency_filter,
            "return_images": return_images,
            "return_related_questions": return_related_questions,
            "site_url": site_url,
            "site_name": site_name,
        }

        if self.primary_valid:
            if self.verbose:
                verbose_print("Attempting to generate response with primary provider (async)...", "info")
            try:
                # Filter params for primary provider
                primary_params = self._filter_params_for_provider(self.primary_llm, all_params)
                response = await self.primary_llm.generate_response_async(**primary_params)
                if self.verbose:
                    verbose_print("Primary provider generated response successfully", "info")
                return (response, self.primary_llm.provider, self.primary_llm.model_name) if return_provider else response
            except Exception as e:
                if self.verbose:
                    verbose_print(f"Primary provider failed: {str(e)}", "warning")
                    verbose_print("Falling back to secondary provider...", "info")
                if self.secondary_valid:
                    # Filter params for secondary provider
                    secondary_params = self._filter_params_for_provider(self.secondary_llm, all_params)
                    response = await self.secondary_llm.generate_response_async(**secondary_params)
                    if self.verbose:
                        verbose_print("Secondary provider generated response successfully", "info")
                    return (response, self.secondary_llm.provider, self.secondary_llm.model_name) if return_provider else response
                if self.verbose:
                    verbose_print("Critical: Both providers failed to generate response", "critical")
                raise ValueError("Both providers failed")
        elif self.secondary_valid:
            if self.verbose:
                verbose_print("Primary provider invalid, using secondary provider (async)...", "warning")
            # Filter params for secondary provider
            secondary_params = self._filter_params_for_provider(self.secondary_llm, all_params)
            response = await self.secondary_llm.generate_response_async(**secondary_params)
            if self.verbose:
                verbose_print("Secondary provider generated response successfully", "info")
            return (response, self.secondary_llm.provider, self.secondary_llm.model_name) if return_provider else response
        if self.verbose:
            verbose_print("Critical: No valid providers available", "critical")
        raise ValueError("No valid providers available")
