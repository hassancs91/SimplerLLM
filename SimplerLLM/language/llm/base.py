from enum import Enum
import os
import time
import asyncio
from typing import Type
from pydantic import BaseModel

from SimplerLLM.utils.custom_verbose import verbose_print
from SimplerLLM.tools.json_helpers import (
    extract_json_from_text,
    convert_json_to_pydantic_model,
    validate_json_with_pydantic_model,
    generate_json_example_from_pydantic,
)

class LLMProvider(Enum):
    OPENAI = 1
    GEMINI = 2
    ANTHROPIC = 3
    OLLAMA = 4
    LWH = 5
    DEEPSEEK = 6
    OPENROUTER = 7
    COHERE = 8

class LLM:
    def __init__(
        self,
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        temperature=0.7,
        top_p=1.0,
        api_key=None,
        user_id=None,
        verbose=False,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key
        self.user_id = user_id
        self.verbose = verbose
        
        if self.verbose:
            verbose_print(f"Initializing {provider.name} LLM with model: {model_name}", "info")
            verbose_print(f"Configuration - Temperature: {temperature}, Top_p: {top_p}", "debug")

    @staticmethod
    def create(
        provider=None,
        model_name=None,
        temperature=0.7,
        top_p=1.0,
        api_key=None,
        user_id=None,
        verbose=False,
    ):
        if provider == LLMProvider.OPENAI:
            from .wrappers.openai_wrapper import OpenAILLM
            return OpenAILLM(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        if provider == LLMProvider.GEMINI:
            from .wrappers.gemini_wrapper import GeminiLLM
            return GeminiLLM(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        if provider == LLMProvider.ANTHROPIC:
            from .wrappers.anthropic_wrapper import AnthropicLLM
            return AnthropicLLM(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        if provider == LLMProvider.OLLAMA:
            from .wrappers.ollama_wrapper import OllamaLLM
            return OllamaLLM(provider, model_name, temperature, top_p, verbose=verbose)
        if provider == LLMProvider.DEEPSEEK:
            from .wrappers.deepseek_wrapper import DeepSeekLLM
            return DeepSeekLLM(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        if provider == LLMProvider.OPENROUTER:
            from .wrappers.openrouter_wrapper import OpenRouterLLM
            return OpenRouterLLM(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        if provider == LLMProvider.COHERE:
            from .wrappers.cohere_wrapper import CohereLLM
            return CohereLLM(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        else:
            return None

    def set_model(self, provider):
        if not isinstance(provider, LLMProvider):
            raise ValueError("Provider must be an instance of LLMProvider Enum")
        self.provider = provider

    def prepare_params(self, model_name, temperature, top_p):
        return {
            "model_name": model_name if model_name else self.model_name,
            "temperature": temperature if temperature else self.temperature,
            "top_p": top_p if top_p else self.top_p,
        }
