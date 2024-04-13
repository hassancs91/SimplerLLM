import SimplerLLM.language.llm_providers.openai_llm as openai_llm
import SimplerLLM.language.llm_providers.gemini_llm as gemini_llm
import SimplerLLM.language.llm_providers.anthropic_llm as anthropic_llm
from enum import Enum


class LLMProvider(Enum):
    OPENAI = 1
    GEMINI = 2
    ANTHROPIC = 3


class LLM:
    def __init__(
        self,
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        top_p=1.0,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    @staticmethod
    def create(
        provider=None,
        model_name=None,
        temperature=0.7,
        top_p=1.0,
    ):
        if provider == LLMProvider.OPENAI:
            return OpenAILLM(provider, model_name, temperature, top_p)
        if provider == LLMProvider.GEMINI:
            return GeminiLLM(provider, model_name, temperature, top_p)
        if provider == LLMProvider.ANTHROPIC:
            return AnthropicLLM(provider, model_name, temperature, top_p)
        else:
            return None

    def set_model(self, provider):
        if not isinstance(provider, LLMProvider):
            raise ValueError("Provider must be an instance of LLMProvider Enum")
        self.provider = provider

    def prepare_params(self, model_name, temperature, top_p):
        # Use instance values as defaults if parameters are not provided
        return {
            "model_name": model_name if model_name else self.model_name,
            "temperature": temperature if temperature else self.temperature,
            "top_p": top_p if top_p else self.top_p,
        }


class OpenAILLM(LLM):
    def __init__(self, model, model_name, temperature, top_p):
        super().__init__(model, model_name, temperature, top_p)

    def generate_response(
        self,
        model_name=None,
        prompt=None,
        messages=None,
        system_prompt="You are a helpful AI Assistant",
        temperature=0.7,
        max_tokens=300,
        top_p=1.0,
        full_response=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)
        params.update(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return openai_llm.generate_response(**params)

    async def generate_response_async(
        self,
        model_name=None,
        prompt=None,
        messages=None,
        system_prompt="You are a helpful AI Assistant",
        temperature=0.7,
        max_tokens=300,
        top_p=1.0,
        full_response=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)
        params.update(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return await openai_llm.generate_response_async(**params)


class GeminiLLM(LLM):
    def __init__(self, model, model_name, temperature, top_p):
        super().__init__(model, model_name, temperature, top_p)

    def generate_response(
        self,
        model_name=None,
        prompt=None,
        messages=None,
        system_prompt="You are a helpful AI Assistant",
        temperature=0.7,
        max_tokens=300,
        top_p=1.0,
        full_response=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)
        params.update(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return gemini_llm.generate_response(**params)

    async def generate_response_async(
        self,
        model_name=None,
        prompt=None,
        messages=None,
        system_prompt="You are a helpful AI Assistant",
        temperature=0.7,
        max_tokens=300,
        top_p=1.0,
        full_response=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)
        params.update(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return await gemini_llm.generate_response_async(**params)


class AnthropicLLM(LLM):
    def __init__(self, model, model_name, temperature, top_p):
        super().__init__(model, model_name, temperature, top_p)

    def generate_response(
        self,
        model_name=None,
        prompt=None,
        messages=None,
        system_prompt="You are a helpful AI Assistant",
        temperature=0.7,
        max_tokens=300,
        top_p=1.0,
        full_response=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)
        params.update(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return anthropic_llm.generate_response(**params)

    async def generate_response_async(
        self,
        model_name=None,
        prompt=None,
        messages=None,
        system_prompt="You are a helpful AI Assistant",
        temperature=0.7,
        max_tokens=300,
        top_p=1.0,
        full_response=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)
        params.update(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return await anthropic_llm.generate_response_async(**params)
