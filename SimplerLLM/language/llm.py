import SimplerLLM.language.llm_providers.openai_llm as openai_llm
import SimplerLLM.language.llm_providers.gemini_llm as gemini_llm
import SimplerLLM.language.llm_providers.anthropic_llm as anthropic_llm
import SimplerLLM.language.llm_providers.ollama_llm as ollama_llm
import SimplerLLM.language.llm_providers.lwh_llm as lwh_llm
from SimplerLLM.prompts.messages_template import MessagesTemplate
from enum import Enum
import os


class LLMProvider(Enum):
    OPENAI = 1
    GEMINI = 2
    ANTHROPIC = 3
    OLLAMA = 4
    LWH = 5


class LLM:
    def __init__(
        self,
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        top_p=1.0,
        api_key=None,
        user_id = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key
        self.user_id = user_id

    @staticmethod
    def create(
        provider=None,
        model_name=None,
        temperature=0.7,
        top_p=1.0,
        api_key=None,
        user_id = None,
        
    ):
        if provider == LLMProvider.OPENAI:
            return OpenAILLM(provider, model_name, temperature, top_p, api_key)
        if provider == LLMProvider.GEMINI:
            return GeminiLLM(provider, model_name, temperature, top_p, api_key)
        if provider == LLMProvider.ANTHROPIC:
            return AnthropicLLM(provider, model_name, temperature, top_p, api_key)
        if provider == LLMProvider.OLLAMA:
            return OllamaLLM(provider, model_name, temperature, top_p, api_key)
        if provider == LLMProvider.LWH:
            return LwhLLM(provider, model_name, temperature, top_p, api_key, user_id)
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
    def __init__(self, provider, model_name, temperature, top_p, api_key):
        super().__init__(provider, model_name, temperature, top_p, api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    
    def append_messages(self, system_prompt : str, messages: list):
        model_messages = [{"role": "system", "content": system_prompt}]
        if messages:
            model_messages.extend(messages)
        return model_messages

   

    def generate_response(
        self,
        model_name: str =None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str="You are a helpful AI Assistant",
        temperature: float=0.7,
        max_tokens: int=300,
        top_p: float=1.0,
        full_response: bool=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        if messages:
            model_messages = self.append_messages(system_prompt, messages)



        params.update(
            {
                "api_key": self.api_key,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return openai_llm.generate_response(**params)

    async def generate_response_async(
        self,
        model_name: str =None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str="You are a helpful AI Assistant",
        temperature: float=0.7,
        max_tokens: int=300,
        top_p: float=1.0,
        full_response: bool=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        if messages:
            model_messages = self.append_messages(system_prompt, messages)


        params.update(
            {
                "api_key": self.api_key,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return await openai_llm.generate_response_async(**params)

class GeminiLLM(LLM):

    def __init__(self, provider, model_name, temperature, top_p, api_key):
        super().__init__(provider, model_name, temperature, top_p, api_key)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def convert_messages_template(self, messages):
        # Convert the unified message template to Gemini's format
        #return [{"role": msg["role"], "parts": [msg["content"]]} for msg in messages]
        return [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages]

    def append_messages(self, system_prompt, messages):
        model_messages = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "ok, confirmed."}]},
            ]
        messages = self.convert_messages_template(messages)
        model_messages.extend(messages)
        return model_messages



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
        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        if prompt:
            model_messages = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "ok, confirmed."}]},
                {"role": "user", "parts": [{"text": prompt}]}
            ]

        if messages:
            model_messages = self.append_messages(system_prompt, messages)

        params.update(
            {
                "api_key": self.api_key,
                "messages": model_messages,
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
        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        if prompt:
            model_messages = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "ok, confirmed."}]},
                {"role": "user", "parts": [{"text": prompt}]}
            ]

        if messages:
            model_messages = self.append_messages(system_prompt, messages)
        params.update(
            {
                "api_key": self.api_key,
                "messages": model_messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return await gemini_llm.generate_response_async(**params)

class AnthropicLLM(LLM):
    def __init__(self, provider, model_name, temperature, top_p, api_key):
        super().__init__(provider, model_name, temperature, top_p, api_key)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    
    def append_messages(self, messages: list):
        model_messages = []
        if messages:
            model_messages.extend(messages)
        return model_messages

    def generate_response(
        self,
        model_name: str =None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str="You are a helpful AI Assistant",
        temperature: float=0.7,
        max_tokens: int=300,
        top_p: float=1.0,
        full_response: bool=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            model_messages = [
                {"role": "user", "content": prompt},
            ]

        if messages:
            model_messages = self.append_messages(messages)

        params.update(
            {
                "api_key": self.api_key,
                "system_prompt": system_prompt,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return anthropic_llm.generate_response(**params)

    async def generate_response_async(
        self,
        model_name=None,
        prompt=None,
        messages: list = None,
        system_prompt="You are a helpful AI Assistant",
        temperature=0.7,
        max_tokens=300,
        top_p=1.0,
        full_response=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            model_messages = [
                {"role": "user", "content": prompt},
            ]

        if messages:
            model_messages = self.append_messages(messages)

        params.update(
            {
                "api_key": self.api_key,
                "system_prompt": system_prompt,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return await anthropic_llm.generate_response_async(**params)
    
class OllamaLLM(LLM):
    def __init__(self, model, model_name, temperature, top_p):
        super().__init__(model, model_name, temperature, top_p)

    
    def append_messages(self, system_prompt : str, messages: list):
        model_messages = [{"role": "system", "content": system_prompt}]
        if messages:
            model_messages.extend(messages)
        return model_messages

   

    def generate_response(
        self,
        model_name: str =None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str="You are a helpful AI Assistant",
        temperature: float=0.7,
        max_tokens: int=300,
        top_p: float=1.0,
        full_response: bool=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        if messages:
            model_messages = self.append_messages(system_prompt, messages)



        params.update(
            {
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return ollama_llm.generate_response(**params)

    async def generate_response_async(
        self,
        model_name: str =None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str="You are a helpful AI Assistant",
        temperature: float=0.7,
        max_tokens: int=300,
        top_p: float=1.0,
        full_response: bool=False,
    ):
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        if messages:
            model_messages = self.append_messages(system_prompt, messages)


        params.update(
            {
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
            }
        )
        return await ollama_llm.generate_response_async(**params)

class LwhLLM(LLM):
    def __init__(self, provider, model_name, temperature, top_p, api_key,user_id):
        super().__init__(provider, model_name, temperature, top_p, api_key,user_id)
        self.api_key = api_key or os.getenv("LWH_API_KEY", "")
        self.user_id = user_id or os.getenv("LWH_USER_ID", "0")

    
    def append_messages(self, messages: list):
        model_messages = []
        if messages:
            model_messages.extend(messages)
        return model_messages

   

    def generate_response(
        self,
        model_name: str = None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str="You are a helpful AI Assistant",
        temperature: float=0.7,
        max_tokens: int=300,
        top_p: float=1.0,
        full_response: bool=False,
    ):
        
    
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            model_messages = [
                {"role": "user", "content": prompt},
            ]

        if messages:
            model_messages = self.append_messages(messages)



        params.update(
            {
                "api_key": self.api_key,
                "user_id": self.user_id,
                "messages": model_messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "temperature" : temperature,
                "top_p" : top_p,
                "max_tokens" : max_tokens
            }
        )
        return lwh_llm.generate_response(**params)



    async def generate_response_async(
        self,
        model_name: str =None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str="You are a helpful AI Assistant",
        temperature: float=0.7,
        max_tokens: int=300,
        top_p: float=1.0,
        full_response: bool=False,
    ):
        
       

        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        if messages:
            model_messages = self.append_messages(system_prompt, messages)


        params.update(
            {
                "api_key": self.api_key,
                "user_id": self.user_id,
                "messages": model_messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "temperature" : temperature,
                "top_p" : top_p,
                "max_tokens" : max_tokens
            }
        )
        return await lwh_llm.generate_response_async(**params)