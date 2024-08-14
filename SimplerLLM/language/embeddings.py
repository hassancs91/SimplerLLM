import SimplerLLM.language.llm_providers.openai_llm as openai_llm
from enum import Enum
import os

class EmbeddingsProvider(Enum):
    OPENAI = 1


class EmbeddingsLLM:
    def __init__(
        self, 
        provider=EmbeddingsProvider.OPENAI,
        model_name="text-embedding-3-small",
        api_key=None,
        user_id = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.user_id = user_id
        

    @staticmethod
    def create(
        provider=None,
        model_name=None,
        api_key=None,
        user_id = None,
    ):
        if provider == EmbeddingsProvider.OPENAI:
            return OpenAILLM(provider, model_name, api_key)

        else:
            return None

    def set_model(self, provider):
        if not isinstance(provider, EmbeddingsProvider):
            raise ValueError("Provider must be an instance of EmbeddingsProvider Enum")
        self.provider = provider


class OpenAILLM(EmbeddingsLLM):
    def __init__(self, model, model_name,api_key):
        super().__init__(model, model_name,api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")


    def generate_embeddings(
        self,
        user_input,
        model_name=None,
        full_response=False,
    ):
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name
        
        return openai_llm.generate_embeddings(
            user_input=user_input,
            model_name=model_name,
            full_response=full_response,
        )
    
    async def generate_embeddings_async(
        self,
        user_input,
        model_name=None,
        full_response=False,
    ):
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name

        return await openai_llm.generate_embeddings_async(
            user_input=user_input,
            model_name=model_name,
            full_response=full_response,
        )


