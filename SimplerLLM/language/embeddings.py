import SimplerLLM.language.llm_providers.openai_llm as openai_llm
import SimplerLLM.language.llm_providers.voyage_llm as voyage_llm
import SimplerLLM.language.llm_providers.cohere_llm as cohere_llm
from enum import Enum
import os

class EmbeddingsProvider(Enum):
    OPENAI = 1
    VOYAGE = 2
    COHERE = 3


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
        user_id=None,
    ):
        if provider == EmbeddingsProvider.OPENAI:
            return OpenAIEmbeddings(provider, model_name, api_key, user_id)
        if provider == EmbeddingsProvider.VOYAGE:
            return VoyageEmbeddings(provider, model_name, api_key, user_id)
        if provider == EmbeddingsProvider.COHERE:
            return CohereEmbeddings(provider, model_name, api_key, user_id)
        else:
            return None

    def set_model(self, provider):
        if not isinstance(provider, EmbeddingsProvider):
            raise ValueError("Provider must be an instance of EmbeddingsProvider Enum")
        self.provider = provider


class OpenAIEmbeddings(EmbeddingsLLM):
    def __init__(self, provider, model_name, api_key, user_id=None):
        super().__init__(provider, model_name, api_key, user_id)
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
            api_key=self.api_key
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
            api_key=self.api_key
        )


class VoyageEmbeddings(EmbeddingsLLM):
    def __init__(self, provider, model_name, api_key, user_id=None):
        super().__init__(provider, model_name, api_key, user_id)
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY", "")

    def generate_embeddings(
        self,
        user_input,
        model_name=None,
        full_response=False,
        input_type=None,
        output_dimension=None,
        output_dtype="float"
    ):
        """
        Generate embeddings using Voyage AI.
        
        Args:
            user_input (str or list): Text(s) to embed
            model_name (str, optional): Model name override
            full_response (bool): Whether to return full response object
            input_type (str, optional): "query" or "document" for retrieval optimization
            output_dimension (int, optional): Embedding dimension (256, 512, 1024, 2048)
            output_dtype (str, optional): Data type ("float", "int8", "uint8", "binary", "ubinary")
        """
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name
        
        return voyage_llm.generate_embeddings(
            user_input=user_input,
            model_name=model_name,
            full_response=full_response,
            api_key=self.api_key,
            input_type=input_type,
            output_dimension=output_dimension,
            output_dtype=output_dtype
        )
    
    async def generate_embeddings_async(
        self,
        user_input,
        model_name=None,
        full_response=False,
        input_type=None,
        output_dimension=None,
        output_dtype="float"
    ):
        """
        Asynchronously generate embeddings using Voyage AI.
        
        Args:
            user_input (str or list): Text(s) to embed
            model_name (str, optional): Model name override
            full_response (bool): Whether to return full response object
            input_type (str, optional): "query" or "document" for retrieval optimization
            output_dimension (int, optional): Embedding dimension (256, 512, 1024, 2048)
            output_dtype (str, optional): Data type ("float", "int8", "uint8", "binary", "ubinary")
        """
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name

        return await voyage_llm.generate_embeddings_async(
            user_input=user_input,
            model_name=model_name,
            full_response=full_response,
            api_key=self.api_key,
            input_type=input_type,
            output_dimension=output_dimension,
            output_dtype=output_dtype
        )


class CohereEmbeddings(EmbeddingsLLM):
    def __init__(self, provider, model_name, api_key, user_id=None):
        super().__init__(provider, model_name, api_key, user_id)
        self.api_key = api_key or os.getenv("COHERE_API_KEY", "")

    def generate_embeddings(
        self,
        user_input,
        model_name=None,
        full_response=False,
        input_type="search_document",
        embedding_types=None,
        truncate="END"
    ):
        """
        Generate embeddings using Cohere.
        
        Args:
            user_input (str or list): Text(s) to embed
            model_name (str, optional): Model name override
            full_response (bool): Whether to return full response object
            input_type (str, optional): "search_document", "search_query", "classification", "clustering"
            embedding_types (list, optional): List of embedding types to return
            truncate (str, optional): "START", "END", or "NONE"
        """
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name
        
        return cohere_llm.generate_embeddings(
            user_input=user_input,
            model_name=model_name,
            full_response=full_response,
            api_key=self.api_key,
            input_type=input_type,
            embedding_types=embedding_types,
            truncate=truncate
        )
    
    async def generate_embeddings_async(
        self,
        user_input,
        model_name=None,
        full_response=False,
        input_type="search_document",
        embedding_types=None,
        truncate="END"
    ):
        """
        Asynchronously generate embeddings using Cohere.
        
        Args:
            user_input (str or list): Text(s) to embed
            model_name (str, optional): Model name override
            full_response (bool): Whether to return full response object
            input_type (str, optional): "search_document", "search_query", "classification", "clustering"
            embedding_types (list, optional): List of embedding types to return
            truncate (str, optional): "START", "END", or "NONE"
        """
        # Use instance values as defaults if not provided
        model_name = model_name if model_name is not None else self.model_name

        return await cohere_llm.generate_embeddings_async(
            user_input=user_input,
            model_name=model_name,
            full_response=full_response,
            api_key=self.api_key,
            input_type=input_type,
            embedding_types=embedding_types,
            truncate=truncate
        )

