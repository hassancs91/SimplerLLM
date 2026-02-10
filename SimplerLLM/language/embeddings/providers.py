"""
Embedding provider implementations.

This module contains the concrete implementations for each embedding provider:
- OpenAIEmbeddings: OpenAI's text-embedding models
- VoyageEmbeddings: Voyage AI embeddings with retrieval optimization
- CohereEmbeddings: Cohere embeddings with multilingual support
"""

import os
from typing import List, Optional, Union, Any

import SimplerLLM.language.llm_providers.openai_llm as openai_llm
import SimplerLLM.language.llm_providers.voyage_llm as voyage_llm
import SimplerLLM.language.llm_providers.cohere_llm as cohere_llm
from SimplerLLM.language.llm_providers.llm_response_models import LLMEmbeddingsResponse

from .models import EmbeddingsProvider


class BaseEmbeddings:
    """
    Base class for embedding providers.

    This class defines the common interface and attributes for all
    embedding providers. Do not instantiate directly.

    Attributes:
        provider: The EmbeddingsProvider enum value.
        model_name: The model identifier for the provider.
        api_key: API key for authentication.
        user_id: Optional user identifier for tracking/billing.
    """

    def __init__(
        self,
        provider: EmbeddingsProvider,
        model_name: str,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.user_id = user_id

    def set_model(self, provider: EmbeddingsProvider) -> None:
        """
        Set the provider for this embeddings instance.

        Args:
            provider: The EmbeddingsProvider enum value to set.

        Raises:
            ValueError: If provider is not an EmbeddingsProvider enum.
        """
        if not isinstance(provider, EmbeddingsProvider):
            raise ValueError("Provider must be an instance of EmbeddingsProvider Enum")
        self.provider = provider


class OpenAIEmbeddings(BaseEmbeddings):
    """
    OpenAI embeddings implementation.

    Generates text embeddings using OpenAI's embedding API. Supports
    text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.

    Attributes:
        provider: Always EmbeddingsProvider.OPENAI
        model_name: OpenAI model identifier (default: "text-embedding-3-small")
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)

    Example:
        >>> embeddings = OpenAIEmbeddings(
        ...     provider=EmbeddingsProvider.OPENAI,
        ...     model_name="text-embedding-3-small"
        ... )
        >>> vector = embeddings.generate_embeddings("Hello world")
        >>> print(len(vector))  # 1536
    """

    def __init__(
        self,
        provider: EmbeddingsProvider,
        model_name: str,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__(provider, model_name, api_key, user_id)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def generate_embeddings(
        self,
        user_input: Union[str, List[str]],
        model_name: Optional[str] = None,
        full_response: bool = False,
    ) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]:
        """
        Generate embeddings using OpenAI's embedding API.

        Args:
            user_input: Text or list of texts to embed. Single strings return
                a single embedding vector. Lists return a list of vectors.
            model_name: Model name override. If not provided, uses the
                instance's model_name.
            full_response: If True, returns LLMEmbeddingsResponse with metadata.
                If False, returns just the embedding vector(s).

        Returns:
            If full_response=False:
                - Single input: List[float] - embedding vector
                - List input: List[List[float]] - list of embedding vectors
            If full_response=True:
                LLMEmbeddingsResponse with generated_embedding, model,
                process_time, and llm_provider_response.

        Raises:
            ValueError: If user_input is empty or None.
            Exception: On API errors after retry attempts exhausted.

        Example:
            >>> embeddings = OpenAIEmbeddings(
            ...     provider=EmbeddingsProvider.OPENAI,
            ...     model_name="text-embedding-3-small"
            ... )
            >>> # Single text
            >>> vector = embeddings.generate_embeddings("Hello world")
            >>> print(f"Dimensions: {len(vector)}")  # 1536
            >>>
            >>> # Multiple texts with metadata
            >>> response = embeddings.generate_embeddings(
            ...     ["Text 1", "Text 2"],
            ...     full_response=True
            ... )
            >>> print(f"Time: {response.process_time:.2f}s")
        """
        model_name = model_name if model_name is not None else self.model_name

        return openai_llm.generate_embeddings(
            user_input=user_input,
            model_name=model_name,
            full_response=full_response,
            api_key=self.api_key
        )

    async def generate_embeddings_async(
        self,
        user_input: Union[str, List[str]],
        model_name: Optional[str] = None,
        full_response: bool = False,
    ) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]:
        """
        Asynchronously generate embeddings using OpenAI's embedding API.

        This is the async version of generate_embeddings(). See
        generate_embeddings() for full parameter documentation.

        Args:
            user_input: Text or list of texts to embed.
            model_name: Model name override.
            full_response: If True, returns LLMEmbeddingsResponse.

        Returns:
            Embedding vector(s) or LLMEmbeddingsResponse.

        Example:
            >>> import asyncio
            >>> embeddings = OpenAIEmbeddings(
            ...     provider=EmbeddingsProvider.OPENAI,
            ...     model_name="text-embedding-3-small"
            ... )
            >>> vector = asyncio.run(
            ...     embeddings.generate_embeddings_async("Hello world")
            ... )
        """
        model_name = model_name if model_name is not None else self.model_name

        return await openai_llm.generate_embeddings_async(
            user_input=user_input,
            model_name=model_name,
            full_response=full_response,
            api_key=self.api_key
        )


class VoyageEmbeddings(BaseEmbeddings):
    """
    Voyage AI embeddings implementation.

    Generates text embeddings using Voyage AI's embedding API. Supports
    retrieval-optimized embeddings with input_type parameter and
    configurable output dimensions.

    Attributes:
        provider: Always EmbeddingsProvider.VOYAGE
        model_name: Voyage model identifier (default: "voyage-3")
        api_key: Voyage API key (falls back to VOYAGE_API_KEY env var)

    Example:
        >>> embeddings = VoyageEmbeddings(
        ...     provider=EmbeddingsProvider.VOYAGE,
        ...     model_name="voyage-3"
        ... )
        >>> # For search queries
        >>> query_vec = embeddings.generate_embeddings(
        ...     "search query",
        ...     input_type="query"
        ... )
        >>> # For documents to index
        >>> doc_vec = embeddings.generate_embeddings(
        ...     "document text",
        ...     input_type="document"
        ... )
    """

    def __init__(
        self,
        provider: EmbeddingsProvider,
        model_name: str,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__(provider, model_name, api_key, user_id)
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY", "")

    def generate_embeddings(
        self,
        user_input: Union[str, List[str]],
        model_name: Optional[str] = None,
        full_response: bool = False,
        input_type: Optional[str] = None,
        output_dimension: Optional[int] = None,
        output_dtype: str = "float",
    ) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]:
        """
        Generate embeddings using Voyage AI.

        Args:
            user_input: Text or list of texts to embed.
            model_name: Model name override.
            full_response: Whether to return full response object.
            input_type: Semantic hint for embeddings optimization.
                - "query": For search queries (shorter text, finding matches)
                - "document": For documents to be searched (longer text, indexed)
                - None: No optimization (default)
            output_dimension: Embedding dimension to return. Options:
                256, 512, 1024, 2048. Lower dimensions = faster search.
            output_dtype: Data type for embeddings. Options:
                - "float": Standard floating point (default)
                - "int8": 8-bit integers (smaller, faster)
                - "uint8": Unsigned 8-bit integers
                - "binary": Binary embeddings
                - "ubinary": Unsigned binary embeddings

        Returns:
            If full_response=False:
                - Single input: List[float] - embedding vector
                - List input: List[List[float]] - list of embedding vectors
            If full_response=True:
                LLMEmbeddingsResponse with metadata.

        Raises:
            ImportError: If voyageai package is not installed.
            ValueError: If user_input is empty or None.
            Exception: On API errors after retry attempts exhausted.

        Example:
            >>> embeddings = VoyageEmbeddings(
            ...     provider=EmbeddingsProvider.VOYAGE,
            ...     model_name="voyage-3"
            ... )
            >>> # Optimized for retrieval
            >>> query_emb = embeddings.generate_embeddings(
            ...     "What is machine learning?",
            ...     input_type="query"
            ... )
            >>> # Smaller dimensions for efficiency
            >>> small_emb = embeddings.generate_embeddings(
            ...     "Some text",
            ...     output_dimension=512
            ... )
        """
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
        user_input: Union[str, List[str]],
        model_name: Optional[str] = None,
        full_response: bool = False,
        input_type: Optional[str] = None,
        output_dimension: Optional[int] = None,
        output_dtype: str = "float",
    ) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]:
        """
        Asynchronously generate embeddings using Voyage AI.

        This is the async version of generate_embeddings(). See
        generate_embeddings() for full parameter documentation.

        Args:
            user_input: Text or list of texts to embed.
            model_name: Model name override.
            full_response: Whether to return full response object.
            input_type: "query" or "document" for retrieval optimization.
            output_dimension: Embedding dimension (256, 512, 1024, 2048).
            output_dtype: Data type ("float", "int8", "uint8", "binary", "ubinary").

        Returns:
            Embedding vector(s) or LLMEmbeddingsResponse.

        Example:
            >>> import asyncio
            >>> embeddings = VoyageEmbeddings(
            ...     provider=EmbeddingsProvider.VOYAGE,
            ...     model_name="voyage-3"
            ... )
            >>> vector = asyncio.run(
            ...     embeddings.generate_embeddings_async(
            ...         "Hello world",
            ...         input_type="query"
            ...     )
            ... )
        """
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


class CohereEmbeddings(BaseEmbeddings):
    """
    Cohere embeddings implementation.

    Generates text embeddings using Cohere's embedding API. Supports
    multilingual embeddings and search-optimized input types.

    Attributes:
        provider: Always EmbeddingsProvider.COHERE
        model_name: Cohere model identifier (default: "embed-english-v3.0")
        api_key: Cohere API key (falls back to COHERE_API_KEY env var)

    Example:
        >>> embeddings = CohereEmbeddings(
        ...     provider=EmbeddingsProvider.COHERE,
        ...     model_name="embed-multilingual-v3.0"
        ... )
        >>> # Multilingual support
        >>> vectors = embeddings.generate_embeddings([
        ...     "Hello world",
        ...     "Bonjour le monde",
        ...     "Hola mundo"
        ... ])
    """

    def __init__(
        self,
        provider: EmbeddingsProvider,
        model_name: str,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__(provider, model_name, api_key, user_id)
        self.api_key = api_key or os.getenv("COHERE_API_KEY", "")

    def generate_embeddings(
        self,
        user_input: Union[str, List[str]],
        model_name: Optional[str] = None,
        full_response: bool = False,
        input_type: str = "search_document",
        embedding_types: Optional[List[str]] = None,
        truncate: str = "END",
    ) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]:
        """
        Generate embeddings using Cohere.

        Args:
            user_input: Text or list of texts to embed.
            model_name: Model name override.
            full_response: Whether to return full response object.
            input_type: Semantic context for embeddings. Options:
                - "search_document": For documents to be searched (default)
                - "search_query": For search queries
                - "classification": For classification tasks
                - "clustering": For clustering tasks
            embedding_types: List of specific embedding types to return.
                Options vary by model. None uses default.
            truncate: How to handle texts exceeding max length. Options:
                - "END": Truncate from end (default)
                - "START": Truncate from beginning
                - "NONE": Raise error if text exceeds limit

        Returns:
            If full_response=False:
                - Single input: List[float] - embedding vector
                - List input: List[List[float]] - list of embedding vectors
            If full_response=True:
                LLMEmbeddingsResponse with metadata.

        Raises:
            ValueError: If COHERE_API_KEY is not set.
            ValueError: If user_input is empty or None.
            Exception: On API errors after retry attempts exhausted.

        Example:
            >>> embeddings = CohereEmbeddings(
            ...     provider=EmbeddingsProvider.COHERE,
            ...     model_name="embed-english-v3.0"
            ... )
            >>> # For documents to be searched
            >>> doc_emb = embeddings.generate_embeddings(
            ...     "This is a document about AI.",
            ...     input_type="search_document"
            ... )
            >>> # For search queries
            >>> query_emb = embeddings.generate_embeddings(
            ...     "What is artificial intelligence?",
            ...     input_type="search_query"
            ... )
        """
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
        user_input: Union[str, List[str]],
        model_name: Optional[str] = None,
        full_response: bool = False,
        input_type: str = "search_document",
        embedding_types: Optional[List[str]] = None,
        truncate: str = "END",
    ) -> Union[List[float], List[List[float]], LLMEmbeddingsResponse]:
        """
        Asynchronously generate embeddings using Cohere.

        This is the async version of generate_embeddings(). See
        generate_embeddings() for full parameter documentation.

        Args:
            user_input: Text or list of texts to embed.
            model_name: Model name override.
            full_response: Whether to return full response object.
            input_type: "search_document", "search_query", "classification", "clustering".
            embedding_types: List of specific embedding types to return.
            truncate: "START", "END", or "NONE".

        Returns:
            Embedding vector(s) or LLMEmbeddingsResponse.

        Example:
            >>> import asyncio
            >>> embeddings = CohereEmbeddings(
            ...     provider=EmbeddingsProvider.COHERE,
            ...     model_name="embed-multilingual-v3.0"
            ... )
            >>> vector = asyncio.run(
            ...     embeddings.generate_embeddings_async(
            ...         "Bonjour le monde",
            ...         input_type="search_document"
            ...     )
            ... )
        """
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
