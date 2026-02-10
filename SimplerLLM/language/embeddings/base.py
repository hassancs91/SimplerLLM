"""
Embeddings base class and factory.

This module provides the EmbeddingsLLM factory class for creating
embedding instances for different providers.
"""

from typing import Optional

from .models import EmbeddingsProvider
from .providers import (
    BaseEmbeddings,
    OpenAIEmbeddings,
    VoyageEmbeddings,
    CohereEmbeddings,
)


# Default model names for each provider
DEFAULT_MODELS = {
    EmbeddingsProvider.OPENAI: "text-embedding-3-small",
    EmbeddingsProvider.VOYAGE: "voyage-3",
    EmbeddingsProvider.COHERE: "embed-english-v3.0",
}


class EmbeddingsLLM:
    """
    Factory class for creating embedding instances.

    Use the static `create()` method to instantiate provider-specific
    embedding classes. Do not instantiate EmbeddingsLLM directly for
    generating embeddings - use the returned provider instance.

    Supported Providers:
        - OPENAI: OpenAI's text-embedding models
        - VOYAGE: Voyage AI embeddings with retrieval optimization
        - COHERE: Cohere embeddings with multilingual support

    Example:
        >>> from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider
        >>>
        >>> # Create OpenAI embeddings instance
        >>> embeddings = EmbeddingsLLM.create(
        ...     provider=EmbeddingsProvider.OPENAI,
        ...     model_name="text-embedding-3-small"
        ... )
        >>>
        >>> # Generate embeddings
        >>> vector = embeddings.generate_embeddings("Hello, world!")
        >>> print(f"Embedding dimension: {len(vector)}")
        >>>
        >>> # Async generation
        >>> import asyncio
        >>> vector = asyncio.run(
        ...     embeddings.generate_embeddings_async("Hello, world!")
        ... )

    See Also:
        - OpenAIEmbeddings: OpenAI-specific implementation
        - VoyageEmbeddings: Voyage AI-specific implementation
        - CohereEmbeddings: Cohere-specific implementation
    """

    def __init__(
        self,
        provider: EmbeddingsProvider = EmbeddingsProvider.OPENAI,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize EmbeddingsLLM base attributes.

        Note: Use EmbeddingsLLM.create() instead of direct instantiation
        to get a provider-specific instance with full functionality.

        Args:
            provider: The embedding provider to use.
            model_name: Model identifier for the provider.
            api_key: API key for authentication (falls back to env var).
            user_id: Optional user identifier for tracking.
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.user_id = user_id

    @staticmethod
    def create(
        provider: Optional[EmbeddingsProvider] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[BaseEmbeddings]:
        """
        Create an embeddings instance for the specified provider.

        This is the recommended way to create embedding instances. The factory
        returns a provider-specific instance with full functionality.

        Args:
            provider: EmbeddingsProvider enum value (required).
                - EmbeddingsProvider.OPENAI: OpenAI embeddings
                - EmbeddingsProvider.VOYAGE: Voyage AI embeddings
                - EmbeddingsProvider.COHERE: Cohere embeddings
            model_name: Model identifier. If not provided, uses provider default:
                - OPENAI: "text-embedding-3-small"
                - VOYAGE: "voyage-3"
                - COHERE: "embed-english-v3.0"
            api_key: API key for authentication. Falls back to environment
                variables (OPENAI_API_KEY, VOYAGE_API_KEY, COHERE_API_KEY).
            user_id: Optional user identifier for tracking/billing.

        Returns:
            Provider-specific embeddings instance (OpenAIEmbeddings,
            VoyageEmbeddings, or CohereEmbeddings), or None if provider
            is not specified.

        Raises:
            ValueError: If provider is not a valid EmbeddingsProvider enum.

        Example:
            >>> # OpenAI with default model
            >>> embeddings = EmbeddingsLLM.create(
            ...     provider=EmbeddingsProvider.OPENAI
            ... )
            >>>
            >>> # Voyage with custom model
            >>> embeddings = EmbeddingsLLM.create(
            ...     provider=EmbeddingsProvider.VOYAGE,
            ...     model_name="voyage-3-lite"
            ... )
            >>>
            >>> # Cohere with explicit API key
            >>> embeddings = EmbeddingsLLM.create(
            ...     provider=EmbeddingsProvider.COHERE,
            ...     model_name="embed-multilingual-v3.0",
            ...     api_key="your-api-key"
            ... )
        """
        if provider is None:
            return None

        if not isinstance(provider, EmbeddingsProvider):
            raise ValueError(
                f"provider must be an EmbeddingsProvider enum, got {type(provider).__name__}. "
                f"Use EmbeddingsProvider.OPENAI, EmbeddingsProvider.VOYAGE, or EmbeddingsProvider.COHERE"
            )

        # Use default model if not specified
        model_name = model_name or DEFAULT_MODELS.get(provider)

        if provider == EmbeddingsProvider.OPENAI:
            return OpenAIEmbeddings(provider, model_name, api_key, user_id)
        elif provider == EmbeddingsProvider.VOYAGE:
            return VoyageEmbeddings(provider, model_name, api_key, user_id)
        elif provider == EmbeddingsProvider.COHERE:
            return CohereEmbeddings(provider, model_name, api_key, user_id)
        else:
            return None

    def set_model(self, provider: EmbeddingsProvider) -> None:
        """
        Set the provider for this instance.

        Args:
            provider: The EmbeddingsProvider enum value to set.

        Raises:
            ValueError: If provider is not an EmbeddingsProvider enum.
        """
        if not isinstance(provider, EmbeddingsProvider):
            raise ValueError("Provider must be an instance of EmbeddingsProvider Enum")
        self.provider = provider
