"""
Vector Database Factory and Base Classes.

This module provides the VectorDB factory class and VectorProvider enum
for creating vector database instances with unified interfaces.

Supported Providers:
    LOCAL: In-memory vector database with file persistence
    QDRANT: Qdrant vector database (self-hosted or cloud)

Example:
    >>> from SimplerLLM.vectors import VectorDB, VectorProvider
    >>>
    >>> # Create local vector database
    >>> db = VectorDB.create(
    ...     provider=VectorProvider.LOCAL,
    ...     db_folder="./my_vectors"
    ... )
    >>>
    >>> # Add a vector
    >>> vector_id = db.add_vector(
    ...     vector=[0.1, 0.2, 0.3],
    ...     meta={"text": "Hello world"}
    ... )
    >>>
    >>> # Search for similar vectors
    >>> results = db.top_cosine_similarity(
    ...     target_vector=[0.1, 0.2, 0.3],
    ...     top_n=5
    ... )
    >>> for vid, meta, score in results:
    ...     print(f"Score: {score:.3f}, Text: {meta.get('text')}")
    >>>
    >>> # Create Qdrant database
    >>> db = VectorDB.create(
    ...     provider=VectorProvider.QDRANT,
    ...     url="localhost",
    ...     port=6333,
    ...     collection_name="my_collection"
    ... )

See Also:
    - SimplerLLM.language.embeddings for generating embeddings
"""

from enum import Enum
from typing import Optional, Union

from .exceptions import VectorDBError


class VectorProvider(Enum):
    """
    Enumeration of supported vector database providers.

    Attributes:
        LOCAL: In-memory vector database with pickle-based file persistence.
               Fast for development and small datasets.
               Data stored in .svdb files.

        QDRANT: Qdrant vector database for production deployments.
                Supports self-hosted instances and Qdrant Cloud.
                Provides automatic persistence and advanced filtering.

    Example:
        >>> from SimplerLLM.vectors import VectorDB, VectorProvider
        >>>
        >>> # Local database (development)
        >>> db = VectorDB.create(
        ...     provider=VectorProvider.LOCAL,
        ...     db_folder="./vectors"
        ... )
        >>>
        >>> # Qdrant database (production)
        >>> db = VectorDB.create(
        ...     provider=VectorProvider.QDRANT,
        ...     url="localhost",
        ...     port=6333
        ... )
    """
    LOCAL = 1
    QDRANT = 2


class VectorDB:
    """
    Factory class for creating vector database instances.

    Use the static `create()` method to instantiate provider-specific
    vector database wrappers with a unified interface.

    This class provides:
        - Factory method for creating database instances
        - Unified interface across all providers
        - Provider-specific configuration handling

    Example:
        >>> from SimplerLLM.vectors import VectorDB, VectorProvider
        >>>
        >>> # Create a local vector database
        >>> db = VectorDB.create(
        ...     provider=VectorProvider.LOCAL,
        ...     db_folder="./my_vectors",
        ...     verbose=True
        ... )
        >>>
        >>> # Add vectors
        >>> id1 = db.add_vector([0.1, 0.2, 0.3], {"text": "Document 1"})
        >>> id2 = db.add_vector([0.2, 0.3, 0.4], {"text": "Document 2"})
        >>>
        >>> # Search
        >>> results = db.top_cosine_similarity([0.15, 0.25, 0.35], top_n=2)
        >>> for vid, meta, score in results:
        ...     print(f"{meta['text']}: {score:.3f}")
        >>>
        >>> # Save and load
        >>> db.save_to_disk("my_collection")
        >>> db.load_from_disk("my_collection")
    """

    @staticmethod
    def create(
        provider: VectorProvider = VectorProvider.LOCAL,
        verbose: bool = False,
        **config
    ) -> Union['LocalVectorDB', 'QdrantVectorDB']:
        """
        Factory method to create a vector database instance.

        Args:
            provider: VectorProvider enum (LOCAL or QDRANT)
            verbose: Enable verbose logging (default: False)
            **config: Provider-specific configuration parameters

        Provider-specific config:

            LOCAL:
                - db_folder (str): Path to store database files (default: "./vectors")
                - dimension (int): Vector dimension for validation (optional)

            QDRANT:
                - url (str): Qdrant server URL (default: 'localhost')
                - port (int): Qdrant server port (default: 6333)
                - collection_name (str): Collection name (default: 'default_collection')
                - dimension (int): Vector dimension (optional, auto-detected)
                - api_key (str): API key for Qdrant Cloud (optional)

        Returns:
            Provider-specific VectorDB wrapper instance

        Raises:
            ValueError: If provider is not supported
            VectorDBConnectionError: If connection to remote database fails
            ImportError: If required provider dependencies are not installed

        Example:
            >>> # Local database for development
            >>> db = VectorDB.create(
            ...     provider=VectorProvider.LOCAL,
            ...     db_folder="./vectors",
            ...     verbose=True
            ... )
            >>>
            >>> # Qdrant database (self-hosted)
            >>> db = VectorDB.create(
            ...     provider=VectorProvider.QDRANT,
            ...     url="localhost",
            ...     port=6333,
            ...     collection_name="my_collection"
            ... )
            >>>
            >>> # Qdrant Cloud
            >>> db = VectorDB.create(
            ...     provider=VectorProvider.QDRANT,
            ...     url="https://your-cluster.qdrant.io",
            ...     api_key="your-api-key",
            ...     collection_name="production"
            ... )
        """
        if provider == VectorProvider.LOCAL:
            from .wrappers.local_wrapper import LocalVectorDB
            return LocalVectorDB(verbose=verbose, **config)
        elif provider == VectorProvider.QDRANT:
            from .wrappers.qdrant_wrapper import QdrantVectorDB
            return QdrantVectorDB(verbose=verbose, **config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def set_provider(provider: VectorProvider) -> None:
        """
        Validate a VectorProvider enum value.

        Args:
            provider: VectorProvider to validate

        Raises:
            ValueError: If provider is not a valid VectorProvider

        Note:
            This is a utility method. To change providers, create a new
            database instance using VectorDB.create().
        """
        if not isinstance(provider, VectorProvider):
            raise ValueError("Provider must be an instance of VectorProvider Enum")
