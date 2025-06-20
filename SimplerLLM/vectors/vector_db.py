from .vector_providers import VectorProvider


class VectorDB:
    def __init__(self, provider=VectorProvider.LOCAL, **config):
        self.provider = provider
        self.config = config

    @staticmethod
    def create(provider=None, **config):
        if provider == VectorProvider.LOCAL:
            from .local_vector_db import LocalVectorDB
            return LocalVectorDB(provider, **config)
        elif provider == VectorProvider.QDRANT:
            from .qdrant_vector_db import QdrantVectorDB
            return QdrantVectorDB(provider, **config)
        else:
            return None

    def set_provider(self, provider):
        if not isinstance(provider, VectorProvider):
            raise ValueError("Provider must be an instance of VectorProvider Enum")
        self.provider = provider
