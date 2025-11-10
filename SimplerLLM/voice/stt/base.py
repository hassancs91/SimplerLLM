from enum import Enum
import os
from SimplerLLM.utils.custom_verbose import verbose_print


class STTProvider(Enum):
    """Enumeration of supported STT providers."""
    OPENAI = 1
    # Future providers can be added here:
    # ASSEMBLYAI = 2
    # DEEPGRAM = 3
    # WHISPER_LOCAL = 4


class STT:
    """
    Base class for Speech-to-Text functionality.
    Provides a unified interface across different STT providers.
    """

    def __init__(
        self,
        provider=STTProvider.OPENAI,
        model_name="whisper-1",
        api_key=None,
        verbose=False,
    ):
        """
        Initialize STT instance.

        Args:
            provider: STT provider to use (STTProvider enum)
            model_name: Model to use (e.g., "whisper-1")
            api_key: API key for the provider (uses env var if not provided)
            verbose: Enable verbose logging
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.verbose = verbose

        if self.verbose:
            verbose_print(
                f"Initializing {provider.name} STT with model: {model_name}",
                "info"
            )

    @staticmethod
    def create(
        provider=None,
        model_name=None,
        api_key=None,
        verbose=False,
    ):
        """
        Factory method to create STT instances for different providers.

        Args:
            provider: STT provider (STTProvider enum)
            model_name: Model to use (provider-specific)
            api_key: API key for the provider
            verbose: Enable verbose logging

        Returns:
            Provider-specific STT instance (e.g., OpenAISTT)
        """
        if provider == STTProvider.OPENAI:
            from .wrappers.openai_wrapper import OpenAISTT
            return OpenAISTT(
                provider=provider,
                model_name=model_name or "whisper-1",
                api_key=api_key,
                verbose=verbose,
            )
        # Future providers can be added here
        # if provider == STTProvider.ASSEMBLYAI:
        #     from .wrappers.assemblyai_wrapper import AssemblyAISTT
        #     return AssemblyAISTT(...)
        else:
            return None

    def prepare_params(self, model=None, language=None, temperature=None):
        """
        Prepare parameters for STT transcription, using instance defaults
        if parameters are not provided.

        Args:
            model: Model to use (None = use instance default)
            language: Language code (None = auto-detect)
            temperature: Temperature for transcription (None = use default 0.0)

        Returns:
            Dictionary of parameters
        """
        return {
            "model_name": model if model is not None else self.model_name,
            "language": language,
            "temperature": temperature if temperature is not None else 0.0,
        }

    def set_provider(self, provider):
        """
        Set the STT provider.

        Args:
            provider: STTProvider enum value
        """
        if not isinstance(provider, STTProvider):
            raise ValueError("Provider must be an instance of STTProvider Enum")
        self.provider = provider
