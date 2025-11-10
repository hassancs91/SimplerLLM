from enum import Enum
import os
from SimplerLLM.utils.custom_verbose import verbose_print


class TTSProvider(Enum):
    """Enumeration of supported TTS providers."""
    OPENAI = 1
    ELEVENLABS = 2
    # Future providers can be added here:
    # GOOGLE = 3
    # AZURE = 4


class TTS:
    """
    Base class for Text-to-Speech functionality.
    Provides a unified interface across different TTS providers.
    """

    def __init__(
        self,
        provider=TTSProvider.OPENAI,
        model_name="tts-1",
        voice="alloy",
        api_key=None,
        verbose=False,
    ):
        """
        Initialize TTS instance.

        Args:
            provider: TTS provider to use (TTSProvider enum)
            model_name: Model to use (e.g., "tts-1", "tts-1-hd")
            voice: Default voice to use (e.g., "alloy", "nova", "shimmer")
            api_key: API key for the provider (uses env var if not provided)
            verbose: Enable verbose logging
        """
        self.provider = provider
        self.model_name = model_name
        self.voice = voice
        self.api_key = api_key
        self.verbose = verbose

        if self.verbose:
            verbose_print(
                f"Initializing {provider.name} TTS with model: {model_name}, voice: {voice}",
                "info"
            )

    @staticmethod
    def create(
        provider=None,
        model_name=None,
        voice=None,
        api_key=None,
        verbose=False,
    ):
        """
        Factory method to create TTS instances for different providers.

        Args:
            provider: TTS provider (TTSProvider enum)
            model_name: Model to use (provider-specific)
            voice: Default voice to use
            api_key: API key for the provider
            verbose: Enable verbose logging

        Returns:
            Provider-specific TTS instance (e.g., OpenAITTS)
        """
        if provider == TTSProvider.OPENAI:
            from .wrappers.openai_wrapper import OpenAITTS
            return OpenAITTS(
                provider=provider,
                model_name=model_name or "tts-1",
                voice=voice or "alloy",
                api_key=api_key,
                verbose=verbose,
            )
        elif provider == TTSProvider.ELEVENLABS:
            from .wrappers.elevenlabs_wrapper import ElevenLabsTTS
            return ElevenLabsTTS(
                provider=provider,
                model_name=model_name or "eleven_turbo_v2",
                voice=voice or "21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                api_key=api_key,
                verbose=verbose,
            )
        # Future providers can be added here
        # if provider == TTSProvider.GOOGLE:
        #     from .wrappers.google_wrapper import GoogleTTS
        #     return GoogleTTS(...)
        else:
            return None

    def prepare_params(self, voice=None, model=None, speed=None):
        """
        Prepare parameters for TTS generation, using instance defaults
        if parameters are not provided.

        Args:
            voice: Voice to use (None = use instance default)
            model: Model to use (None = use instance default)
            speed: Speed to use (None = use default 1.0)

        Returns:
            Dictionary of parameters
        """
        return {
            "voice": voice if voice is not None else self.voice,
            "model_name": model if model is not None else self.model_name,
            "speed": speed if speed is not None else 1.0,
        }

    def set_provider(self, provider):
        """
        Set the TTS provider.

        Args:
            provider: TTSProvider enum value
        """
        if not isinstance(provider, TTSProvider):
            raise ValueError("Provider must be an instance of TTSProvider Enum")
        self.provider = provider
