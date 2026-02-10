"""
TTS Factory

Factory class for creating TTS provider instances.
"""

from typing import Optional

from .base import TTSBase
from .models import TTSProvider, TTSError


class TTS:
    """
    Factory for creating TTS provider instances.

    Example:
        # Create OpenAI TTS
        tts = TTS.create(TTSProvider.OPENAI)

        # Create ElevenLabs TTS
        tts = TTS.create(TTSProvider.ELEVENLABS)

        # With custom defaults
        tts = TTS.create(
            TTSProvider.OPENAI,
            model="gpt-4o-mini-tts",
            voice="coral"
        )
    """

    @staticmethod
    def create(
        provider: TTSProvider = TTSProvider.OPENAI,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSBase:
        """
        Create a TTS provider instance.

        Args:
            provider: TTS provider to use (TTSProvider.OPENAI or TTSProvider.ELEVENLABS)
            api_key: API key for the provider (uses env var if not provided)
            model: Default model to use
            voice: Default voice to use

        Returns:
            Provider-specific TTS instance (OpenAITTS or ElevenLabsTTS)

        Raises:
            TTSError: If provider is not supported
        """
        if provider == TTSProvider.OPENAI:
            from .providers.openai_tts import OpenAITTS
            kwargs = {"api_key": api_key}
            if model is not None:
                kwargs["default_model"] = model
            if voice is not None:
                kwargs["default_voice"] = voice
            return OpenAITTS(**kwargs)

        elif provider == TTSProvider.ELEVENLABS:
            from .providers.elevenlabs_tts import ElevenLabsTTS
            kwargs = {"api_key": api_key}
            if model is not None:
                kwargs["default_model"] = model
            if voice is not None:
                kwargs["default_voice"] = voice
            return ElevenLabsTTS(**kwargs)

        elif provider == TTSProvider.LAHAJATI:
            from .providers.lahajati_tts import LahajatiTTS
            kwargs = {"api_key": api_key}
            if model is not None:
                kwargs["default_model"] = model
            if voice is not None:
                kwargs["default_voice"] = voice
            return LahajatiTTS(**kwargs)

        else:
            raise TTSError(f"Unsupported provider: {provider}")
