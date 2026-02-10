"""
TTS Base Class

Abstract base class for all TTS providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import os

from .models import Voice, TTSResponse, TTSProvider


class TTSBase(ABC):
    """
    Abstract base class for Text-to-Speech providers.

    All TTS providers must implement this interface to ensure
    a unified API across different providers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_voice: Optional[str] = None,
    ):
        """
        Initialize TTS provider.

        Args:
            api_key: API key for the provider (uses env var if not provided)
            default_model: Default model to use for generation
            default_voice: Default voice to use for generation
        """
        self.api_key = api_key
        self.default_model = default_model
        self.default_voice = default_voice

    @property
    @abstractmethod
    def provider(self) -> TTSProvider:
        """Return the provider type"""
        pass

    @abstractmethod
    def generate_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: float = 1.0,
        output_format: str = "mp3",
        output_path: Optional[str] = None,
        language_code: Optional[str] = None,
        instructions: Optional[str] = None,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        style: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> TTSResponse:
        """
        Generate speech from text.

        Args:
            text: Text to convert to speech
            voice: Voice to use (provider-specific)
            model: Model to use (provider-specific)
            speed: Speech speed (provider-specific range)
            output_format: Audio format (mp3, wav, etc.)
            output_path: Path to save the audio file
            language_code: ISO 639-1 language code (ElevenLabs only)
            instructions: Voice instructions (OpenAI gpt-4o-mini-tts only)
            stability: Voice stability 0-1 (ElevenLabs only)
            similarity_boost: Similarity boost 0-1 (ElevenLabs only)
            style: Style exaggeration 0-1 (ElevenLabs only)
            seed: Seed for reproducibility (ElevenLabs only)

        Returns:
            TTSResponse with audio data and metadata
        """
        pass

    @abstractmethod
    async def generate_speech_async(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: float = 1.0,
        output_format: str = "mp3",
        output_path: Optional[str] = None,
        language_code: Optional[str] = None,
        instructions: Optional[str] = None,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        style: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> TTSResponse:
        """
        Generate speech from text asynchronously.

        Args:
            Same as generate_speech()

        Returns:
            TTSResponse with audio data and metadata
        """
        pass

    @abstractmethod
    def list_voices(self) -> List[Voice]:
        """
        List all available voices for this provider.

        Returns:
            List of Voice objects
        """
        pass

    @abstractmethod
    def get_voice(self, voice_id: str) -> Voice:
        """
        Get details for a specific voice.

        Args:
            voice_id: Voice ID to look up

        Returns:
            Voice object with details

        Raises:
            TTSVoiceNotFoundError: If voice is not found
        """
        pass

    def _get_voice(self, voice: Optional[str]) -> str:
        """Get voice to use, falling back to default"""
        return voice if voice is not None else self.default_voice

    def _get_model(self, model: Optional[str]) -> str:
        """Get model to use, falling back to default"""
        return model if model is not None else self.default_model

    def _save_audio(self, audio_data: bytes, output_path: str) -> str:
        """
        Save audio data to a file.

        Args:
            audio_data: Audio bytes to save
            output_path: Path to save to

        Returns:
            Absolute path to saved file
        """
        # Ensure directory exists (handle case where file is in current directory)
        abs_path = os.path.abspath(output_path)
        dir_path = os.path.dirname(abs_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(abs_path, "wb") as f:
            f.write(audio_data)

        return abs_path
