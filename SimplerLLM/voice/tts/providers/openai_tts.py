"""
OpenAI TTS Provider

Implements TTS using OpenAI's Audio API.
"""

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os
import time
from typing import List, Optional

from ..base import TTSBase
from ..models import (
    Voice,
    TTSResponse,
    TTSProvider,
    TTSValidationError,
    TTSProviderError,
    TTSVoiceNotFoundError,
    OPENAI_VOICES,
    OPENAI_MODELS,
    OPENAI_FORMATS,
)

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

# OpenAI voice metadata (hardcoded since API doesn't provide this)
OPENAI_VOICE_DATA = {
    "alloy": {"gender": "neutral", "description": "Balanced, versatile voice"},
    "ash": {"gender": "male", "description": "Warm, engaging voice"},
    "ballad": {"gender": "male", "description": "Soft, melodic voice"},
    "coral": {"gender": "female", "description": "Clear, expressive voice"},
    "echo": {"gender": "male", "description": "Smooth, refined voice"},
    "fable": {"gender": "neutral", "description": "Storytelling voice"},
    "onyx": {"gender": "male", "description": "Deep, authoritative voice"},
    "nova": {"gender": "female", "description": "Bright, energetic voice"},
    "sage": {"gender": "female", "description": "Calm, wise voice"},
    "shimmer": {"gender": "female", "description": "Light, airy voice"},
    "verse": {"gender": "male", "description": "Dynamic, versatile voice"},
    "marin": {"gender": "female", "description": "Natural, warm voice"},
    "cedar": {"gender": "male", "description": "Rich, resonant voice"},
}


class OpenAITTS(TTSBase):
    """
    OpenAI TTS provider implementation.

    Supports models: tts-1, tts-1-hd, gpt-4o-mini-tts
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini-tts",
        default_voice: str = "alloy",
    ):
        """
        Initialize OpenAI TTS provider.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            default_model: Default model (tts-1, tts-1-hd, gpt-4o-mini-tts)
            default_voice: Default voice (alloy, echo, fable, etc.)
        """
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            default_voice=default_voice,
        )
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

    @property
    def provider(self) -> TTSProvider:
        return TTSProvider.OPENAI

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    def _validate_params(
        self,
        text: str,
        voice: str,
        model: str,
        speed: float,
        output_format: str,
        instructions: Optional[str] = None,
    ) -> None:
        """Validate input parameters"""
        if not text or not text.strip():
            raise TTSValidationError("Text cannot be empty")

        if voice not in OPENAI_VOICES:
            raise TTSValidationError(
                f"Invalid voice '{voice}'. Supported voices: {', '.join(OPENAI_VOICES)}"
            )

        if model not in OPENAI_MODELS:
            raise TTSValidationError(
                f"Invalid model '{model}'. Supported models: {', '.join(OPENAI_MODELS)}"
            )

        if output_format not in OPENAI_FORMATS:
            raise TTSValidationError(
                f"Invalid format '{output_format}'. Supported formats: {', '.join(OPENAI_FORMATS)}"
            )

        if not 0.25 <= speed <= 4.0:
            raise TTSValidationError(
                f"Speed must be between 0.25 and 4.0, got {speed}"
            )

        if instructions and model != "gpt-4o-mini-tts":
            raise TTSValidationError(
                "Instructions parameter is only supported with 'gpt-4o-mini-tts' model"
            )

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
        Generate speech from text using OpenAI TTS API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer, etc.)
            model: Model to use (tts-1, tts-1-hd, gpt-4o-mini-tts)
            speed: Speech speed from 0.25 to 4.0 (default: 1.0)
            output_format: Audio format (mp3, opus, aac, flac, wav, pcm)
            output_path: Optional file path to save audio
            instructions: Voice instructions (only for gpt-4o-mini-tts)

        Note:
            language_code, stability, similarity_boost, style, seed are ignored
            as they are not supported by OpenAI TTS.

        Returns:
            TTSResponse with audio data and metadata
        """
        # Get defaults
        voice = self._get_voice(voice)
        model = self._get_model(model)

        # Validate
        self._validate_params(text, voice, model, speed, output_format, instructions)

        start_time = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": model,
                    "voice": voice,
                    "input": text,
                    "speed": speed,
                    "response_format": output_format,
                }

                # Add instructions for gpt-4o-mini-tts
                if instructions and model == "gpt-4o-mini-tts":
                    request_kwargs["instructions"] = instructions

                # Create speech
                response = self.client.audio.speech.create(**request_kwargs)

                # Get audio data
                audio_data = response.content

                process_time = time.time() - start_time

                # Save to file if requested
                file_path = None
                if output_path:
                    file_path = self._save_audio(audio_data, output_path)

                return TTSResponse(
                    audio_data=audio_data if not output_path else file_path,
                    model=model,
                    voice=voice,
                    format=output_format,
                    duration_ms=None,  # OpenAI doesn't provide duration
                    file_path=file_path,
                    provider=self.provider.value,
                    process_time=process_time,
                    instructions=instructions,
                )

            except TTSValidationError:
                raise
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise TTSProviderError(
                        f"Failed to generate speech after {MAX_RETRIES} attempts: {e}",
                        provider=self.provider.value,
                    )

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
        Async version: Generate speech from text using OpenAI TTS API.

        Args:
            Same as generate_speech()

        Returns:
            TTSResponse with audio data and metadata
        """
        # Get defaults
        voice = self._get_voice(voice)
        model = self._get_model(model)

        # Validate
        self._validate_params(text, voice, model, speed, output_format, instructions)

        start_time = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": model,
                    "voice": voice,
                    "input": text,
                    "speed": speed,
                    "response_format": output_format,
                }

                # Add instructions for gpt-4o-mini-tts
                if instructions and model == "gpt-4o-mini-tts":
                    request_kwargs["instructions"] = instructions

                # Create speech
                response = await self.async_client.audio.speech.create(**request_kwargs)

                # Get audio data
                audio_data = response.content

                process_time = time.time() - start_time

                # Save to file if requested
                file_path = None
                if output_path:
                    file_path = self._save_audio(audio_data, output_path)

                return TTSResponse(
                    audio_data=audio_data if not output_path else file_path,
                    model=model,
                    voice=voice,
                    format=output_format,
                    duration_ms=None,
                    file_path=file_path,
                    provider=self.provider.value,
                    process_time=process_time,
                    instructions=instructions,
                )

            except TTSValidationError:
                raise
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise TTSProviderError(
                        f"Failed to generate speech after {MAX_RETRIES} attempts: {e}",
                        provider=self.provider.value,
                    )

    def list_voices(self) -> List[Voice]:
        """
        List all available OpenAI TTS voices.

        Returns:
            List of Voice objects
        """
        voices = []
        for voice_id in OPENAI_VOICES:
            voice_data = OPENAI_VOICE_DATA.get(voice_id, {})
            voices.append(
                Voice(
                    voice_id=voice_id,
                    name=voice_id.capitalize(),
                    language="en",
                    languages=["en"],  # OpenAI voices are primarily English
                    gender=voice_data.get("gender"),
                    preview_url=None,  # OpenAI doesn't provide preview URLs
                    provider=self.provider.value,
                    description=voice_data.get("description"),
                )
            )
        return voices

    def get_voice(self, voice_id: str) -> Voice:
        """
        Get details for a specific OpenAI voice.

        Args:
            voice_id: Voice ID to look up

        Returns:
            Voice object with details

        Raises:
            TTSVoiceNotFoundError: If voice is not found
        """
        if voice_id not in OPENAI_VOICES:
            raise TTSVoiceNotFoundError(voice_id, self.provider.value)

        voice_data = OPENAI_VOICE_DATA.get(voice_id, {})
        return Voice(
            voice_id=voice_id,
            name=voice_id.capitalize(),
            language="en",
            languages=["en"],
            gender=voice_data.get("gender"),
            preview_url=None,
            provider=self.provider.value,
            description=voice_data.get("description"),
        )
