"""
ElevenLabs TTS Provider

Implements TTS using ElevenLabs API.
"""

from elevenlabs import ElevenLabs, VoiceSettings
from elevenlabs.client import AsyncElevenLabs
from dotenv import load_dotenv
import asyncio
import os
import time
import warnings
from typing import List, Optional

from ..base import TTSBase
from ..models import (
    Voice,
    TTSResponse,
    TTSProvider,
    TTSValidationError,
    TTSProviderError,
    TTSVoiceNotFoundError,
    ELEVENLABS_MODELS,
    ELEVENLABS_FORMATS,
)

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

# Default voice ID (Rachel)
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


class ElevenLabsTTS(TTSBase):
    """
    ElevenLabs TTS provider implementation.

    Supports models: eleven_flash_v2_5, eleven_turbo_v2_5, eleven_turbo_v2,
                     eleven_multilingual_v2, eleven_monolingual_v1
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "eleven_flash_v2_5",
        default_voice: str = DEFAULT_VOICE_ID,
    ):
        """
        Initialize ElevenLabs TTS provider.

        Args:
            api_key: ElevenLabs API key (uses ELEVENLABS_API_KEY env var if not provided)
            default_model: Default model (eleven_flash_v2_5, eleven_turbo_v2, etc.)
            default_voice: Default voice ID
        """
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            default_voice=default_voice,
        )
        self._client: Optional[ElevenLabs] = None
        self._async_client: Optional[AsyncElevenLabs] = None
        self._voices_cache: Optional[List[Voice]] = None

    @property
    def provider(self) -> TTSProvider:
        return TTSProvider.ELEVENLABS

    @property
    def client(self) -> ElevenLabs:
        if self._client is None:
            # Only pass api_key if explicitly provided, otherwise let client read from env
            if self.api_key:
                self._client = ElevenLabs(api_key=self.api_key)
            else:
                self._client = ElevenLabs()
        return self._client

    @property
    def async_client(self) -> AsyncElevenLabs:
        if self._async_client is None:
            # Only pass api_key if explicitly provided, otherwise let client read from env
            if self.api_key:
                self._async_client = AsyncElevenLabs(api_key=self.api_key)
            else:
                self._async_client = AsyncElevenLabs()
        return self._async_client

    def _validate_params(
        self,
        text: str,
        model: str,
        speed: Optional[float] = None,
        output_format: str = "mp3_44100_128",
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        style: Optional[float] = None,
    ) -> None:
        """Validate input parameters"""
        if not text or not text.strip():
            raise TTSValidationError("Text cannot be empty")

        if model not in ELEVENLABS_MODELS:
            raise TTSValidationError(
                f"Invalid model '{model}'. Supported models: {', '.join(ELEVENLABS_MODELS)}"
            )

        if output_format not in ELEVENLABS_FORMATS:
            raise TTSValidationError(
                f"Invalid format '{output_format}'. Supported formats: {', '.join(ELEVENLABS_FORMATS)}"
            )

        if speed is not None and not 0.7 <= speed <= 1.2:
            raise TTSValidationError(
                f"Speed must be between 0.7 and 1.2, got {speed}"
            )

        if stability is not None and not 0.0 <= stability <= 1.0:
            raise TTSValidationError(
                f"Stability must be between 0.0 and 1.0, got {stability}"
            )

        if similarity_boost is not None and not 0.0 <= similarity_boost <= 1.0:
            raise TTSValidationError(
                f"Similarity boost must be between 0.0 and 1.0, got {similarity_boost}"
            )

        if style is not None and not 0.0 <= style <= 1.0:
            raise TTSValidationError(
                f"Style must be between 0.0 and 1.0, got {style}"
            )

    def _map_output_format(self, output_format: str) -> str:
        """Map simple format names to ElevenLabs format strings"""
        format_map = {
            "mp3": "mp3_44100_128",
            "wav": "pcm_44100",
            "pcm": "pcm_16000",
            "opus": "opus_48000_128",
        }
        return format_map.get(output_format, output_format)

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
        Generate speech from text using ElevenLabs TTS API.

        Args:
            text: Text to convert to speech
            voice: Voice ID to use
            model: Model to use (eleven_flash_v2_5, eleven_turbo_v2, etc.)
            speed: Speech speed from 0.7 to 1.2 (default: 1.0)
            output_format: Audio format (mp3, pcm, opus, or full format string)
            output_path: Optional file path to save audio
            language_code: ISO 639-1 language code (e.g., "en", "es", "fr")
            stability: Voice stability 0-1 (default: 0.5)
            similarity_boost: Similarity boost 0-1 (default: 0.75)
            style: Style exaggeration 0-1 (default: 0.0)
            seed: Seed for reproducibility

        Note:
            instructions is ignored as it's not supported by ElevenLabs.

        Returns:
            TTSResponse with audio data and metadata
        """
        # Get defaults
        voice = self._get_voice(voice)
        model = self._get_model(model)
        output_format = self._map_output_format(output_format)

        # Warn if speed is not default (ElevenLabs doesn't support speed)
        if speed != 1.0:
            warnings.warn(
                "Speed parameter is not supported by ElevenLabs and will be ignored",
                UserWarning,
                stacklevel=2
            )

        # Validate (pass None for speed since ElevenLabs doesn't support it)
        self._validate_params(text, model, None, output_format, stability, similarity_boost, style)

        start_time = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                # Prepare voice settings if any are specified
                voice_settings = None
                if any([stability is not None, similarity_boost is not None, style is not None]):
                    voice_settings = VoiceSettings(
                        stability=stability if stability is not None else 0.5,
                        similarity_boost=similarity_boost if similarity_boost is not None else 0.75,
                        style=style if style is not None else 0.0,
                        use_speaker_boost=True,
                    )

                # Build request kwargs
                request_kwargs = {
                    "voice_id": voice,
                    "text": text,
                    "model_id": model,
                    "output_format": output_format,
                }

                if voice_settings:
                    request_kwargs["voice_settings"] = voice_settings

                if language_code:
                    request_kwargs["language_code"] = language_code

                if seed is not None:
                    request_kwargs["seed"] = seed

                # Create speech
                response = self.client.text_to_speech.convert(**request_kwargs)

                # Collect audio data from generator
                audio_bytes = b"".join(chunk for chunk in response)

                process_time = time.time() - start_time

                # Save to file if requested
                file_path = None
                if output_path:
                    file_path = self._save_audio(audio_bytes, output_path)

                # Extract format name from full format string
                format_name = output_format.split("_")[0]

                return TTSResponse(
                    audio_data=audio_bytes if not output_path else file_path,
                    model=model,
                    voice=voice,
                    format=format_name,
                    duration_ms=None,
                    file_path=file_path,
                    provider=self.provider.value,
                    process_time=process_time,
                    language_code=language_code,
                    seed=seed,
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=style,
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
        Async version: Generate speech from text using ElevenLabs TTS API.

        Args:
            Same as generate_speech()

        Returns:
            TTSResponse with audio data and metadata
        """
        # Get defaults
        voice = self._get_voice(voice)
        model = self._get_model(model)
        output_format = self._map_output_format(output_format)

        # Warn if speed is not default (ElevenLabs doesn't support speed)
        if speed != 1.0:
            warnings.warn(
                "Speed parameter is not supported by ElevenLabs and will be ignored",
                UserWarning,
                stacklevel=2
            )

        # Validate (pass None for speed since ElevenLabs doesn't support it)
        self._validate_params(text, model, None, output_format, stability, similarity_boost, style)

        start_time = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                # Prepare voice settings if any are specified
                voice_settings = None
                if any([stability is not None, similarity_boost is not None, style is not None]):
                    voice_settings = VoiceSettings(
                        stability=stability if stability is not None else 0.5,
                        similarity_boost=similarity_boost if similarity_boost is not None else 0.75,
                        style=style if style is not None else 0.0,
                        use_speaker_boost=True,
                    )

                # Build request kwargs
                request_kwargs = {
                    "voice_id": voice,
                    "text": text,
                    "model_id": model,
                    "output_format": output_format,
                }

                if voice_settings:
                    request_kwargs["voice_settings"] = voice_settings

                if language_code:
                    request_kwargs["language_code"] = language_code

                if seed is not None:
                    request_kwargs["seed"] = seed

                # Create speech (returns async generator directly, no await needed)
                response = self.async_client.text_to_speech.convert(**request_kwargs)

                # Collect audio data from async generator
                audio_bytes = b"".join([chunk async for chunk in response])

                process_time = time.time() - start_time

                # Save to file if requested
                file_path = None
                if output_path:
                    file_path = self._save_audio(audio_bytes, output_path)

                # Extract format name from full format string
                format_name = output_format.split("_")[0]

                return TTSResponse(
                    audio_data=audio_bytes if not output_path else file_path,
                    model=model,
                    voice=voice,
                    format=format_name,
                    duration_ms=None,
                    file_path=file_path,
                    provider=self.provider.value,
                    process_time=process_time,
                    language_code=language_code,
                    seed=seed,
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=style,
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

    def list_voices(self, use_cache: bool = True) -> List[Voice]:
        """
        List all available ElevenLabs voices.

        Args:
            use_cache: If True, use cached voices if available

        Returns:
            List of Voice objects
        """
        if use_cache and self._voices_cache is not None:
            return self._voices_cache

        try:
            response = self.client.voices.get_all()
            voices = []

            for v in response.voices:
                # Extract language from labels if available
                labels = v.labels or {}
                language = labels.get("language", "en")

                # Get verified languages if available
                languages = []
                if hasattr(v, "verified_languages") and v.verified_languages:
                    languages = [lang.language for lang in v.verified_languages]
                elif language:
                    languages = [language]

                voices.append(
                    Voice(
                        voice_id=v.voice_id,
                        name=v.name,
                        language=language,
                        languages=languages,
                        gender=labels.get("gender"),
                        preview_url=v.preview_url,
                        provider=self.provider.value,
                        description=labels.get("description") or labels.get("use_case"),
                    )
                )

            self._voices_cache = voices
            return voices

        except Exception as e:
            raise TTSProviderError(
                f"Failed to list voices: {e}",
                provider=self.provider.value,
            )

    def get_voice(self, voice_id: str) -> Voice:
        """
        Get details for a specific ElevenLabs voice.

        Args:
            voice_id: Voice ID to look up

        Returns:
            Voice object with details

        Raises:
            TTSVoiceNotFoundError: If voice is not found
        """
        try:
            v = self.client.voices.get(voice_id=voice_id)

            # Extract language from labels if available
            labels = v.labels or {}
            language = labels.get("language", "en")

            # Get verified languages if available
            languages = []
            if hasattr(v, "verified_languages") and v.verified_languages:
                languages = [lang.language for lang in v.verified_languages]
            elif language:
                languages = [language]

            return Voice(
                voice_id=v.voice_id,
                name=v.name,
                language=language,
                languages=languages,
                gender=labels.get("gender"),
                preview_url=v.preview_url,
                provider=self.provider.value,
                description=labels.get("description") or labels.get("use_case"),
            )

        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                raise TTSVoiceNotFoundError(voice_id, self.provider.value)
            raise TTSProviderError(
                f"Failed to get voice: {e}",
                provider=self.provider.value,
            )

    def clear_cache(self) -> None:
        """Clear the voices cache"""
        self._voices_cache = None
