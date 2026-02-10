"""
Lahajati TTS Provider

Implements TTS using Lahajati's Arabic Voice API.
https://lahajati.ai/
"""

import httpx
import asyncio
import os
import time
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from ..base import TTSBase
from ..models import (
    Voice,
    TTSResponse,
    TTSProvider,
    TTSValidationError,
    TTSProviderError,
    TTSVoiceNotFoundError,
    Dialect,
    Performance,
    LAHAJATI_FORMATS,
    LAHAJATI_INPUT_MODES,
)

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


class LahajatiTTS(TTSBase):
    """
    Lahajati TTS provider implementation for Arabic voice synthesis.

    Features:
    - 500+ professional Arabic voices
    - 192+ Arabic dialects (Egyptian, Saudi Najdi, etc.)
    - Performance styles (dramatic, news, advertising, etc.)
    - Custom prompt mode for fine-grained control
    """

    BASE_URL = "https://lahajati.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_voice: Optional[str] = None,
    ):
        """
        Initialize Lahajati TTS provider.

        Args:
            api_key: Lahajati API key (uses LAHAJATI_API_KEY env var if not provided)
            default_model: Default model (not used by Lahajati, reserved for future)
            default_voice: Default voice ID to use
        """
        api_key = api_key or os.getenv("LAHAJATI_API_KEY")
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            default_voice=default_voice,
        )
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

        # Caches
        self._voices_cache: Optional[List[Voice]] = None
        self._dialects_cache: Optional[List[Dialect]] = None
        self._performances_cache: Optional[List[Performance]] = None

    @property
    def provider(self) -> TTSProvider:
        return TTSProvider.LAHAJATI

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.BASE_URL,
                headers=self._get_headers(),
                timeout=60.0,
            )
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self._get_headers(),
                timeout=60.0,
            )
        return self._async_client

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

    def _validate_params(
        self,
        text: str,
        input_mode: int,
        custom_prompt: Optional[str] = None,
        dialect_id: Optional[str] = None,
        performance_id: Optional[str] = None,
    ) -> None:
        """Validate input parameters"""
        if not text or not text.strip():
            raise TTSValidationError("Text cannot be empty")

        if input_mode not in [0, 1]:
            raise TTSValidationError(
                f"Invalid input_mode '{input_mode}'. Must be 0 (structured) or 1 (custom)"
            )

        if input_mode == 0:  # Structured mode
            if not dialect_id or not performance_id:
                raise TTSValidationError(
                    "Both dialect_id and performance_id are required when using structured mode (input_mode=0). "
                    "Use list_dialects() and list_performances() to get available options."
                )

        if input_mode == 1 and not custom_prompt:
            raise TTSValidationError(
                "custom_prompt is required when input_mode is 1 (custom)"
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
        # Lahajati-specific parameters
        input_mode: int = 0,
        dialect_id: Optional[str] = None,
        performance_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> TTSResponse:
        """
        Generate speech from text using Lahajati TTS API.

        Args:
            text: Text to convert to speech (Arabic text recommended)
            voice: Voice ID to use (e.g., "ac_voice_xyz789")
            input_mode: 0 for structured mode (use dialect_id/performance_id),
                       1 for custom mode (use custom_prompt)
            dialect_id: Dialect ID for structured mode (Egyptian, Saudi Najdi, etc.)
            performance_id: Performance style ID (dramatic, news, advertising, etc.)
            custom_prompt: Custom instructions for custom mode
            output_path: Optional file path to save audio

        Note:
            model, speed, language_code, instructions, stability, similarity_boost,
            style, seed are ignored as they are not supported by Lahajati TTS.

        Returns:
            TTSResponse with audio data and metadata
        """
        # Get default voice
        voice = self._get_voice(voice)
        if not voice:
            raise TTSValidationError("Voice ID is required for Lahajati TTS")

        # Validate
        self._validate_params(text, input_mode, custom_prompt, dialect_id, performance_id)

        start_time = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                # Build request body
                # Note: input_mode must be sent as string per API spec
                request_body: Dict[str, Any] = {
                    "text": text,
                    "id_voice": voice,
                    "input_mode": str(input_mode),
                }

                # Add mode-specific parameters
                if input_mode == 0:  # Structured mode - both are required
                    request_body["dialect_id"] = str(dialect_id)
                    request_body["performance_id"] = str(performance_id)
                else:  # Custom mode
                    request_body["custom_prompt_text"] = custom_prompt

                # Make request
                response = self.client.post(
                    "/text-to-speech-absolute-control",
                    json=request_body,
                )

                # Check for errors
                if response.status_code == 401:
                    raise TTSProviderError(
                        "Unauthorized: Invalid API key",
                        provider=self.provider.value,
                        status_code=401,
                    )
                elif response.status_code == 422:
                    raise TTSValidationError(
                        f"Validation error: {response.text}"
                    )
                elif response.status_code != 200:
                    raise TTSProviderError(
                        f"API error: {response.status_code} - {response.text}",
                        provider=self.provider.value,
                        status_code=response.status_code,
                    )

                # Get audio data
                audio_data = response.content

                process_time = time.time() - start_time

                # Save to file if requested
                file_path = None
                if output_path:
                    file_path = self._save_audio(audio_data, output_path)

                return TTSResponse(
                    audio_data=audio_data if not output_path else file_path,
                    model="lahajati-tts",
                    voice=voice,
                    format="mp3",
                    duration_ms=None,
                    file_path=file_path,
                    provider=self.provider.value,
                    process_time=process_time,
                    dialect_id=dialect_id,
                    performance_id=performance_id,
                    custom_prompt=custom_prompt if input_mode == 1 else None,
                    input_mode=input_mode,
                )

            except (TTSValidationError, TTSProviderError):
                raise
            except httpx.TimeoutException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise TTSProviderError(
                        f"Request timed out after {MAX_RETRIES} attempts",
                        provider=self.provider.value,
                    )
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
        # Lahajati-specific parameters
        input_mode: int = 0,
        dialect_id: Optional[str] = None,
        performance_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> TTSResponse:
        """
        Async version: Generate speech from text using Lahajati TTS API.

        Args:
            Same as generate_speech()

        Returns:
            TTSResponse with audio data and metadata
        """
        # Get default voice
        voice = self._get_voice(voice)
        if not voice:
            raise TTSValidationError("Voice ID is required for Lahajati TTS")

        # Validate
        self._validate_params(text, input_mode, custom_prompt, dialect_id, performance_id)

        start_time = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                # Build request body
                # Note: input_mode must be sent as string per API spec
                request_body: Dict[str, Any] = {
                    "text": text,
                    "id_voice": voice,
                    "input_mode": str(input_mode),
                }

                # Add mode-specific parameters
                if input_mode == 0:  # Structured mode - both are required
                    request_body["dialect_id"] = str(dialect_id)
                    request_body["performance_id"] = str(performance_id)
                else:  # Custom mode
                    request_body["custom_prompt_text"] = custom_prompt

                # Make request
                response = await self.async_client.post(
                    "/text-to-speech-absolute-control",
                    json=request_body,
                )

                # Check for errors
                if response.status_code == 401:
                    raise TTSProviderError(
                        "Unauthorized: Invalid API key",
                        provider=self.provider.value,
                        status_code=401,
                    )
                elif response.status_code == 422:
                    raise TTSValidationError(
                        f"Validation error: {response.text}"
                    )
                elif response.status_code != 200:
                    raise TTSProviderError(
                        f"API error: {response.status_code} - {response.text}",
                        provider=self.provider.value,
                        status_code=response.status_code,
                    )

                # Get audio data
                audio_data = response.content

                process_time = time.time() - start_time

                # Save to file if requested
                file_path = None
                if output_path:
                    file_path = self._save_audio(audio_data, output_path)

                return TTSResponse(
                    audio_data=audio_data if not output_path else file_path,
                    model="lahajati-tts",
                    voice=voice,
                    format="mp3",
                    duration_ms=None,
                    file_path=file_path,
                    provider=self.provider.value,
                    process_time=process_time,
                    dialect_id=dialect_id,
                    performance_id=performance_id,
                    custom_prompt=custom_prompt if input_mode == 1 else None,
                    input_mode=input_mode,
                )

            except (TTSValidationError, TTSProviderError):
                raise
            except httpx.TimeoutException:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise TTSProviderError(
                        f"Request timed out after {MAX_RETRIES} attempts",
                        provider=self.provider.value,
                    )
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
        List all available Lahajati voices.

        Args:
            use_cache: Whether to use cached voices (default: True)

        Returns:
            List of Voice objects
        """
        if use_cache and self._voices_cache is not None:
            return self._voices_cache

        try:
            response = self.client.get("/voices-absolute-control")

            if response.status_code != 200:
                raise TTSProviderError(
                    f"Failed to list voices: {response.status_code} - {response.text}",
                    provider=self.provider.value,
                    status_code=response.status_code,
                )

            data = response.json()
            voices_data = data.get("data", data) if isinstance(data, dict) else data

            voices = []
            for voice_data in voices_data:
                voice = Voice(
                    voice_id=voice_data.get("id_voice", voice_data.get("voice_id", "")),
                    name=voice_data.get("name", voice_data.get("display_name", "")),
                    language="ar",
                    languages=["ar"],
                    gender=voice_data.get("gender"),
                    preview_url=voice_data.get("preview_url"),
                    provider=self.provider.value,
                    description=voice_data.get("description"),
                    is_cloned=voice_data.get("is_cloned", False),
                    average_rating=voice_data.get("average_rating"),
                )
                voices.append(voice)

            self._voices_cache = voices
            return voices

        except TTSProviderError:
            raise
        except Exception as e:
            raise TTSProviderError(
                f"Failed to list voices: {e}",
                provider=self.provider.value,
            )

    def get_voice(self, voice_id: str) -> Voice:
        """
        Get details for a specific Lahajati voice.

        Args:
            voice_id: Voice ID to look up

        Returns:
            Voice object with details

        Raises:
            TTSVoiceNotFoundError: If voice is not found
        """
        voices = self.list_voices()
        for voice in voices:
            if voice.voice_id == voice_id:
                return voice

        raise TTSVoiceNotFoundError(voice_id, self.provider.value)

    def list_dialects(self, use_cache: bool = True) -> List[Dialect]:
        """
        List all available Arabic dialects.

        Args:
            use_cache: Whether to use cached dialects (default: True)

        Returns:
            List of Dialect objects
        """
        if use_cache and self._dialects_cache is not None:
            return self._dialects_cache

        try:
            response = self.client.get("/dialect-absolute-control")

            if response.status_code != 200:
                raise TTSProviderError(
                    f"Failed to list dialects: {response.status_code} - {response.text}",
                    provider=self.provider.value,
                    status_code=response.status_code,
                )

            data = response.json()
            dialects_data = data.get("data", data) if isinstance(data, dict) else data

            dialects = []
            for dialect_data in dialects_data:
                dialect = Dialect(
                    dialect_id=dialect_data.get("dialect_id", ""),
                    display_name=dialect_data.get("display_name", ""),
                    description=dialect_data.get("description"),
                )
                dialects.append(dialect)

            self._dialects_cache = dialects
            return dialects

        except TTSProviderError:
            raise
        except Exception as e:
            raise TTSProviderError(
                f"Failed to list dialects: {e}",
                provider=self.provider.value,
            )

    def get_dialect(self, dialect_id: str) -> Dialect:
        """
        Get details for a specific dialect.

        Args:
            dialect_id: Dialect ID to look up

        Returns:
            Dialect object with details

        Raises:
            TTSProviderError: If dialect is not found
        """
        dialects = self.list_dialects()
        for dialect in dialects:
            if dialect.dialect_id == dialect_id:
                return dialect

        raise TTSProviderError(
            f"Dialect '{dialect_id}' not found",
            provider=self.provider.value,
        )

    def list_performances(self, use_cache: bool = True) -> List[Performance]:
        """
        List all available performance styles.

        Args:
            use_cache: Whether to use cached performances (default: True)

        Returns:
            List of Performance objects
        """
        if use_cache and self._performances_cache is not None:
            return self._performances_cache

        try:
            response = self.client.get("/performance-absolute-control")

            if response.status_code != 200:
                raise TTSProviderError(
                    f"Failed to list performances: {response.status_code} - {response.text}",
                    provider=self.provider.value,
                    status_code=response.status_code,
                )

            data = response.json()
            performances_data = data.get("data", data) if isinstance(data, dict) else data

            performances = []
            for perf_data in performances_data:
                performance = Performance(
                    performance_id=perf_data.get("performance_id", ""),
                    display_name=perf_data.get("display_name", ""),
                    description=perf_data.get("description"),
                )
                performances.append(performance)

            self._performances_cache = performances
            return performances

        except TTSProviderError:
            raise
        except Exception as e:
            raise TTSProviderError(
                f"Failed to list performances: {e}",
                provider=self.provider.value,
            )

    def get_performance(self, performance_id: str) -> Performance:
        """
        Get details for a specific performance style.

        Args:
            performance_id: Performance ID to look up

        Returns:
            Performance object with details

        Raises:
            TTSProviderError: If performance is not found
        """
        performances = self.list_performances()
        for perf in performances:
            if perf.performance_id == performance_id:
                return perf

        raise TTSProviderError(
            f"Performance '{performance_id}' not found",
            provider=self.provider.value,
        )

    def clear_cache(self) -> None:
        """Clear all cached data (voices, dialects, performances)"""
        self._voices_cache = None
        self._dialects_cache = None
        self._performances_cache = None

    def close(self) -> None:
        """Close HTTP clients"""
        if self._client:
            self._client.close()
            self._client = None
        if self._async_client:
            asyncio.get_event_loop().run_until_complete(self._async_client.aclose())
            self._async_client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
