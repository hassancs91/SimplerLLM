"""
TTS Models and Exceptions

This module contains all data models and exceptions for the TTS module.
"""

from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Union
from enum import Enum


class TTSProvider(Enum):
    """Supported TTS providers"""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    LAHAJATI = "lahajati"


class Voice(BaseModel):
    """Unified voice model for all providers"""
    model_config = ConfigDict(extra="allow")

    voice_id: str
    name: str
    language: Optional[str] = None
    languages: List[str] = []
    gender: Optional[str] = None
    preview_url: Optional[str] = None
    provider: str
    description: Optional[str] = None


class TTSResponse(BaseModel):
    """Unified response model for TTS generation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_data: Union[bytes, str]  # bytes or file path
    model: str
    voice: str
    format: str
    duration_ms: Optional[int] = None
    file_path: Optional[str] = None
    provider: str
    process_time: Optional[float] = None  # Time in seconds to generate audio

    # Provider-specific metadata
    instructions: Optional[str] = None  # OpenAI gpt-4o-mini-tts
    language_code: Optional[str] = None  # ElevenLabs
    seed: Optional[int] = None  # ElevenLabs

    # Voice settings used (ElevenLabs)
    stability: Optional[float] = None
    similarity_boost: Optional[float] = None
    style: Optional[float] = None

    # Lahajati-specific metadata
    dialect_id: Optional[Union[str, int]] = None
    performance_id: Optional[Union[str, int]] = None
    custom_prompt: Optional[str] = None
    input_mode: Optional[int] = None  # 0=structured, 1=custom


# Exceptions

class TTSError(Exception):
    """Base exception for TTS errors"""
    pass


class TTSValidationError(TTSError):
    """Raised when input parameters are invalid"""
    pass


class TTSProviderError(TTSError):
    """Raised when provider API returns an error"""
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class TTSVoiceNotFoundError(TTSError):
    """Raised when a voice is not found"""
    def __init__(self, voice_id: str, provider: str):
        super().__init__(f"Voice '{voice_id}' not found for provider '{provider}'")
        self.voice_id = voice_id
        self.provider = provider


# Constants

OPENAI_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "onyx", "nova", "sage", "shimmer", "verse", "marin", "cedar"
]

OPENAI_MODELS = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]

OPENAI_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

ELEVENLABS_MODELS = [
    "eleven_flash_v2_5",
    "eleven_turbo_v2_5",
    "eleven_turbo_v2",
    "eleven_multilingual_v2",
    "eleven_monolingual_v1"
]

ELEVENLABS_FORMATS = [
    "mp3_44100_128",
    "mp3_44100_64",
    "mp3_44100_32",
    "mp3_22050_32",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_44100",
    "ulaw_8000",
    "opus_48000_128",
    "opus_48000_64"
]

# Lahajati constants
LAHAJATI_FORMATS = ["mp3"]  # API returns audio/mpeg

LAHAJATI_INPUT_MODES = {
    "structured": 0,  # Uses performance_id and dialect_id
    "custom": 1,      # Uses custom_prompt_text
}


class Dialect(BaseModel):
    """Lahajati dialect model"""
    model_config = ConfigDict(extra="allow")

    dialect_id: Union[str, int]  # API may return int
    display_name: str
    description: Optional[str] = None


class Performance(BaseModel):
    """Lahajati performance style model"""
    model_config = ConfigDict(extra="allow")

    performance_id: Union[str, int]  # API may return int
    display_name: str
    description: Optional[str] = None
