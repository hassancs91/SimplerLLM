from pydantic import BaseModel
from typing import Any, Optional, Union


class TTSFullResponse(BaseModel):
    """Full response from TTS generation with metadata."""

    audio_data: Union[bytes, str]
    """The generated audio - either bytes or file path"""

    model: str
    """The TTS model used (e.g., 'tts-1', 'tts-1-hd')"""

    voice: str
    """The voice used (e.g., 'alloy', 'nova', 'shimmer')"""

    format: str
    """The audio format (e.g., 'mp3', 'wav', 'opus')"""

    process_time: float
    """Time taken to generate the audio in seconds"""

    speed: Optional[float] = 1.0
    """The speech speed used (0.25 to 4.0). None if provider doesn't support speed control."""

    file_size: Optional[int] = None
    """Size of the audio data in bytes"""

    output_path: Optional[str] = None
    """File path if audio was saved to disk"""

    provider: Optional[str] = None
    """The TTS provider used (e.g., 'OPENAI')"""

    llm_provider_response: Optional[Any] = None
    """Raw response from the provider API"""

    class Config:
        json_schema_extra = {
            "example": {
                "audio_data": "output/speech.mp3",
                "model": "tts-1-hd",
                "voice": "alloy",
                "format": "mp3",
                "process_time": 1.23,
                "speed": 1.0,
                "file_size": 24576,
                "output_path": "output/speech.mp3",
                "provider": "OPENAI"
            }
        }
