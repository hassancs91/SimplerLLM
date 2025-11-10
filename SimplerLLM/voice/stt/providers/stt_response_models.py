from pydantic import BaseModel
from typing import Any, Optional


class STTFullResponse(BaseModel):
    """Full response from STT transcription with metadata."""

    text: str
    """The transcribed text"""

    model: str
    """The STT model used (e.g., 'whisper-1')"""

    language: Optional[str] = None
    """Detected or specified language code (e.g., 'en', 'es', 'fr')"""

    process_time: float
    """Time taken to transcribe the audio in seconds"""

    provider: Optional[str] = None
    """The STT provider used (e.g., 'OPENAI')"""

    audio_file: Optional[str] = None
    """Path to the audio file that was transcribed"""

    duration: Optional[float] = None
    """Duration of the audio file in seconds"""

    response_format: Optional[str] = None
    """Response format used (e.g., 'text', 'json', 'verbose_json')"""

    llm_provider_response: Optional[Any] = None
    """Raw response from the provider API"""

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a transcription of the audio file.",
                "model": "whisper-1",
                "language": "en",
                "process_time": 2.45,
                "provider": "OPENAI",
                "audio_file": "recording.mp3",
                "duration": 10.5,
                "response_format": "text"
            }
        }
