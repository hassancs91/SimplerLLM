"""
Pydantic models for video dubbing.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class DubbedSegment(BaseModel):
    """Represents a single dubbed audio segment with timing information."""

    index: int
    start_time: float  # Original start time in seconds
    end_time: float    # Original end time in seconds
    original_text: str
    translated_text: str
    audio_file: Optional[str] = None  # Path to generated audio segment
    original_duration: float
    dubbed_duration: Optional[float] = None  # Duration of generated audio
    speed_adjustment: Optional[float] = None  # Speed factor applied (e.g., 1.2 = 20% faster)


class DubbingConfig(BaseModel):
    """Configuration for video dubbing."""

    target_language: str
    match_timing: bool = True  # Whether to adjust speech speed to match original timing
    voice: Optional[str] = None  # Voice ID or name (provider-specific)
    speed_range: tuple = (0.75, 1.5)  # Min and max speed adjustment factors
    audio_format: str = "mp3"
    sample_rate: int = 44100


class VideoDubbingResult(BaseModel):
    """Result of video dubbing operation."""

    original_video_path: str
    output_video_path: str
    target_language: str
    source_language: Optional[str] = None
    segments: List[DubbedSegment]
    total_segments: int
    duration: float  # Total video duration
    process_time: float
    tts_provider: Optional[str] = None
    tts_model: Optional[str] = None
    average_speed_adjustment: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "original_video_path": "video.mp4",
                "output_video_path": "video_dubbed_es.mp4",
                "target_language": "Spanish",
                "source_language": "English",
                "total_segments": 15,
                "duration": 120.5,
                "process_time": 45.2,
                "tts_provider": "ELEVENLABS",
                "average_speed_adjustment": 1.05
            }
        }
