"""
Video dubbing functionality for creating dubbed videos in different languages.

This module provides functionality for:
- Transcribing videos and translating to target languages
- Generating TTS audio for translated segments
- Adjusting audio timing to match original video
- Replacing video audio tracks with dubbed audio
"""

from .base import VideoDubber
from .models import DubbedSegment, DubbingConfig, VideoDubbingResult
from .audio_sync import (
    adjust_audio_speed,
    adjust_audio_speed_advanced,
    get_audio_duration,
    merge_audio_segments
)
from .video_processor import (
    replace_video_audio,
    get_video_info,
    trim_video
)

__all__ = [
    'VideoDubber',
    'DubbedSegment',
    'DubbingConfig',
    'VideoDubbingResult',
    'adjust_audio_speed',
    'adjust_audio_speed_advanced',
    'get_audio_duration',
    'merge_audio_segments',
    'replace_video_audio',
    'get_video_info',
    'trim_video'
]
