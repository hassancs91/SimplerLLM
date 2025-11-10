"""
Video transcription and multi-language caption generation.

This module provides functionality for:
- Transcribing videos (local files or YouTube URLs) to text with timing
- Generating multi-language captions using LLM translation
- Exporting captions in SRT and VTT formats
"""

from .base import VideoTranscriber
from .caption_generator import MultiLanguageCaptionGenerator
from .models import (
    VideoTranscriptionResult,
    CaptionSegment,
    LanguageCaptions,
    MultiLanguageCaptionsResult
)

__all__ = [
    'VideoTranscriber',
    'MultiLanguageCaptionGenerator',
    'VideoTranscriptionResult',
    'CaptionSegment',
    'LanguageCaptions',
    'MultiLanguageCaptionsResult'
]
