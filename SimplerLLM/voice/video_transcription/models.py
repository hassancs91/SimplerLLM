"""
Pydantic models for video transcription and caption generation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import timedelta


class CaptionSegment(BaseModel):
    """Represents a single caption segment with timing information."""

    index: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str
    duration: float    # in seconds

    def to_srt_time(self, time_seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        td = timedelta(seconds=time_seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = int(td.total_seconds() % 60)
        milliseconds = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def to_vtt_time(self, time_seconds: float) -> str:
        """Convert seconds to VTT time format (HH:MM:SS.mmm)."""
        td = timedelta(seconds=time_seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = int(td.total_seconds() % 60)
        milliseconds = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def to_srt(self) -> str:
        """Convert segment to SRT format."""
        start = self.to_srt_time(self.start_time)
        end = self.to_srt_time(self.end_time)
        return f"{self.index}\n{start} --> {end}\n{self.text}\n"

    def to_vtt(self) -> str:
        """Convert segment to VTT format."""
        start = self.to_vtt_time(self.start_time)
        end = self.to_vtt_time(self.end_time)
        return f"{start} --> {end}\n{self.text}\n"


class VideoTranscriptionResult(BaseModel):
    """Result of video transcription with timing information."""

    text: str
    language: Optional[str] = None
    segments: List[CaptionSegment]
    duration: float  # Total video duration in seconds
    process_time: float
    source_type: str  # 'local_file' or 'youtube'
    source_path: str  # File path or YouTube URL
    model: Optional[str] = None
    provider: Optional[str] = None

    def to_srt(self) -> str:
        """Convert all segments to SRT format."""
        srt_content = "\n".join(segment.to_srt() for segment in self.segments)
        return srt_content

    def to_vtt(self) -> str:
        """Convert all segments to VTT format."""
        vtt_content = "WEBVTT\n\n"
        vtt_content += "\n".join(segment.to_vtt() for segment in self.segments)
        return vtt_content


class LanguageCaptions(BaseModel):
    """Captions in a specific language."""

    language: str
    language_code: str  # e.g., 'es', 'fr', 'de'
    segments: List[CaptionSegment]
    format: str  # 'srt' or 'vtt'
    content: str  # Formatted caption content

    def save_to_file(self, file_path: str):
        """Save captions to a file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.content)


class MultiLanguageCaptionsResult(BaseModel):
    """Result containing captions in multiple languages."""

    original_language: str
    original_transcription: VideoTranscriptionResult
    captions: Dict[str, LanguageCaptions]  # language_code -> LanguageCaptions
    target_languages: List[str]
    process_time: float
    total_segments: int

    def get_captions(self, language_code: str) -> Optional[LanguageCaptions]:
        """Get captions for a specific language."""
        return self.captions.get(language_code)

    def save_all(self, output_dir: str, base_filename: str):
        """Save all captions to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for lang_code, caption in self.captions.items():
            ext = 'srt' if caption.format == 'srt' else 'vtt'
            filename = f"{base_filename}.{lang_code}.{ext}"
            filepath = os.path.join(output_dir, filename)
            caption.save_to_file(filepath)
