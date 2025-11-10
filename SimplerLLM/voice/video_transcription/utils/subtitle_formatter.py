"""
Subtitle formatting utilities for SRT and VTT formats.
"""
from typing import List
from ..models import CaptionSegment


def format_segments_to_srt(segments: List[CaptionSegment]) -> str:
    """
    Format caption segments to SRT format.

    Args:
        segments: List of CaptionSegment objects

    Returns:
        SRT formatted string
    """
    srt_content = []

    for segment in segments:
        srt_content.append(segment.to_srt())

    return "\n".join(srt_content)


def format_segments_to_vtt(segments: List[CaptionSegment]) -> str:
    """
    Format caption segments to VTT (WebVTT) format.

    Args:
        segments: List of CaptionSegment objects

    Returns:
        VTT formatted string
    """
    vtt_content = ["WEBVTT\n"]

    for segment in segments:
        vtt_content.append(segment.to_vtt())

    return "\n".join(vtt_content)


def save_subtitles(
    content: str,
    output_path: str,
    encoding: str = 'utf-8'
) -> None:
    """
    Save subtitle content to a file.

    Args:
        content: The formatted subtitle content (SRT or VTT)
        output_path: Path where to save the file
        encoding: File encoding (default: utf-8)
    """
    with open(output_path, 'w', encoding=encoding) as f:
        f.write(content)
