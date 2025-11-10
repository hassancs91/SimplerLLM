"""
Video transcription functionality using existing STT capabilities.
"""
import os
import time
from typing import Optional, List
from .models import VideoTranscriptionResult, CaptionSegment
from .utils.video_utils import (
    is_youtube_url,
    extract_audio_from_video,
    get_video_duration,
    cleanup_temp_file
)
from SimplerLLM.utils.custom_verbose import verbose_print


class VideoTranscriber:
    """
    Transcribe videos to text with timing information for caption generation.

    This class provides video transcription capabilities by:
    1. Extracting audio from video files (MP4, AVI, MOV)
    2. Supporting YouTube URLs
    3. Using STT for transcription with timing
    4. Generating caption segments for subtitle creation
    """

    def __init__(
        self,
        stt_instance,
        verbose: bool = False
    ):
        """
        Initialize VideoTranscriber.

        Args:
            stt_instance: An instance of STT (e.g., from STT.create())
            verbose: Enable verbose logging
        """
        self.stt = stt_instance
        self.verbose = verbose

        if self.verbose:
            verbose_print("VideoTranscriber initialized", "info")

    def transcribe_video(
        self,
        video_source: str,
        language: Optional[str] = None,
        output_format: str = "srt",
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> VideoTranscriptionResult:
        """
        Transcribe a video to text with timing information.

        Args:
            video_source: Path to video file or YouTube URL
            language: Language code (e.g., "en", "es") - auto-detect if None
            output_format: Subtitle format - "srt" or "vtt"
            prompt: Optional text to guide transcription style
            temperature: Sampling temperature (0.0 - 1.0)

        Returns:
            VideoTranscriptionResult with transcription and caption segments

        Raises:
            ValueError: If invalid parameters
            FileNotFoundError: If video file not found
            Exception: If transcription fails
        """
        start_time = time.time()

        # Validate format
        if output_format not in ["srt", "vtt"]:
            raise ValueError(f"output_format must be 'srt' or 'vtt', got: {output_format}")

        # Determine source type
        is_youtube = is_youtube_url(video_source)
        source_type = "youtube" if is_youtube else "local_file"

        if self.verbose:
            verbose_print(
                f"Transcribing {source_type}: {video_source}",
                "info"
            )

        # Handle YouTube URLs
        if is_youtube:
            return self._transcribe_youtube(
                video_source,
                language,
                output_format,
                prompt,
                temperature
            )

        # Handle local video files
        return self._transcribe_local_video(
            video_source,
            language,
            output_format,
            prompt,
            temperature,
            start_time
        )

    def _transcribe_local_video(
        self,
        video_path: str,
        language: Optional[str],
        output_format: str,
        prompt: Optional[str],
        temperature: float,
        start_time: float
    ) -> VideoTranscriptionResult:
        """Transcribe a local video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        temp_audio_file = None

        try:
            # Get video duration
            if self.verbose:
                verbose_print("Getting video duration...", "debug")
            duration = get_video_duration(video_path)

            # Extract audio from video
            if self.verbose:
                verbose_print("Extracting audio from video...", "info")

            temp_audio_file = extract_audio_from_video(
                video_path,
                verbose=self.verbose
            )

            # Transcribe with verbose_json to get segments
            if self.verbose:
                verbose_print("Transcribing audio with STT...", "info")

            stt_response = self.stt.transcribe(
                audio_file=temp_audio_file,
                language=language,
                prompt=prompt,
                response_format="verbose_json",
                temperature=temperature,
                full_response=True
            )

            # Parse segments from response
            segments = self._parse_segments_from_response(stt_response)

            # Calculate total process time
            process_time = time.time() - start_time

            # Create result
            result = VideoTranscriptionResult(
                text=stt_response.text,
                language=stt_response.language,
                segments=segments,
                duration=duration,
                process_time=process_time,
                source_type="local_file",
                source_path=video_path,
                model=stt_response.model,
                provider=stt_response.provider
            )

            if self.verbose:
                verbose_print(
                    f"Transcription completed - {len(segments)} segments, {process_time:.2f}s",
                    "info"
                )

            return result

        finally:
            # Clean up temporary audio file
            if temp_audio_file:
                cleanup_temp_file(temp_audio_file, verbose=self.verbose)

    def _transcribe_youtube(
        self,
        youtube_url: str,
        language: Optional[str],
        output_format: str,
        prompt: Optional[str],
        temperature: float
    ) -> VideoTranscriptionResult:
        """Transcribe a YouTube video."""
        # For YouTube, we can try to use existing transcript first,
        # or download and transcribe the audio
        # For now, we'll use the download approach for consistency

        try:
            from .utils.video_utils import download_youtube_audio
        except ImportError:
            raise ImportError(
                "yt-dlp is required for YouTube support. "
                "Install it with: pip install yt-dlp"
            )

        start_time = time.time()
        temp_audio_file = None

        try:
            # Download YouTube audio
            if self.verbose:
                verbose_print("Downloading YouTube audio...", "info")

            temp_audio_file, duration = download_youtube_audio(
                youtube_url,
                verbose=self.verbose
            )

            # Transcribe with verbose_json to get segments
            if self.verbose:
                verbose_print("Transcribing audio with STT...", "info")

            stt_response = self.stt.transcribe(
                audio_file=temp_audio_file,
                language=language,
                prompt=prompt,
                response_format="verbose_json",
                temperature=temperature,
                full_response=True
            )

            # Parse segments from response
            segments = self._parse_segments_from_response(stt_response)

            # Calculate total process time
            process_time = time.time() - start_time

            # Create result
            result = VideoTranscriptionResult(
                text=stt_response.text,
                language=stt_response.language,
                segments=segments,
                duration=duration,
                process_time=process_time,
                source_type="youtube",
                source_path=youtube_url,
                model=stt_response.model,
                provider=stt_response.provider
            )

            if self.verbose:
                verbose_print(
                    f"YouTube transcription completed - {len(segments)} segments, {process_time:.2f}s",
                    "info"
                )

            return result

        finally:
            # Clean up temporary audio file
            if temp_audio_file:
                cleanup_temp_file(temp_audio_file, verbose=self.verbose)

    def _parse_segments_from_response(self, stt_response) -> List[CaptionSegment]:
        """
        Parse timing segments from STT response.

        Args:
            stt_response: STTFullResponse object

        Returns:
            List of CaptionSegment objects
        """
        segments = []

        # Check if we have the raw response with segments
        if hasattr(stt_response, 'llm_provider_response'):
            raw_response = stt_response.llm_provider_response

            # OpenAI verbose_json response has segments attribute
            if hasattr(raw_response, 'segments'):
                for idx, segment in enumerate(raw_response.segments, start=1):
                    caption_segment = CaptionSegment(
                        index=idx,
                        start_time=segment.get('start', 0) if isinstance(segment, dict) else segment.start,
                        end_time=segment.get('end', 0) if isinstance(segment, dict) else segment.end,
                        text=segment.get('text', '').strip() if isinstance(segment, dict) else segment.text.strip(),
                        duration=(segment.get('end', 0) - segment.get('start', 0)) if isinstance(segment, dict) else (segment.end - segment.start)
                    )
                    segments.append(caption_segment)

        # Fallback: create a single segment if no timing info available
        if not segments and stt_response.text:
            segments = [
                CaptionSegment(
                    index=1,
                    start_time=0.0,
                    end_time=stt_response.duration or 0.0,
                    text=stt_response.text,
                    duration=stt_response.duration or 0.0
                )
            ]

        return segments

    def save_transcription(
        self,
        result: VideoTranscriptionResult,
        output_path: str,
        format: str = "srt"
    ) -> None:
        """
        Save transcription to a subtitle file.

        Args:
            result: VideoTranscriptionResult object
            output_path: Path to save the subtitle file
            format: Output format - "srt" or "vtt"
        """
        if format == "srt":
            content = result.to_srt()
        elif format == "vtt":
            content = result.to_vtt()
        else:
            raise ValueError(f"format must be 'srt' or 'vtt', got: {format}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        if self.verbose:
            verbose_print(f"Transcription saved to: {output_path}", "info")
