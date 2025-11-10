"""
Video dubbing functionality for creating dubbed videos in different languages.
"""
import os
import time
import tempfile
from typing import Optional, List
from pathlib import Path

from .models import DubbedSegment, DubbingConfig, VideoDubbingResult
from .audio_sync import (
    adjust_audio_speed,
    adjust_audio_speed_advanced,
    get_audio_duration,
    merge_audio_segments
)
from .video_processor import replace_video_audio
from SimplerLLM.voice.video_transcription import VideoTranscriber
from SimplerLLM.utils.custom_verbose import verbose_print


class VideoDubber:
    """
    Create dubbed videos by replacing audio with translated speech.

    This class provides video dubbing capabilities by:
    1. Transcribing the original video
    2. Translating to target language
    3. Generating TTS audio for each segment
    4. Adjusting timing to match original
    5. Replacing the video's audio track
    """

    def __init__(
        self,
        tts_instance,
        llm_instance,
        stt_instance,
        verbose: bool = False
    ):
        """
        Initialize VideoDubber.

        Args:
            tts_instance: An instance of TTS (e.g., from TTS.create())
            llm_instance: An instance of LLM (e.g., from LLM.create())
            stt_instance: An instance of STT (e.g., from STT.create())
            verbose: Enable verbose logging
        """
        self.tts = tts_instance
        self.llm = llm_instance
        self.stt = stt_instance
        self.transcriber = VideoTranscriber(stt_instance, verbose=verbose)
        self.verbose = verbose

        if self.verbose:
            verbose_print("VideoDubber initialized", "info")

    def dub_video(
        self,
        video_path: str,
        target_language: str,
        output_path: str,
        match_timing: bool = True,
        speed_range: tuple = (0.75, 1.5),
        preserve_pitch: bool = False,
        source_language: Optional[str] = None,
        cleanup_temp_files: bool = True
    ) -> VideoDubbingResult:
        """
        Dub a video to a target language.

        Args:
            video_path: Path to the video file
            target_language: Target language code (e.g., "es", "fr", "de")
            output_path: Path to save the dubbed video
            match_timing: Whether to adjust speech speed to match original timing
            speed_range: Min and max speed adjustment factors (min, max)
            preserve_pitch: Whether to preserve pitch when adjusting speed (requires ffmpeg)
            source_language: Source language code (auto-detect if None)
            cleanup_temp_files: Whether to clean up temporary audio files

        Returns:
            VideoDubbingResult with dubbing information

        Example:
            >>> dubber = VideoDubber(tts, llm, stt)
            >>> result = dubber.dub_video(
            ...     "video.mp4",
            ...     target_language="es",
            ...     output_path="video_es.mp4"
            ... )
        """
        start_time = time.time()
        temp_audio_files = []

        try:
            # Step 1: Transcribe the video
            if self.verbose:
                verbose_print("Step 1: Transcribing video...", "info")

            transcription = self.transcriber.transcribe_video(
                video_path,
                language=source_language
            )

            if self.verbose:
                verbose_print(
                    f"Transcription complete: {len(transcription.segments)} segments",
                    "info"
                )

            # Step 2: Translate and generate TTS for each segment
            if self.verbose:
                verbose_print(
                    f"Step 2: Generating dubbed audio for {len(transcription.segments)} segments...",
                    "info"
                )

            dubbed_segments = []
            segment_audio_files = []

            for idx, segment in enumerate(transcription.segments, 1):
                if self.verbose:
                    verbose_print(
                        f"Processing segment {idx}/{len(transcription.segments)}",
                        "debug"
                    )

                # Translate segment
                translated_text = self._translate_segment(
                    segment.text,
                    target_language,
                    transcription.language
                )

                # Generate TTS audio
                audio_path = self._generate_segment_audio(
                    translated_text,
                    segment.index
                )
                temp_audio_files.append(audio_path)

                # Get duration of generated audio
                dubbed_duration = get_audio_duration(audio_path)

                # Adjust speed if needed
                speed_adjustment = 1.0
                if match_timing and abs(dubbed_duration - segment.duration) > 0.1:
                    if self.verbose:
                        verbose_print(
                            f"Adjusting segment {idx} speed: {dubbed_duration:.2f}s -> {segment.duration:.2f}s",
                            "debug"
                        )

                    if preserve_pitch:
                        adjusted_path, speed_adjustment = adjust_audio_speed_advanced(
                            audio_path,
                            segment.duration,
                            speed_range=speed_range,
                            preserve_pitch=True,
                            verbose=self.verbose
                        )
                    else:
                        adjusted_path, speed_adjustment = adjust_audio_speed(
                            audio_path,
                            segment.duration,
                            speed_range=speed_range,
                            verbose=self.verbose
                        )

                    temp_audio_files.append(adjusted_path)
                    segment_audio_files.append(adjusted_path)
                else:
                    segment_audio_files.append(audio_path)

                # Create dubbed segment record
                dubbed_segment = DubbedSegment(
                    index=segment.index,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    original_text=segment.text,
                    translated_text=translated_text,
                    audio_file=audio_path,
                    original_duration=segment.duration,
                    dubbed_duration=dubbed_duration,
                    speed_adjustment=speed_adjustment if speed_adjustment != 1.0 else None
                )
                dubbed_segments.append(dubbed_segment)

            # Step 3: Merge all audio segments
            if self.verbose:
                verbose_print("Step 3: Merging audio segments...", "info")

            merged_audio_path = self._merge_dubbed_audio(
                segment_audio_files,
                dubbed_segments
            )
            temp_audio_files.append(merged_audio_path)

            # Step 4: Replace video audio
            if self.verbose:
                verbose_print("Step 4: Replacing video audio...", "info")

            replace_video_audio(
                video_path,
                merged_audio_path,
                output_path,
                verbose=self.verbose
            )

            # Calculate statistics
            process_time = time.time() - start_time
            avg_speed = sum(
                s.speed_adjustment for s in dubbed_segments if s.speed_adjustment
            ) / max(1, sum(1 for s in dubbed_segments if s.speed_adjustment))

            # Get TTS provider info
            tts_provider = getattr(self.tts, 'provider', None)
            tts_model = getattr(self.tts, 'model_name', None)

            # Create result
            result = VideoDubbingResult(
                original_video_path=video_path,
                output_video_path=output_path,
                target_language=self._get_language_name(target_language),
                source_language=self._get_language_name(transcription.language) if transcription.language else None,
                segments=dubbed_segments,
                total_segments=len(dubbed_segments),
                duration=transcription.duration,
                process_time=process_time,
                tts_provider=str(tts_provider.name) if tts_provider else None,
                tts_model=tts_model,
                average_speed_adjustment=avg_speed if avg_speed != 1.0 else None
            )

            if self.verbose:
                verbose_print(
                    f"Dubbing complete! Processed {len(dubbed_segments)} segments in {process_time:.2f}s",
                    "info"
                )
                verbose_print(f"Output saved to: {output_path}", "info")

            return result

        finally:
            # Clean up temporary files
            if cleanup_temp_files:
                self._cleanup_temp_files(temp_audio_files)

    def _translate_segment(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> str:
        """Translate a text segment using LLM."""
        lang_name = self._get_language_name(target_language)

        if source_language:
            source_name = self._get_language_name(source_language)
            prompt = f"Translate the following text from {source_name} to {lang_name}. Only return the translated text, no explanations:\n\n{text}"
        else:
            prompt = f"Translate the following text to {lang_name}. Only return the translated text, no explanations:\n\n{text}"

        try:
            response = self.llm.generate_text(
                prompt=prompt,
                temperature=0.3
            )

            # Extract text from response
            if isinstance(response, str):
                translated = response.strip()
            elif hasattr(response, 'generated_text'):
                translated = response.generated_text.strip()
            else:
                translated = str(response).strip()

            return translated

        except Exception as e:
            if self.verbose:
                verbose_print(f"Translation failed: {str(e)}", "error")
            return text  # Fallback to original

    def _generate_segment_audio(
        self,
        text: str,
        segment_index: int
    ) -> str:
        """Generate TTS audio for a text segment."""
        # Create temp file for audio
        temp_dir = tempfile.gettempdir()
        audio_filename = f"segment_{segment_index}_{os.getpid()}.mp3"
        audio_path = os.path.join(temp_dir, audio_filename)

        try:
            # Generate TTS audio
            self.tts.generate_audio(
                text=text,
                output_file=audio_path
            )

            return audio_path

        except Exception as e:
            if self.verbose:
                verbose_print(f"TTS generation failed for segment {segment_index}: {str(e)}", "error")
            raise

    def _merge_dubbed_audio(
        self,
        segment_audio_files: List[str],
        dubbed_segments: List[DubbedSegment]
    ) -> str:
        """Merge audio segments with proper timing."""
        temp_dir = tempfile.gettempdir()
        merged_path = os.path.join(temp_dir, f"merged_dubbed_audio_{os.getpid()}.mp3")

        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub is required. Install with: pip install pydub")

        # Create combined audio with proper timing
        combined = AudioSegment.empty()
        current_time = 0.0

        for audio_file, segment in zip(segment_audio_files, dubbed_segments):
            # Add silence to match segment start time
            silence_duration = max(0, (segment.start_time - current_time) * 1000)  # ms
            if silence_duration > 0:
                combined += AudioSegment.silent(duration=int(silence_duration))

            # Add segment audio
            segment_audio = AudioSegment.from_file(audio_file)
            combined += segment_audio

            # Update current time
            current_time = segment.start_time + (len(segment_audio) / 1000.0)

        # Export merged audio
        combined.export(merged_path, format="mp3")

        return merged_path

    def _get_language_name(self, language_code: str) -> str:
        """Get language name from code."""
        language_map = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'nl': 'Dutch', 'sv': 'Swedish', 'no': 'Norwegian', 'da': 'Danish',
            'fi': 'Finnish', 'pl': 'Polish', 'tr': 'Turkish', 'he': 'Hebrew',
            'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay',
        }
        return language_map.get(language_code.lower(), language_code.upper())

    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    if self.verbose:
                        verbose_print(f"Cleaned up: {file_path}", "debug")
            except Exception as e:
                if self.verbose:
                    verbose_print(f"Failed to clean up {file_path}: {str(e)}", "warning")
