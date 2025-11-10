"""
Multi-language caption generation for videos.
"""
import time
from typing import List, Optional, Dict
from .models import (
    CaptionSegment,
    LanguageCaptions,
    MultiLanguageCaptionsResult,
    VideoTranscriptionResult
)
from .base import VideoTranscriber
from SimplerLLM.utils.custom_verbose import verbose_print


class MultiLanguageCaptionGenerator:
    """
    Generate captions for videos in multiple languages.

    This class provides multi-language caption generation by:
    1. Transcribing the video in its original language
    2. Translating segments to target languages using LLM
    3. Generating formatted captions (SRT/VTT)
    """

    def __init__(
        self,
        stt_instance,
        llm_instance,
        verbose: bool = False
    ):
        """
        Initialize MultiLanguageCaptionGenerator.

        Args:
            stt_instance: An instance of STT (e.g., from STT.create())
            llm_instance: An instance of LLM (e.g., from LLM.create())
            verbose: Enable verbose logging
        """
        self.stt = stt_instance
        self.llm = llm_instance
        self.transcriber = VideoTranscriber(stt_instance, verbose=verbose)
        self.verbose = verbose

        if self.verbose:
            verbose_print("MultiLanguageCaptionGenerator initialized", "info")

    def generate_captions(
        self,
        video_source: str,
        target_languages: List[str],
        format: str = "srt",
        original_language: Optional[str] = None,
        prompt_template: Optional[str] = None
    ) -> MultiLanguageCaptionsResult:
        """
        Generate captions for a video in multiple languages.

        Args:
            video_source: Path to video file or YouTube URL
            target_languages: List of language codes (e.g., ["es", "fr", "de"])
            format: Caption format - "srt" or "vtt"
            original_language: Original language code (auto-detect if None)
            prompt_template: Custom translation prompt template

        Returns:
            MultiLanguageCaptionsResult with captions in all languages

        Example:
            >>> caption_gen = MultiLanguageCaptionGenerator(stt, llm)
            >>> result = caption_gen.generate_captions(
            ...     "video.mp4",
            ...     target_languages=["es", "fr", "de"],
            ...     format="srt"
            ... )
            >>> result.save_all("output", "my_video")
        """
        start_time = time.time()

        if self.verbose:
            verbose_print(
                f"Generating captions for {len(target_languages)} languages: {target_languages}",
                "info"
            )

        # Step 1: Transcribe the video
        if self.verbose:
            verbose_print("Step 1: Transcribing video...", "info")

        transcription = self.transcriber.transcribe_video(
            video_source,
            language=original_language,
            output_format=format
        )

        # Step 2: Generate captions for each target language
        captions_dict: Dict[str, LanguageCaptions] = {}

        for lang_code in target_languages:
            if self.verbose:
                verbose_print(f"Step 2: Generating captions for '{lang_code}'...", "info")

            captions = self._generate_language_captions(
                transcription,
                lang_code,
                format,
                prompt_template
            )
            captions_dict[lang_code] = captions

        # Calculate total process time
        process_time = time.time() - start_time

        # Create result
        result = MultiLanguageCaptionsResult(
            original_language=transcription.language or "unknown",
            original_transcription=transcription,
            captions=captions_dict,
            target_languages=target_languages,
            process_time=process_time,
            total_segments=len(transcription.segments)
        )

        if self.verbose:
            verbose_print(
                f"Caption generation completed - {len(target_languages)} languages, {process_time:.2f}s",
                "info"
            )

        return result

    def _generate_language_captions(
        self,
        transcription: VideoTranscriptionResult,
        target_language: str,
        format: str,
        prompt_template: Optional[str] = None
    ) -> LanguageCaptions:
        """
        Generate captions in a specific language by translating segments.

        Args:
            transcription: Original video transcription
            target_language: Target language code
            format: Caption format ("srt" or "vtt")
            prompt_template: Custom translation prompt

        Returns:
            LanguageCaptions object
        """
        # Translate each segment
        translated_segments = []

        for segment in transcription.segments:
            translated_text = self._translate_text(
                segment.text,
                target_language,
                transcription.language,
                prompt_template
            )

            # Create new segment with translated text
            translated_segment = CaptionSegment(
                index=segment.index,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=translated_text,
                duration=segment.duration
            )
            translated_segments.append(translated_segment)

        # Format captions
        if format == "srt":
            content = "\n".join(seg.to_srt() for seg in translated_segments)
        elif format == "vtt":
            content = "WEBVTT\n\n" + "\n".join(seg.to_vtt() for seg in translated_segments)
        else:
            raise ValueError(f"format must be 'srt' or 'vtt', got: {format}")

        # Create LanguageCaptions object
        language_name = self._get_language_name(target_language)

        return LanguageCaptions(
            language=language_name,
            language_code=target_language,
            segments=translated_segments,
            format=format,
            content=content
        )

    def _translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Translate text to target language using LLM.

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (optional)
            prompt_template: Custom prompt template

        Returns:
            Translated text
        """
        # Build translation prompt
        if prompt_template:
            prompt = prompt_template.format(
                text=text,
                target_language=target_language,
                source_language=source_language or "auto"
            )
        else:
            lang_name = self._get_language_name(target_language)
            if source_language:
                source_name = self._get_language_name(source_language)
                prompt = f"Translate the following text from {source_name} to {lang_name}. Only return the translated text, no explanations:\n\n{text}"
            else:
                prompt = f"Translate the following text to {lang_name}. Only return the translated text, no explanations:\n\n{text}"

        try:
            # Generate translation using LLM
            response = self.llm.generate_text(
                prompt=prompt,
                temperature=0.3  # Low temperature for more consistent translations
            )

            # Extract translated text (handle both string and object responses)
            if isinstance(response, str):
                translated = response.strip()
            elif hasattr(response, 'generated_text'):
                translated = response.generated_text.strip()
            else:
                translated = str(response).strip()

            return translated

        except Exception as e:
            if self.verbose:
                verbose_print(
                    f"Translation failed for '{text[:50]}...': {str(e)}",
                    "error"
                )
            # Fallback: return original text
            return text

    def _get_language_name(self, language_code: str) -> str:
        """
        Get language name from language code.

        Args:
            language_code: ISO 639-1 language code

        Returns:
            Language name
        """
        # Common language code to name mapping
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'he': 'Hebrew',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'ro': 'Romanian',
            'uk': 'Ukrainian',
            'el': 'Greek',
        }

        return language_map.get(language_code.lower(), language_code.upper())

    def translate_existing_captions(
        self,
        transcription: VideoTranscriptionResult,
        target_languages: List[str],
        format: str = "srt",
        prompt_template: Optional[str] = None
    ) -> MultiLanguageCaptionsResult:
        """
        Translate existing transcription to multiple languages.

        This is useful when you already have a transcription and want to
        generate captions in additional languages without re-transcribing.

        Args:
            transcription: Existing VideoTranscriptionResult
            target_languages: List of language codes
            format: Caption format ("srt" or "vtt")
            prompt_template: Custom translation prompt

        Returns:
            MultiLanguageCaptionsResult with captions in all languages
        """
        start_time = time.time()

        if self.verbose:
            verbose_print(
                f"Translating existing captions to {len(target_languages)} languages",
                "info"
            )

        # Generate captions for each target language
        captions_dict: Dict[str, LanguageCaptions] = {}

        for lang_code in target_languages:
            captions = self._generate_language_captions(
                transcription,
                lang_code,
                format,
                prompt_template
            )
            captions_dict[lang_code] = captions

        # Calculate total process time
        process_time = time.time() - start_time

        # Create result
        result = MultiLanguageCaptionsResult(
            original_language=transcription.language or "unknown",
            original_transcription=transcription,
            captions=captions_dict,
            target_languages=target_languages,
            process_time=process_time,
            total_segments=len(transcription.segments)
        )

        if self.verbose:
            verbose_print(
                f"Translation completed - {len(target_languages)} languages, {process_time:.2f}s",
                "info"
            )

        return result
