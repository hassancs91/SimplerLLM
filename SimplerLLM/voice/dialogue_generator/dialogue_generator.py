"""
DialogueGenerator - Generate and synthesize multi-speaker dialogues.

This module provides functionality to:
1. Generate dialogue text using LLM
2. Load dialogue from structured data or files
3. Convert dialogue to audio using TTS
4. Merge audio files into complete dialogue
"""

import os
import json
import time
import asyncio
from typing import Optional, Dict, List, Union, Any
from pathlib import Path

from .models import (
    Dialogue,
    DialogueLine,
    SpeakerConfig,
    DialogueGenerationConfig,
    AudioDialogueResult,
    DialogueStyle
)
from .audio_merger import merge_audio_files, get_audio_duration
from SimplerLLM.utils.custom_verbose import verbose_print


class DialogueGenerator:
    """
    Generate and synthesize multi-speaker dialogues.

    Supports multiple input methods:
    - Generate dialogue from topic using LLM
    - Load from structured data (dict/array)
    - Load from JSON file

    Outputs:
    - Individual audio files per line
    - Combined audio file with pauses
    """

    # Default voices for auto-assignment (OpenAI voices)
    DEFAULT_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(
        self,
        tts_instance,
        llm_instance=None,
        verbose: bool = False
    ):
        """
        Initialize DialogueGenerator.

        Args:
            tts_instance: TTS instance (any provider)
            llm_instance: Optional LLM instance for dialogue generation
            verbose: Enable verbose logging
        """
        self.tts = tts_instance
        self.llm = llm_instance
        self.verbose = verbose

        if self.verbose:
            verbose_print("DialogueGenerator initialized", "info")
            verbose_print(f"TTS provider: {self.tts.provider.name if hasattr(self.tts, 'provider') else 'Unknown'}", "debug")
            if self.llm:
                verbose_print(f"LLM provider: {self.llm.provider.name if hasattr(self.llm, 'provider') else 'Unknown'}", "debug")

    def generate_dialogue_text(
        self,
        topic: str,
        config: Optional[DialogueGenerationConfig] = None
    ) -> Dialogue:
        """
        Generate dialogue text using LLM with structured output.

        Args:
            topic: Topic for the dialogue
            config: Optional configuration for generation

        Returns:
            Generated Dialogue object

        Raises:
            ValueError: If LLM instance is not provided
        """
        if not self.llm:
            raise ValueError(
                "LLM instance is required for dialogue generation. "
                "Provide llm_instance when creating DialogueGenerator."
            )

        if config is None:
            config = DialogueGenerationConfig()

        if self.verbose:
            verbose_print(f"Generating dialogue on topic: {topic}", "info")
            verbose_print(f"Config: {config.num_speakers} speakers, {config.num_exchanges} exchanges", "debug")

        # Determine speaker names
        if config.speaker_roles:
            speakers = config.speaker_roles[:config.num_speakers]
        else:
            # Generate generic speaker names
            if config.include_narrator:
                speakers = ["Narrator"] + [f"Speaker{i}" for i in range(1, config.num_speakers)]
            else:
                speakers = [f"Speaker{i}" for i in range(1, config.num_speakers + 1)]

        # Build prompt for LLM
        prompt = self._build_dialogue_prompt(topic, speakers, config)

        if self.verbose:
            verbose_print("Sending request to LLM...", "debug")

        # Generate using LLM
        try:
            response = self.llm.generate_response(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=2000,
                json_mode=True
            )

            # Parse JSON response
            dialogue_data = json.loads(response)

            # Create Dialogue object
            lines = [
                DialogueLine(
                    speaker=line.get("speaker"),
                    text=line.get("text"),
                    pause_after=line.get("pause_after", 0.5)
                )
                for line in dialogue_data.get("lines", [])
            ]

            dialogue = Dialogue(
                title=dialogue_data.get("title", f"Dialogue about {topic}"),
                description=dialogue_data.get("description"),
                speakers=speakers,
                lines=lines,
                metadata={
                    "topic": topic,
                    "generated_by": "LLM",
                    "config": config.dict()
                }
            )

            if self.verbose:
                verbose_print(f"Generated dialogue with {len(lines)} lines", "info")

            return dialogue

        except Exception as e:
            raise Exception(f"Failed to generate dialogue: {str(e)}")

    def _build_dialogue_prompt(
        self,
        topic: str,
        speakers: List[str],
        config: DialogueGenerationConfig
    ) -> str:
        """Build prompt for LLM dialogue generation."""

        speaker_list = ", ".join(speakers)

        prompt = f"""Generate a {config.dialogue_style.value} dialogue about the topic: "{topic}"

Requirements:
- Number of speakers: {len(speakers)}
- Speaker names: {speaker_list}
- Number of exchanges: approximately {config.num_exchanges}
- Language: {config.language}
- Style: {config.dialogue_style.value}

"""

        if config.context:
            prompt += f"Additional context: {config.context}\n\n"

        prompt += """Generate a natural, engaging dialogue. Return ONLY valid JSON with this exact structure:
{
    "title": "Dialogue Title",
    "description": "Brief description",
    "lines": [
        {
            "speaker": "Speaker1",
            "text": "What they say...",
            "pause_after": 0.5
        },
        {
            "speaker": "Speaker2",
            "text": "Their response...",
            "pause_after": 0.5
        }
    ]
}

Important:
- Make the dialogue natural and contextually appropriate
- Each speaker should have roughly equal participation
- Use the exact speaker names provided
- Set pause_after to 0.3-0.8 seconds depending on context
- Generate approximately {config.num_exchanges * len(speakers)} lines total

Return ONLY the JSON, no other text."""

        return prompt

    def load_dialogue(self, dialogue_data: Union[Dict, List]) -> Dialogue:
        """
        Load dialogue from structured data.

        Args:
            dialogue_data: Dict or List representing dialogue

        Returns:
            Dialogue object

        Example dict format:
            {
                "title": "Sample Dialogue",
                "speakers": ["Alice", "Bob"],
                "lines": [
                    {"speaker": "Alice", "text": "Hello!", "voice": "nova"},
                    {"speaker": "Bob", "text": "Hi!", "voice": "echo"}
                ]
            }

        Example list format:
            [
                {"speaker": "Alice", "text": "Hello!"},
                {"speaker": "Bob", "text": "Hi!"}
            ]
        """
        if self.verbose:
            verbose_print("Loading dialogue from structured data", "info")

        # Handle list format (just lines)
        if isinstance(dialogue_data, list):
            lines = [DialogueLine(**line) for line in dialogue_data]
            speakers = list(set(line.speaker for line in lines))

            return Dialogue(
                speakers=speakers,
                lines=lines
            )

        # Handle dict format (full dialogue)
        if isinstance(dialogue_data, dict):
            # Parse speaker configs if provided
            speaker_configs = None
            if "speaker_configs" in dialogue_data and dialogue_data["speaker_configs"] is not None:
                speaker_configs = {
                    name: SpeakerConfig(**config)
                    for name, config in dialogue_data["speaker_configs"].items()
                }

            # Parse lines
            lines = [DialogueLine(**line) for line in dialogue_data.get("lines", [])]

            return Dialogue(
                title=dialogue_data.get("title"),
                description=dialogue_data.get("description"),
                speakers=dialogue_data.get("speakers", []),
                speaker_configs=speaker_configs,
                lines=lines,
                metadata=dialogue_data.get("metadata")
            )

        raise ValueError("dialogue_data must be a dict or list")

    def load_dialogue_from_file(self, file_path: str) -> Dialogue:
        """
        Load dialogue from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Dialogue object
        """
        if self.verbose:
            verbose_print(f"Loading dialogue from file: {file_path}", "info")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dialogue file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)

        return self.load_dialogue(dialogue_data)

    def save_dialogue_to_file(self, dialogue: Dialogue, file_path: str):
        """
        Save dialogue to JSON file.

        Args:
            dialogue: Dialogue object to save
            file_path: Output file path
        """
        if self.verbose:
            verbose_print(f"Saving dialogue to file: {file_path}", "info")

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convert to dict and save
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dialogue.dict(), f, indent=2, ensure_ascii=False)

        if self.verbose:
            verbose_print(f"Dialogue saved successfully", "info")

    def generate_audio(
        self,
        dialogue: Dialogue,
        output_dir: str = "output",
        save_individual: bool = True,
        save_combined: bool = True,
        pause_duration: float = 0.5,
        audio_format: str = "mp3",
        filename_prefix: str = "dialogue"
    ) -> AudioDialogueResult:
        """
        Generate audio from dialogue.

        Args:
            dialogue: Dialogue object
            output_dir: Output directory for audio files
            save_individual: Save individual line audio files
            save_combined: Create combined audio file
            pause_duration: Default pause between lines (seconds)
            audio_format: Audio format (mp3, wav, etc.)
            filename_prefix: Prefix for output files

        Returns:
            AudioDialogueResult with file paths and metadata
        """
        start_time = time.time()

        if self.verbose:
            verbose_print(f"Generating audio for {len(dialogue.lines)} lines", "info")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Auto-assign voices if not specified
        dialogue = self._assign_voices(dialogue)

        # Generate audio for each line
        individual_files = []
        audio_files_for_merge = []

        for i, line in enumerate(dialogue.lines, 1):
            # Determine voice and speed for this line
            voice = line.voice
            speed = line.speed

            # If not specified in line, use speaker config
            if not voice or not speed:
                if dialogue.speaker_configs and line.speaker in dialogue.speaker_configs:
                    config = dialogue.speaker_configs[line.speaker]
                    voice = voice or config.voice
                    speed = speed or config.speed

            # Defaults
            voice = voice or "alloy"
            speed = speed or 1.0

            # Generate filename
            speaker_safe = line.speaker.replace(" ", "_")
            filename = f"{filename_prefix}_line_{i:03d}_{speaker_safe}.{audio_format}"
            file_path = os.path.join(output_dir, filename)

            if self.verbose:
                verbose_print(f"[{i}/{len(dialogue.lines)}] Generating: {line.speaker} - {line.text[:50]}...", "debug")

            # Generate audio
            self.tts.generate_speech(
                text=line.text,
                voice=voice,
                speed=speed,
                response_format=audio_format,
                output_path=file_path
            )

            individual_files.append(file_path)
            audio_files_for_merge.append(file_path)

        # Create combined audio if requested
        combined_file = None
        if save_combined and audio_files_for_merge:
            combined_filename = f"{filename_prefix}_combined.{audio_format}"
            combined_file = os.path.join(output_dir, combined_filename)

            if self.verbose:
                verbose_print("Merging audio files...", "info")

            try:
                merge_audio_files(
                    file_paths=audio_files_for_merge,
                    output_path=combined_file,
                    pause_duration=pause_duration,
                    format=audio_format,
                    verbose=self.verbose
                )
            except ImportError as e:
                if self.verbose:
                    verbose_print(f"Warning: Could not merge audio files: {e}", "warning")
                combined_file = None

        # Calculate duration estimate
        total_duration = None
        try:
            if combined_file and os.path.exists(combined_file):
                total_duration = get_audio_duration(combined_file)
        except:
            pass

        process_time = time.time() - start_time

        # Create result
        result = AudioDialogueResult(
            dialogue=dialogue,
            individual_files=individual_files if save_individual else [],
            combined_file=combined_file,
            total_lines=len(dialogue.lines),
            total_duration_estimate=total_duration,
            process_time=process_time,
            metadata={
                "output_dir": output_dir,
                "audio_format": audio_format,
                "pause_duration": pause_duration
            }
        )

        if self.verbose:
            verbose_print(f"Audio generation complete in {process_time:.2f}s", "info")
            if combined_file:
                verbose_print(f"Combined file: {combined_file}", "info")

        return result

    async def generate_audio_async(
        self,
        dialogue: Dialogue,
        output_dir: str = "output",
        save_individual: bool = True,
        save_combined: bool = True,
        pause_duration: float = 0.5,
        audio_format: str = "mp3",
        filename_prefix: str = "dialogue"
    ) -> AudioDialogueResult:
        """
        Async version: Generate audio from dialogue.

        Generates all lines concurrently for faster processing.
        """
        start_time = time.time()

        if self.verbose:
            verbose_print(f"Generating audio asynchronously for {len(dialogue.lines)} lines", "info")

        os.makedirs(output_dir, exist_ok=True)
        dialogue = self._assign_voices(dialogue)

        # Create tasks for all lines
        tasks = []
        filenames = []

        for i, line in enumerate(dialogue.lines, 1):
            voice = line.voice
            speed = line.speed

            if not voice or not speed:
                if dialogue.speaker_configs and line.speaker in dialogue.speaker_configs:
                    config = dialogue.speaker_configs[line.speaker]
                    voice = voice or config.voice
                    speed = speed or config.speed

            voice = voice or "alloy"
            speed = speed or 1.0

            speaker_safe = line.speaker.replace(" ", "_")
            filename = f"{filename_prefix}_line_{i:03d}_{speaker_safe}.{audio_format}"
            file_path = os.path.join(output_dir, filename)

            filenames.append(file_path)

            task = self.tts.generate_speech_async(
                text=line.text,
                voice=voice,
                speed=speed,
                response_format=audio_format,
                output_path=file_path
            )
            tasks.append(task)

        # Generate all concurrently
        await asyncio.gather(*tasks)

        individual_files = filenames

        # Create combined audio
        combined_file = None
        if save_combined and individual_files:
            combined_filename = f"{filename_prefix}_combined.{audio_format}"
            combined_file = os.path.join(output_dir, combined_filename)

            try:
                merge_audio_files(
                    file_paths=individual_files,
                    output_path=combined_file,
                    pause_duration=pause_duration,
                    format=audio_format,
                    verbose=self.verbose
                )
            except ImportError:
                combined_file = None

        total_duration = None
        try:
            if combined_file:
                total_duration = get_audio_duration(combined_file)
        except:
            pass

        process_time = time.time() - start_time

        result = AudioDialogueResult(
            dialogue=dialogue,
            individual_files=individual_files if save_individual else [],
            combined_file=combined_file,
            total_lines=len(dialogue.lines),
            total_duration_estimate=total_duration,
            process_time=process_time,
            metadata={
                "output_dir": output_dir,
                "audio_format": audio_format,
                "pause_duration": pause_duration,
                "async": True
            }
        )

        if self.verbose:
            verbose_print(f"Async audio generation complete in {process_time:.2f}s", "info")

        return result

    def generate_complete(
        self,
        topic: str,
        num_speakers: int = 2,
        num_exchanges: int = 5,
        speaker_configs: Optional[Dict[str, Dict]] = None,
        dialogue_style: DialogueStyle = DialogueStyle.CASUAL,
        output_dir: str = "output",
        save_individual: bool = True,
        save_combined: bool = True,
        pause_duration: float = 0.5,
        audio_format: str = "mp3",
        **kwargs
    ) -> AudioDialogueResult:
        """
        Complete workflow: Generate dialogue text and convert to audio.

        Args:
            topic: Topic for dialogue
            num_speakers: Number of speakers
            num_exchanges: Number of exchanges
            speaker_configs: Optional voice configs for speakers
            dialogue_style: Style of dialogue
            output_dir: Output directory
            save_individual: Save individual files
            save_combined: Save combined file
            pause_duration: Pause between lines
            audio_format: Audio format
            **kwargs: Additional config options

        Returns:
            AudioDialogueResult
        """
        if self.verbose:
            verbose_print(f"Starting complete dialogue generation: {topic}", "info")

        # Create generation config
        config = DialogueGenerationConfig(
            num_speakers=num_speakers,
            num_exchanges=num_exchanges,
            dialogue_style=dialogue_style,
            **kwargs
        )

        # Generate dialogue text
        dialogue = self.generate_dialogue_text(topic, config)

        # Apply speaker configs if provided
        if speaker_configs:
            dialogue.speaker_configs = {
                name: SpeakerConfig(**config)
                for name, config in speaker_configs.items()
            }

        # Generate audio
        result = self.generate_audio(
            dialogue=dialogue,
            output_dir=output_dir,
            save_individual=save_individual,
            save_combined=save_combined,
            pause_duration=pause_duration,
            audio_format=audio_format,
            filename_prefix=topic.replace(" ", "_")[:50]
        )

        return result

    def _assign_voices(self, dialogue: Dialogue) -> Dialogue:
        """Auto-assign voices to speakers if not specified."""

        # Check if all speakers have voice configs
        if dialogue.speaker_configs:
            # All speakers have configs
            if all(speaker in dialogue.speaker_configs for speaker in dialogue.speakers):
                return dialogue

        # Need to assign voices
        if not dialogue.speaker_configs:
            dialogue.speaker_configs = {}

        for i, speaker in enumerate(dialogue.speakers):
            if speaker not in dialogue.speaker_configs:
                # Assign voice from default list
                voice = self.DEFAULT_VOICES[i % len(self.DEFAULT_VOICES)]
                dialogue.speaker_configs[speaker] = SpeakerConfig(voice=voice)

        return dialogue
