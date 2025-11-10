# SimplerLLM Video Features

This document describes the new video transcription and dubbing capabilities added to SimplerLLM.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Video Transcription](#video-transcription)
- [Multi-Language Captions](#multi-language-captions)
- [Video Dubbing](#video-dubbing)
- [API Reference](#api-reference)
- [Examples](#examples)

## Overview

SimplerLLM now supports comprehensive video processing capabilities:

1. **Video Transcription**: Transcribe videos (local files or YouTube URLs) to text with precise timing information
2. **Multi-Language Captions**: Generate subtitles in multiple languages using LLM translation
3. **Video Dubbing**: Replace video audio with AI-generated speech in different languages

## Features

### Video Transcription & Captions

- ✅ Transcribe local video files (MP4, AVI, MOV)
- ✅ Transcribe YouTube videos from URLs
- ✅ Generate timing-accurate captions
- ✅ Export to SRT or VTT subtitle formats
- ✅ Auto-detect or specify source language
- ✅ Translate captions to multiple languages
- ✅ LLM-powered translation with customizable prompts

### Video Dubbing

- ✅ Dub videos to any target language
- ✅ Auto-translate dialogue using LLM
- ✅ Generate natural TTS audio
- ✅ Adjust speech speed to match original timing
- ✅ Optional pitch preservation (requires ffmpeg)
- ✅ Support multiple TTS providers (OpenAI, ElevenLabs)
- ✅ Customizable speed adjustment ranges
- ✅ Detailed segment-level tracking

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Required Dependencies

The following dependencies are required and included in `requirements.txt`:

```
moviepy>=1.0.3      # Video processing
yt-dlp>=2023.3.4    # YouTube video download
pydub>=0.25.1       # Audio manipulation
openai>=1.59        # OpenAI API (STT/TTS/LLM)
```

### Optional Dependencies

For pitch-preserving speed adjustment:
```bash
# Install ffmpeg (platform-specific)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows:
# Download from https://ffmpeg.org/
```

### API Keys

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"  # Optional, for ElevenLabs TTS
```

Or use a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
ELEVENLABS_API_KEY=your-elevenlabs-api-key
```

## Quick Start

### Video Transcription

```python
from SimplerLLM.voice import STT, STTProvider, VideoTranscriber

# Initialize STT
stt = STT.create(provider=STTProvider.OPENAI)

# Create transcriber
transcriber = VideoTranscriber(stt_instance=stt, verbose=True)

# Transcribe video
result = transcriber.transcribe_video(
    video_source="video.mp4",
    language="en",
    output_format="srt"
)

# Save transcription
transcriber.save_transcription(result, "output.srt", format="srt")
```

### Multi-Language Captions

```python
from SimplerLLM.voice import STT, STTProvider, MultiLanguageCaptionGenerator
from SimplerLLM.language import LLM, LLMProvider

# Initialize STT and LLM
stt = STT.create(provider=STTProvider.OPENAI)
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")

# Create caption generator
caption_gen = MultiLanguageCaptionGenerator(
    stt_instance=stt,
    llm_instance=llm,
    verbose=True
)

# Generate captions in multiple languages
result = caption_gen.generate_captions(
    video_source="video.mp4",
    target_languages=["es", "fr", "de"],
    format="srt"
)

# Save all captions
result.save_all(output_dir="captions", base_filename="my_video")
# Creates: captions/my_video.es.srt, captions/my_video.fr.srt, etc.
```

### Video Dubbing

```python
from SimplerLLM.voice import TTS, TTSProvider, STT, STTProvider, VideoDubber
from SimplerLLM.language import LLM, LLMProvider

# Initialize providers
tts = TTS.create(provider=TTSProvider.OPENAI, voice="alloy")
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")
stt = STT.create(provider=STTProvider.OPENAI)

# Create dubber
dubber = VideoDubber(
    tts_instance=tts,
    llm_instance=llm,
    stt_instance=stt,
    verbose=True
)

# Dub video
result = dubber.dub_video(
    video_path="video.mp4",
    target_language="es",
    output_path="video_spanish.mp4",
    match_timing=True
)

print(f"Dubbed video saved to: {result.output_video_path}")
```

## Video Transcription

### Transcribe Local Video

```python
transcriber = VideoTranscriber(stt_instance=stt, verbose=True)

result = transcriber.transcribe_video(
    video_source="path/to/video.mp4",
    language="en",           # Optional: auto-detect if None
    output_format="srt",     # "srt" or "vtt"
    prompt=None,             # Optional: guide transcription style
    temperature=0.0          # 0.0-1.0, higher = more random
)
```

### Transcribe YouTube Video

```python
result = transcriber.transcribe_video(
    video_source="https://www.youtube.com/watch?v=VIDEO_ID",
    output_format="vtt"
)
```

### Access Transcription Data

```python
# Access transcription text
print(result.text)

# Access segments with timing
for segment in result.segments:
    print(f"{segment.start_time:.2f}s - {segment.end_time:.2f}s: {segment.text}")

# Get duration
print(f"Video duration: {result.duration:.2f}s")

# Get language
print(f"Detected language: {result.language}")
```

### Export Formats

```python
# Export as SRT
srt_content = result.to_srt()
with open("output.srt", "w", encoding="utf-8") as f:
    f.write(srt_content)

# Export as VTT
vtt_content = result.to_vtt()
with open("output.vtt", "w", encoding="utf-8") as f:
    f.write(vtt_content)

# Or use the convenience method
transcriber.save_transcription(result, "output.srt", format="srt")
```

## Multi-Language Captions

### Generate Captions in Multiple Languages

```python
caption_gen = MultiLanguageCaptionGenerator(
    stt_instance=stt,
    llm_instance=llm,
    verbose=True
)

result = caption_gen.generate_captions(
    video_source="video.mp4",
    target_languages=["es", "fr", "de", "ja", "zh"],
    format="srt",
    original_language="en"  # Optional: specify if known
)
```

### Supported Language Codes

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ar` - Arabic
- `hi` - Hindi
- And more...

### Access Translated Captions

```python
# Get specific language captions
spanish_captions = result.get_captions("es")
print(spanish_captions.content)

# Access segments
for segment in spanish_captions.segments:
    print(f"{segment.start_time:.2f}s: {segment.text}")

# Save individual caption file
spanish_captions.save_to_file("spanish_captions.srt")
```

### Save All Captions

```python
# Save all languages to a directory
result.save_all(
    output_dir="captions",
    base_filename="my_video"
)
# Creates:
# - captions/my_video.es.srt
# - captions/my_video.fr.srt
# - captions/my_video.de.srt
# - etc.
```

### Translate Existing Transcription

```python
# If you already have a transcription
transcription = transcriber.transcribe_video("video.mp4")

# Translate it to new languages without re-transcribing
result = caption_gen.translate_existing_captions(
    transcription=transcription,
    target_languages=["it", "pt", "ru"],
    format="vtt"
)
```

### Custom Translation Prompts

```python
custom_prompt = """
Translate the following text from {source_language} to {target_language}.
Use a casual, conversational tone.
Only return the translated text:

{text}
"""

result = caption_gen.generate_captions(
    video_source="video.mp4",
    target_languages=["es"],
    format="srt",
    prompt_template=custom_prompt
)
```

## Video Dubbing

### Basic Dubbing

```python
dubber = VideoDubber(
    tts_instance=tts,
    llm_instance=llm,
    stt_instance=stt,
    verbose=True
)

result = dubber.dub_video(
    video_path="video.mp4",
    target_language="es",
    output_path="video_spanish.mp4",
    match_timing=True,              # Adjust speed to match original
    speed_range=(0.75, 1.5),        # Min and max speed adjustment
    preserve_pitch=False,            # Requires ffmpeg
    source_language=None,            # Auto-detect if None
    cleanup_temp_files=True          # Clean up temporary audio files
)
```

### Using Different TTS Providers

#### OpenAI TTS

```python
tts = TTS.create(
    provider=TTSProvider.OPENAI,
    model_name="tts-1",     # or "tts-1-hd" for higher quality
    voice="alloy"           # alloy, echo, fable, onyx, nova, shimmer
)
```

#### ElevenLabs TTS (Higher Quality)

```python
tts = TTS.create(
    provider=TTSProvider.ELEVENLABS,
    voice="Adam"            # Or any ElevenLabs voice name/ID
)
```

### Timing Control

#### Match Original Timing

```python
result = dubber.dub_video(
    video_path="video.mp4",
    target_language="fr",
    output_path="video_french.mp4",
    match_timing=True,           # Adjust speed to match
    speed_range=(0.8, 1.3)       # Allow 80%-130% speed
)
```

#### Natural Speech (No Timing Match)

```python
result = dubber.dub_video(
    video_path="video.mp4",
    target_language="de",
    output_path="video_german.mp4",
    match_timing=False          # Use natural TTS speed
)
```

### Pitch Preservation

Requires ffmpeg:

```python
result = dubber.dub_video(
    video_path="video.mp4",
    target_language="ja",
    output_path="video_japanese.mp4",
    match_timing=True,
    preserve_pitch=True         # Preserve pitch when adjusting speed
)
```

### Inspect Dubbing Results

```python
result = dubber.dub_video(...)

# Overall statistics
print(f"Total segments: {result.total_segments}")
print(f"Duration: {result.duration:.2f}s")
print(f"Process time: {result.process_time:.2f}s")
print(f"TTS Provider: {result.tts_provider}")
print(f"Average speed adjustment: {result.average_speed_adjustment:.2f}x")

# Segment details
for segment in result.segments:
    print(f"\nSegment {segment.index}:")
    print(f"  Original: {segment.original_text}")
    print(f"  Translated: {segment.translated_text}")
    print(f"  Time: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
    if segment.speed_adjustment:
        print(f"  Speed: {segment.speed_adjustment:.2f}x")
```

### Batch Processing

Dub to multiple languages:

```python
languages = ["es", "fr", "de", "ja"]

for lang in languages:
    print(f"Dubbing to {lang}...")
    result = dubber.dub_video(
        video_path="original.mp4",
        target_language=lang,
        output_path=f"video_{lang}.mp4"
    )
    print(f"✓ Completed: video_{lang}.mp4")
```

## API Reference

### VideoTranscriber

```python
class VideoTranscriber:
    def __init__(self, stt_instance, verbose=False)

    def transcribe_video(
        self,
        video_source: str,
        language: Optional[str] = None,
        output_format: str = "srt",
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> VideoTranscriptionResult

    def save_transcription(
        self,
        result: VideoTranscriptionResult,
        output_path: str,
        format: str = "srt"
    ) -> None
```

### MultiLanguageCaptionGenerator

```python
class MultiLanguageCaptionGenerator:
    def __init__(self, stt_instance, llm_instance, verbose=False)

    def generate_captions(
        self,
        video_source: str,
        target_languages: List[str],
        format: str = "srt",
        original_language: Optional[str] = None,
        prompt_template: Optional[str] = None
    ) -> MultiLanguageCaptionsResult

    def translate_existing_captions(
        self,
        transcription: VideoTranscriptionResult,
        target_languages: List[str],
        format: str = "srt",
        prompt_template: Optional[str] = None
    ) -> MultiLanguageCaptionsResult
```

### VideoDubber

```python
class VideoDubber:
    def __init__(
        self,
        tts_instance,
        llm_instance,
        stt_instance,
        verbose=False
    )

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
    ) -> VideoDubbingResult
```

## Examples

See the `examples/` directory for complete working examples:

- [`video_transcription_example.py`](examples/video_transcription_example.py) - Video transcription and multi-language captions
- [`video_dubbing_example.py`](examples/video_dubbing_example.py) - Video dubbing examples

## Architecture

The video features are built on SimplerLLM's modular architecture:

```
SimplerLLM/voice/
├── video_transcription/
│   ├── base.py                 # VideoTranscriber
│   ├── caption_generator.py   # MultiLanguageCaptionGenerator
│   ├── models.py               # Pydantic models
│   └── utils/
│       ├── video_utils.py      # Video/audio extraction
│       └── subtitle_formatter.py
│
└── video_dubbing/
    ├── base.py                 # VideoDubber
    ├── models.py               # Pydantic models
    ├── audio_sync.py           # Speed adjustment
    └── video_processor.py      # Audio replacement
```

### Key Design Principles

1. **Modular**: Pass provider instances (TTS, STT, LLM) for full flexibility
2. **Reusable**: Built on existing SimplerLLM voice features
3. **Provider-Agnostic**: Works with any TTS/STT/LLM provider
4. **Type-Safe**: Uses Pydantic models for all data
5. **Production-Ready**: Includes error handling, cleanup, and logging

## Supported Video Formats

### Input Formats
- MP4
- AVI
- MOV
- YouTube URLs

### Output Formats
- Subtitles: SRT, VTT
- Videos: MP4 (default)

## Performance Tips

1. **Use appropriate models**:
   - `gpt-3.5-turbo` for faster/cheaper translation
   - `gpt-4` for higher quality translation

2. **Batch processing**: Process multiple videos in parallel

3. **Reuse instances**: Initialize TTS/LLM/STT once, reuse for multiple videos

4. **Cache transcriptions**: Save transcriptions and translate later

5. **Speed ranges**: Tighter speed ranges = faster processing but may affect quality

## Troubleshooting

### Common Issues

**Issue**: `ImportError: moviepy is required`
```bash
pip install moviepy
```

**Issue**: `ffmpeg not found`
```bash
# Install ffmpeg for your platform
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
```

**Issue**: YouTube download fails
```bash
pip install --upgrade yt-dlp
```

**Issue**: Audio quality issues
- Use ElevenLabs TTS for better quality
- Enable pitch preservation with `preserve_pitch=True`
- Adjust `speed_range` for more natural pacing

## License

This feature is part of SimplerLLM and follows the same license.

## Contributing

Contributions are welcome! Please see the main SimplerLLM repository for contribution guidelines.
