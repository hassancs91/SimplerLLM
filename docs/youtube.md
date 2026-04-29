# YouTube Transcripts

Extract transcripts from YouTube videos as plain text or with timing data.

## Setup

Set your SearchAPI key in `.env`:

```env
SEARCHAPI_API_KEY=your_searchapi_key
```

## Basic Usage

Get the full transcript as a single string:

```python
from SimplerLLM.tools.youtube import get_youtube_transcript

transcript = get_youtube_transcript("https://www.youtube.com/watch?v=VIDEO_ID")
print(transcript)  # "First sentence. Second sentence. Third sentence."
```

## With Timing

Get the transcript with start time and duration for each segment:

```python
from SimplerLLM.tools.youtube import get_youtube_transcript_with_timing

transcript = get_youtube_transcript_with_timing("https://www.youtube.com/watch?v=VIDEO_ID")

for segment in transcript.segments:
    print(f"[{segment.start:.1f}s] {segment.text}")
```

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `get_youtube_transcript(video_url)` | `str` | Full transcript as a single string with periods added |
| `get_youtube_transcript_with_timing(video_url)` | `Transcript` | Transcript with timing data per segment |

Both functions accept the same parameter:

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_url` | `str` | YouTube video URL |

## Response Format

`get_youtube_transcript_with_timing()` returns a `Transcript` object:

```python
transcript = get_youtube_transcript_with_timing("https://youtu.be/VIDEO_ID")

print(len(transcript.segments))        # Number of segments

segment = transcript.segments[0]
print(segment.text)                    # "Hello and welcome"
print(segment.start)                   # 0.0 (seconds)
print(segment.duration)                # 2.5 (seconds)
```

| Field | Type | Description |
|-------|------|-------------|
| `Transcript.segments` | `List[TranscriptSegment]` | List of transcript segments |
| `TranscriptSegment.text` | `str` | Segment text content |
| `TranscriptSegment.start` | `float` | Start time in seconds |
| `TranscriptSegment.duration` | `float` | Duration in seconds |

## Supported URL Formats

Both standard and shortened YouTube URLs work:

```python
# Standard
get_youtube_transcript("https://www.youtube.com/watch?v=VIDEO_ID")

# Shortened
get_youtube_transcript("https://youtu.be/VIDEO_ID")
```
