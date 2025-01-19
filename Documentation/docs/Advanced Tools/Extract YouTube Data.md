---
sidebar_position: 3
--- 

# Extract YouTube Data

The functions in this section are designed to extract the transcript of any YouTube videoalong with their timestamps if needed. You can benefit from these capabilities to build powerful APIs / tools / applications.

## `get_youtube_transcript(video_url)` Function

This function also takes only the `video_url`, and it returns the transcript of the YouTube video, formatting it into a simple readable string. 

### Example Usage

```python
from SimplerLLM.tools.youtube import get_youtube_transcript

video_transcript = get_youtube_transcript("https://www.youtube.com/watch?v=r9PjzmUmk1w")

print(video_transcript)
```

## `get_youtube_transcript_with_timing(video_url)` Function

This function also takes only the `video_url`, and retrieves the transcript of a YouTube video, including timing information for each line. It returns a list of dictionaries, where each dictionary refers to a part of the transcript and it contains the following:
- `text`: The transcript text of a specific segment of the video.
- `start`: The start time of the segment in seconds.
- `duration`: The duration of the segment in seconds.

### Example Usage

```python
from SimplerLLM.tools.youtube import get_youtube_transcript_with_timing

video_transcript = get_youtube_transcript_with_timing("https://www.youtube.com/watch?v=r9PjzmUmk1w")

print(video_transcript)
```

Here's the output format of a small section:
```
[{'text': 'hi friends in this video I will show you', 'start': 0.12, 'duration': 6.08}, {'text': 'how to turn any WordPress website into a', 'start': 2.639, 'duration': 7.481}, {'text': 'full SAS business using only three', 'start': 6.2, 'duration': 7.639}, {'text': 'plugins this is exactly what I did on my', 'start': 10.12, 'duration': 6.56}, {'text': 'website you will see here I have a list', 'start': 13.839, 'duration': 5.401}]
```

That's how you can benefit from SimplerLLM to make extracting YouTube data Simpler!