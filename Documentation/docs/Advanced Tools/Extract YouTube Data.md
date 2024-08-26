---
sidebar_position: 3
--- 

# Extract YouTube Data

The functions in this section are designed to extract detailed information from YouTube videos, including metadata and transcripts. You can benefit from these capabilities to build powerful APIs / tools / applications.

## `get_video_meta(video_url)` Function

This function takes only the `video_url` as input and it fetches detailed metadata from a specified YouTube video. It retrieves a ton of information returning them in a dictionary format: 
- `video_id`: The unique identifier for the video.
- `video_title`: The title of the video.
- `video_description`: A description of the video.
- `video_length`: The duration of the video in seconds.
- `video_views`: The number of times the video has been viewed.
- `video_author`: The creator of the video.
- `video_publish_date`: The publication date of the video.
- `video_thumbnail_url`: The URL of the video's thumbnail.
- `video_rating`: The average user rating of the video.
- `video_keywords`: Keywords associated with the video.

### Example Usage

```python
from SimplerLLM.tools.youtube import get_video_meta

video_meta = get_video_meta("https://www.youtube.com/watch?v=r9PjzmUmk1w")

print(video_meta)
```

The video meta is returned in the following format:
```
{'video_id': 'r9PjzmUmk1w', 'video_title': 'Build SaaS with WordPress With 3 Plugins Only!', 'video_description': None, 'video_length': 252, 'video_views': 25845, 'video_author': 'Hasan Aboul Hasan', 'video_publish_date': datetime.datetime(2024, 2, 15, 0, 0), 'video_thumbnail_url': 'https://i.ytimg.com/vi/r9PjzmUmk1w/hqdefault.jpg?sqp=-oaymwEXCJADEOABSFryq4qpAwkIARUAAIhCGAE=&rs=AOn4CLDLDIEv0NjGrhaAKQ8GL2SpvwDDng', 'video_rating': None, 'video_keywords': []}
```

Let's say you only want to get the video title, here's how you get it:

```python
from SimplerLLM.tools.youtube import get_video_meta

video_meta = get_video_meta("https://www.youtube.com/watch?v=r9PjzmUmk1w")

print(video_meta.get('video_title'))
```

Use the same method to extract any value you want.

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