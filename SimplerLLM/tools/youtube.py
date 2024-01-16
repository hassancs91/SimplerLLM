import re
from youtube_transcript_api import YouTubeTranscriptApi



def get_video_transcript(video_url):
    """
    Fetches the transcript of a YouTube video given its URL.

    Parameters:
    video_url (str): The URL of the YouTube video.

    Returns:
    str: The transcript of the video if available, raises an error otherwise.
    """
    # Enhanced regex to handle different YouTube URL formats
    match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", video_url)
    if match:
        video_id = match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([line["text"] for line in transcript])
        return transcript_text
    except Exception as e:
        # Handle other exceptions, e.g., transcript not available
        raise ValueError(f"An error occurred while fetching the transcript: {e}")