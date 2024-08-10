from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import re
from urllib.parse import urlparse
from pydantic import BaseModel
from typing import Optional


def get_video_meta(video_url):
    """
    Retrieves detailed metadata of a YouTube video based on the provided URL.

    Args:
        video_url (str): The URL of the YouTube video.

    Returns:
        dict: A dictionary containing various metadata details of the video, including:
              - video_id: The ID of the video
              - video_title: The title of the video
              - video_description: The description of the video
              - video_length: The duration of the video in seconds
              - video_views: The number of views the video has
              - video_author: The name of the channel that uploaded the video
              - video_publish_date: The publish date of the video
              - video_thumbnail_url: The URL of the video's thumbnail
              - video_rating: The average rating of the video
              - video_keywords: A list of keywords associated with the video

    Raises:
        ValueError: If the provided YouTube URL is invalid.
        Exception: If an error occurs while fetching the video details.
    """
    # Extract the video ID from the URL
    match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", video_url)
    if match:
        video_id = match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

    try:
        yt = YouTube(video_url)

        # Get video details
        video_title = yt.title
        video_description = yt.description
        video_length = yt.length
        video_views = yt.views
        video_author = yt.author
        video_publish_date = yt.publish_date
        video_thumbnail_url = yt.thumbnail_url
        video_rating = yt.rating
        video_keywords = yt.keywords

        # Return all details in a dictionary
        return {
            "video_id": video_id,
            "video_title": video_title,
            "video_description": video_description,
            "video_length": video_length,
            "video_views": video_views,
            "video_author": video_author,
            "video_publish_date": video_publish_date,
            "video_thumbnail_url": video_thumbnail_url,
            "video_rating": video_rating,
            "video_keywords": video_keywords
        }

    except Exception as e:
        raise Exception(f"An error occurred while fetching the video details: {e}")

def get_youtube_transcript_with_timing(video_url):
    """
    Fetches the transcript of a YouTube video given its URL.

    Parameters:
    video_url (str): The URL of the YouTube video.

    Returns:
            list: A list of dictionaries, where each dictionary contains the 'text' of the transcript 
                and its associated 'start' time and 'duration'.
                Example:
                [
                    {'text': 'Hello world!', 'start': 0.0, 'duration': 2.0},
                    {'text': 'This is a transcript.', 'start': 2.0, 'duration': 3.0},
                    ...
                ]
            str: An error message if the transcript cannot be retrieved.

        Raises:
            Exception: If an error occurs while fetching the transcript.
        """
    # Enhanced regex to handle different YouTube URL formats
    match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", video_url)
    if match:
        video_id = match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

    try:
        
        # Get the transcript of the video
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript_list
    
    except Exception as e:
        raise Exception(f"An error occurred while fetching the video details: {e}")
    
def get_youtube_transcript(video_id):
    """
    Retrieves the transcript of a YouTube video and returns it as a single string with sentences.

    Args:
        video_id (str): The YouTube video ID for which to retrieve the transcript.

    Returns:
        str: A single string containing the transcript with sentences. 
             The function attempts to maintain sentence structure by adding periods where necessary.
        str: An error message if the transcript cannot be retrieved.

    Raises:
        Exception: If an error occurs while fetching the transcript.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Join sentences with a space, adding a period at the end of each sentence if needed
        transcript_text = " ".join(
            [line["text"].strip() + "." if not line["text"].endswith('.') else line["text"].strip()
             for line in transcript_list]
        )
        return transcript_text
    except Exception as e:
        raise Exception(f"An error occurred while fetching the video details: {e}")