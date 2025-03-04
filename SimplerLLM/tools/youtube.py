import os
import re
import requests
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv(override=True) 

class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float

class Transcript(BaseModel):
    segments: List[TranscriptSegment]

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
        api_key = os.getenv("SEARCHAPI_API_KEY")
        url = "https://www.searchapi.io/api/v1/search"
        params = {
        "engine": "youtube_transcripts",
        "video_id": video_id,
        "api_key": api_key
        }

        response = requests.get(url, params=params)
        response.raise_for_status()  
        api_response = response.json()
        
        transcript = api_response["transcripts"]  
        transcript_list = Transcript(segments=transcript)  
        #transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        return transcript_list

    except Exception as e:
        raise Exception(f"An error occurred while fetching the video details: {e}")
    
def get_youtube_transcript(video_url):
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
    match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", video_url)
    if match:
        video_id = match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")
    
    try:
        api_key = os.getenv("SEARCHAPI_API_KEY")
        url = "https://www.searchapi.io/api/v1/search"
        params = {
        "engine": "youtube_transcripts",
        "video_id": video_id,
        "api_key": api_key
        }

        response = requests.get(url, params=params)
        response.raise_for_status()  
        api_response = response.json()
        
        transcript = api_response["transcripts"]  
        transcript_list = Transcript(segments=transcript)  
        #transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        transcript_text = " ".join(
            [segment.text.strip() + "." if not segment.text.endswith('.') else segment.text.strip()
             for segment in transcript_list.segments]
        )

        return transcript_text
    except Exception as e:
        raise Exception(f"An error occurred while fetching the video details: {e}")