import newspaper
import re
from youtube_transcript_api import YouTubeTranscriptApi


#Load Files


##load content from URL - Like Blog Post Content
def read_blog_from_url(url):
    """
    Extracts the text content from a given URL using the newspaper package.

    Parameters:
    url (str): The URL of the article to extract text from.

    Returns:
    str: The text content of the article if extraction is successful, None otherwise.
    """
    try:
        article = newspaper.Article(url)
        article.download()

        if article.download_state == 2:
            article.parse()
            return article
        else:
            print(f"An error occurred while fetching the article")
            return None
    except newspaper.ArticleException as e:
        print(f"An error occurred while fetching the article: {e}")
        return None




def get_youtube_video_transcript(video_url):
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
        raise f"An error occurred while fetching the transcript: {e}"