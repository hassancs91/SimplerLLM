"""
Video and audio extraction utilities.
"""
import os
import tempfile
import re
from typing import Optional, Tuple
from pathlib import Path


def is_youtube_url(url: str) -> bool:
    """
    Check if a string is a YouTube URL.

    Args:
        url: The string to check

    Returns:
        True if it's a YouTube URL, False otherwise
    """
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?(www\.)?youtu\.be/',
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    audio_format: str = "mp3",
    verbose: bool = False
) -> str:
    """
    Extract audio from a video file using moviepy.

    Args:
        video_path: Path to the video file
        output_path: Optional output path for the audio file. If None, creates a temp file.
        audio_format: Audio format (mp3, wav, etc.)
        verbose: Whether to print progress

    Returns:
        Path to the extracted audio file

    Raises:
        ImportError: If moviepy is not installed
        FileNotFoundError: If video file doesn't exist
        Exception: If audio extraction fails
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. "
            "Install it with: pip install moviepy"
        )

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output path if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_audio_{os.getpid()}.{audio_format}"
        output_path = os.path.join(temp_dir, temp_filename)

    try:
        if verbose:
            print(f"Extracting audio from {video_path}...")

        # Load video and extract audio
        video = VideoFileClip(video_path)
        audio = video.audio

        if audio is None:
            raise Exception(f"No audio track found in video: {video_path}")

        # Write audio file
        audio.write_audiofile(
            output_path,
            verbose=verbose,
            logger='bar' if verbose else None
        )

        # Close clips to free resources
        audio.close()
        video.close()

        if verbose:
            print(f"Audio extracted to: {output_path}")

        return output_path

    except Exception as e:
        # Clean up output file if it was created
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise Exception(f"Failed to extract audio from video: {str(e)}")


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds

    Raises:
        ImportError: If moviepy is not installed
        FileNotFoundError: If video file doesn't exist
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. "
            "Install it with: pip install moviepy"
        )

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        return duration
    except Exception as e:
        raise Exception(f"Failed to get video duration: {str(e)}")


def download_youtube_audio(
    youtube_url: str,
    output_path: Optional[str] = None,
    verbose: bool = False
) -> Tuple[str, float]:
    """
    Download audio from a YouTube video.

    Args:
        youtube_url: YouTube video URL
        output_path: Optional output path for the audio file
        verbose: Whether to print progress

    Returns:
        Tuple of (audio_file_path, duration_in_seconds)

    Raises:
        ImportError: If required packages are not installed
        Exception: If download fails
    """
    try:
        from moviepy.editor import AudioFileClip
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required for YouTube audio download. "
            "Install it with: pip install yt-dlp"
        )

    if output_path is None:
        temp_dir = tempfile.gettempdir()
        temp_filename = f"youtube_audio_{os.getpid()}.mp3"
        output_path = os.path.join(temp_dir, temp_filename)

    try:
        if verbose:
            print(f"Downloading audio from YouTube: {youtube_url}")

        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path.replace('.mp3', ''),
            'quiet': not verbose,
        }

        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            duration = info.get('duration', 0)

        if verbose:
            print(f"Audio downloaded to: {output_path}")

        return output_path, duration

    except Exception as e:
        # Clean up output file if it was created
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise Exception(f"Failed to download YouTube audio: {str(e)}")


def cleanup_temp_file(file_path: str, verbose: bool = False) -> None:
    """
    Clean up a temporary file.

    Args:
        file_path: Path to the file to delete
        verbose: Whether to print status
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            if verbose:
                print(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to clean up temp file {file_path}: {str(e)}")
