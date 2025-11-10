"""
Video processing utilities for replacing audio tracks.
"""
import os
from typing import Optional


def replace_video_audio(
    video_path: str,
    new_audio_path: str,
    output_path: str,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    verbose: bool = False
) -> str:
    """
    Replace the audio track of a video file.

    Args:
        video_path: Path to the original video file
        new_audio_path: Path to the new audio file
        output_path: Path to save the output video
        video_codec: Video codec to use (default: libx264)
        audio_codec: Audio codec to use (default: aac)
        verbose: Whether to print progress

    Returns:
        Path to the output video file

    Raises:
        ImportError: If moviepy is not installed
        FileNotFoundError: If input files don't exist
        Exception: If video processing fails
    """
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. "
            "Install it with: pip install moviepy"
        )

    # Validate input files
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(new_audio_path):
        raise FileNotFoundError(f"Audio file not found: {new_audio_path}")

    try:
        if verbose:
            print(f"Loading video: {video_path}")
            print(f"Loading audio: {new_audio_path}")

        # Load video and new audio
        video = VideoFileClip(video_path)
        new_audio = AudioFileClip(new_audio_path)

        # Set the new audio to the video
        video_with_new_audio = video.set_audio(new_audio)

        if verbose:
            print(f"Writing output video to: {output_path}")

        # Write the output video
        video_with_new_audio.write_videofile(
            output_path,
            codec=video_codec,
            audio_codec=audio_codec,
            verbose=verbose,
            logger='bar' if verbose else None
        )

        # Close clips to free resources
        video.close()
        new_audio.close()
        video_with_new_audio.close()

        if verbose:
            print(f"Video successfully created: {output_path}")

        return output_path

    except Exception as e:
        raise Exception(f"Failed to replace video audio: {str(e)}")


def trim_video(
    video_path: str,
    output_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    verbose: bool = False
) -> str:
    """
    Trim a video to a specific time range.

    Args:
        video_path: Path to the video file
        output_path: Path to save the trimmed video
        start_time: Start time in seconds (None = start of video)
        end_time: End time in seconds (None = end of video)
        verbose: Whether to print progress

    Returns:
        Path to the trimmed video

    Raises:
        ImportError: If moviepy is not installed
        Exception: If trimming fails
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. "
            "Install it with: pip install moviepy"
        )

    try:
        if verbose:
            print(f"Loading video: {video_path}")

        video = VideoFileClip(video_path)

        # Trim video
        if start_time is not None or end_time is not None:
            trimmed = video.subclip(start_time, end_time)
        else:
            trimmed = video

        if verbose:
            print(f"Writing trimmed video to: {output_path}")

        # Write output
        trimmed.write_videofile(output_path, verbose=verbose, logger='bar' if verbose else None)

        # Close clips
        video.close()
        trimmed.close()

        return output_path

    except Exception as e:
        raise Exception(f"Failed to trim video: {str(e)}")


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video information:
        - duration: Duration in seconds
        - fps: Frames per second
        - size: (width, height) tuple
        - has_audio: Whether video has audio

    Raises:
        ImportError: If moviepy is not installed
        Exception: If unable to get video info
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. "
            "Install it with: pip install moviepy"
        )

    try:
        video = VideoFileClip(video_path)

        info = {
            'duration': video.duration,
            'fps': video.fps,
            'size': video.size,
            'has_audio': video.audio is not None,
        }

        video.close()

        return info

    except Exception as e:
        raise Exception(f"Failed to get video info: {str(e)}")


def extract_video_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    verbose: bool = False
) -> str:
    """
    Extract a segment from a video file.

    Args:
        video_path: Path to the video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save the segment
        verbose: Whether to print progress

    Returns:
        Path to the extracted segment

    Raises:
        ImportError: If moviepy is not installed
        Exception: If extraction fails
    """
    return trim_video(video_path, output_path, start_time, end_time, verbose)


def combine_video_clips(
    video_paths: list,
    output_path: str,
    method: str = "compose",
    verbose: bool = False
) -> str:
    """
    Combine multiple video clips into one.

    Args:
        video_paths: List of paths to video files
        output_path: Path to save the combined video
        method: Method to combine clips - "compose" or "concatenate"
        verbose: Whether to print progress

    Returns:
        Path to the combined video

    Raises:
        ImportError: If moviepy is not installed
        ValueError: If invalid method
        Exception: If combining fails
    """
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. "
            "Install it with: pip install moviepy"
        )

    if method not in ["compose", "concatenate"]:
        raise ValueError(f"method must be 'compose' or 'concatenate', got: {method}")

    try:
        if verbose:
            print(f"Loading {len(video_paths)} video clips...")

        # Load all video clips
        clips = [VideoFileClip(path) for path in video_paths]

        if method == "concatenate":
            # Concatenate clips one after another
            final_clip = concatenate_videoclips(clips)
        else:
            # Compose clips (overlay)
            final_clip = CompositeVideoClip(clips)

        if verbose:
            print(f"Writing combined video to: {output_path}")

        # Write output
        final_clip.write_videofile(output_path, verbose=verbose, logger='bar' if verbose else None)

        # Close all clips
        for clip in clips:
            clip.close()
        final_clip.close()

        return output_path

    except Exception as e:
        raise Exception(f"Failed to combine video clips: {str(e)}")
