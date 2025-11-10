"""
Audio merging utilities for combining dialogue audio files.

This module provides utilities to combine multiple audio files with
configurable pauses between them.

Supports two methods:
1. pydub (recommended for Python 3.12 and earlier)
2. ffmpeg direct (fallback for Python 3.13+)
"""

import os
import subprocess
import tempfile
from typing import List, Optional


def _get_ffmpeg_command():
    """Get the correct ffmpeg command for the current platform."""
    import shutil
    # Try to find ffmpeg in PATH
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    # Try common Windows locations
    common_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    # Fallback to just 'ffmpeg' and let subprocess handle it
    return 'ffmpeg'


def _simple_concatenate_mp3(
    file_paths: List[str],
    output_path: str,
    verbose: bool = False
) -> str:
    """
    Simple MP3 concatenation without pauses (fallback when ffmpeg not available).
    Note: This won't have pauses between files, but will create a working combined file.
    """
    if verbose:
        print("Using simple binary concatenation (no pauses - ffmpeg not found)...")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Concatenate MP3 files
    with open(output_path, 'wb') as outfile:
        for i, file_path in enumerate(file_paths):
            if verbose:
                print(f"  Adding file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            with open(file_path, 'rb') as infile:
                outfile.write(infile.read())

    if verbose:
        print(f"Combined audio created successfully (simple concatenation)")
        print(f"  Note: No pauses between lines (ffmpeg not available)")

    return output_path


def _merge_with_ffmpeg(
    file_paths: List[str],
    output_path: str,
    pause_duration: float = 0.5,
    verbose: bool = False
) -> str:
    """
    Merge audio files using ffmpeg directly (fallback method).
    Works with Python 3.13+ where pydub has compatibility issues.
    """
    if verbose:
        print("Using ffmpeg direct merging method...")

    ffmpeg_cmd = _get_ffmpeg_command()

    # Create a temporary file list for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name

        for i, file_path in enumerate(file_paths):
            # Write file path
            f.write(f"file '{os.path.abspath(file_path)}'\n")

            # Add silence between files (except after last file)
            if i < len(file_paths) - 1 and pause_duration > 0:
                # Create a silent audio file for pause
                silence_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name

                # Generate silence using ffmpeg
                subprocess.run([
                    ffmpeg_cmd, '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo',
                    '-t', str(pause_duration), '-q:a', '9', '-acodec', 'libmp3lame',
                    '-y', silence_file
                ], capture_output=True, check=True, shell=False)

                f.write(f"file '{os.path.abspath(silence_file)}'\n")

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Concatenate using ffmpeg
        result = subprocess.run([
            ffmpeg_cmd, '-f', 'concat', '-safe', '0', '-i', concat_file,
            '-c', 'copy', '-y', output_path
        ], capture_output=True, text=True, check=True, shell=False)

        if verbose:
            print(f"Combined audio created with ffmpeg successfully")

        return output_path

    finally:
        # Clean up temp files
        if os.path.exists(concat_file):
            os.unlink(concat_file)


def _merge_with_pydub(
    file_paths: List[str],
    output_path: str,
    pause_duration: float = 0.5,
    format: str = "mp3",
    verbose: bool = False
) -> str:
    """Merge audio files using pydub (preferred method for Python 3.12-)."""
    from pydub import AudioSegment

    if verbose:
        print("Using pydub merging method...")

    # Create silence segment for pauses
    pause_ms = int(pause_duration * 1000)
    silence = AudioSegment.silent(duration=pause_ms)

    # Load and combine audio files
    combined = None

    for i, file_path in enumerate(file_paths):
        if verbose:
            print(f"  Loading file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")

        audio = AudioSegment.from_file(file_path)

        if combined is None:
            combined = audio
        else:
            combined = combined + silence + audio

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    combined.export(output_path, format=format)

    if verbose:
        duration_seconds = len(combined) / 1000
        print(f"Combined audio created: {duration_seconds:.2f}s duration")

    return output_path


def merge_audio_files(
    file_paths: List[str],
    output_path: str,
    pause_duration: float = 0.5,
    format: str = "mp3",
    verbose: bool = False
) -> str:
    """
    Merge multiple audio files into a single file with pauses.

    Automatically selects the best merging method:
    - pydub (if available, for Python 3.12 and earlier)
    - ffmpeg direct (fallback for Python 3.13+ or if pydub unavailable)

    Args:
        file_paths: List of paths to audio files to merge
        output_path: Path for the output combined file
        pause_duration: Pause duration between files in seconds (default: 0.5)
        format: Output audio format (default: "mp3")
        verbose: Print progress information

    Returns:
        Path to the combined audio file

    Raises:
        FileNotFoundError: If any input file doesn't exist
        RuntimeError: If neither pydub nor ffmpeg are available
    """

    if not file_paths:
        raise ValueError("No audio files provided to merge")

    # Check all files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

    if verbose:
        print(f"Merging {len(file_paths)} audio files...")

    # Try pydub first (if available)
    try:
        return _merge_with_pydub(file_paths, output_path, pause_duration, format, verbose)
    except ImportError as e:
        if verbose:
            print(f"pydub not available (Python 3.13+ compatibility issue), falling back to ffmpeg direct method...")

    # Fallback to ffmpeg direct
    try:
        return _merge_with_ffmpeg(file_paths, output_path, pause_duration, verbose)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        if verbose:
            print(f"ffmpeg not available, using simple concatenation fallback...")

    # Final fallback: simple binary concatenation (MP3 only, no pauses)
    if format == "mp3":
        try:
            return _simple_concatenate_mp3(file_paths, output_path, verbose)
        except Exception as e:
            raise RuntimeError(
                f"Failed to merge audio files with all methods.\n"
                f"Error: {e}\n"
                f"For best results:\n"
                f"  - Python 3.13+: Install ffmpeg and add to PATH\n"
                f"  - Python 3.12 and earlier: Install pydub with 'pip install pydub'"
            )
    else:
        raise RuntimeError(
            f"Failed to merge audio files. Format '{format}' requires pydub or ffmpeg.\n"
            f"For Python 3.13+: Ensure ffmpeg is installed and in your PATH\n"
            f"For Python 3.12 and earlier: Install pydub with 'pip install pydub'"
        )


def add_silence(
    audio_path: str,
    output_path: str,
    duration: float = 1.0,
    position: str = "end",
    format: str = "mp3"
) -> str:
    """
    Add silence to an audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path for output file
        duration: Duration of silence in seconds
        position: Where to add silence ('start', 'end', or 'both')
        format: Output audio format

    Returns:
        Path to the output file

    Raises:
        ImportError: If pydub is not installed
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required. Install with: pip install pydub"
        )

    # Load audio
    audio = AudioSegment.from_file(audio_path)

    # Create silence
    silence_ms = int(duration * 1000)
    silence = AudioSegment.silent(duration=silence_ms)

    # Add silence
    if position == "start":
        result = silence + audio
    elif position == "end":
        result = audio + silence
    elif position == "both":
        result = silence + audio + silence
    else:
        raise ValueError(f"Invalid position: {position}. Use 'start', 'end', or 'both'")

    # Export
    result.export(output_path, format=format)

    return output_path


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Uses pydub if available, otherwise falls back to ffprobe.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    # Try pydub first
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000  # Convert ms to seconds
    except ImportError:
        pass

    # Fallback to ffprobe
    try:
        import shutil
        ffprobe_cmd = shutil.which('ffprobe') or 'ffprobe'
        result = subprocess.run(
            [ffprobe_cmd, '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            capture_output=True,
            text=True,
            check=True,
            shell=False
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        # If all else fails, estimate based on file size
        # For MP3 at 128kbps: ~16KB per second
        file_size = os.path.getsize(file_path)
        return file_size / 16000


def estimate_duration_from_bytes(audio_bytes: bytes, format: str = "mp3") -> Optional[float]:
    """
    Estimate audio duration from bytes.

    Args:
        audio_bytes: Audio data as bytes
        format: Audio format

    Returns:
        Estimated duration in seconds, or None if pydub not available
    """
    try:
        from pydub import AudioSegment
        import io

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
        return len(audio) / 1000
    except ImportError:
        # Rough estimate based on file size and bitrate
        # For mp3 at 128kbps: ~16KB per second
        return len(audio_bytes) / 16000
