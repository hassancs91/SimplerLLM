"""
Audio timing synchronization for video dubbing.

This module provides utilities to adjust audio speed to match original video timing.
"""
import os
import tempfile
from typing import Optional, Tuple


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration in seconds

    Raises:
        ImportError: If pydub is not installed
        Exception: If unable to get duration
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio processing. "
            "Install it with: pip install pydub"
        )

    try:
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # Convert milliseconds to seconds
        return duration
    except Exception as e:
        raise Exception(f"Failed to get audio duration: {str(e)}")


def adjust_audio_speed(
    audio_path: str,
    target_duration: float,
    output_path: Optional[str] = None,
    speed_range: Tuple[float, float] = (0.75, 1.5),
    verbose: bool = False
) -> Tuple[str, float]:
    """
    Adjust audio speed to match target duration.

    Args:
        audio_path: Path to the input audio file
        target_duration: Target duration in seconds
        output_path: Optional output path. If None, creates a temp file.
        speed_range: Min and max speed adjustment factors (min, max)
        verbose: Whether to print progress

    Returns:
        Tuple of (output_audio_path, speed_factor)
        speed_factor: The speed multiplier applied (e.g., 1.2 = 20% faster)

    Raises:
        ImportError: If required libraries are not installed
        ValueError: If speed adjustment is outside allowed range
        Exception: If audio processing fails
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import speedup
    except ImportError:
        raise ImportError(
            "pydub is required for audio speed adjustment. "
            "Install it with: pip install pydub"
        )

    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        original_duration = len(audio) / 1000.0  # Convert to seconds

        # Calculate required speed factor
        speed_factor = original_duration / target_duration

        if verbose:
            print(f"Original duration: {original_duration:.2f}s")
            print(f"Target duration: {target_duration:.2f}s")
            print(f"Speed factor: {speed_factor:.2f}x")

        # Check if speed adjustment is within allowed range
        min_speed, max_speed = speed_range
        if speed_factor < min_speed or speed_factor > max_speed:
            if verbose:
                print(f"Warning: Speed factor {speed_factor:.2f}x is outside allowed range [{min_speed}, {max_speed}]")
            # Clamp to allowed range
            speed_factor = max(min_speed, min(max_speed, speed_factor))
            if verbose:
                print(f"Clamping to: {speed_factor:.2f}x")

        # Apply speed adjustment if needed
        if abs(speed_factor - 1.0) > 0.01:  # Only adjust if difference is significant
            # Method 1: Using speedup (changes both speed and pitch)
            # This is simpler but changes pitch
            if speed_factor > 1.0:
                # Speed up audio
                adjusted_audio = speedup(audio, playback_speed=speed_factor)
            else:
                # Slow down audio
                # Pydub doesn't have a slowdown function, so we use frame_rate manipulation
                adjusted_audio = audio._spawn(
                    audio.raw_data,
                    overrides={
                        "frame_rate": int(audio.frame_rate * speed_factor)
                    }
                )
                adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)
        else:
            adjusted_audio = audio
            if verbose:
                print("No speed adjustment needed")

        # Create output path if not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            temp_filename = f"adjusted_audio_{os.getpid()}_{hash(audio_path)}.mp3"
            output_path = os.path.join(temp_dir, temp_filename)

        # Export adjusted audio
        adjusted_audio.export(output_path, format="mp3")

        if verbose:
            final_duration = len(adjusted_audio) / 1000.0
            print(f"Adjusted audio duration: {final_duration:.2f}s")
            print(f"Saved to: {output_path}")

        return output_path, speed_factor

    except Exception as e:
        raise Exception(f"Failed to adjust audio speed: {str(e)}")


def adjust_audio_speed_advanced(
    audio_path: str,
    target_duration: float,
    output_path: Optional[str] = None,
    speed_range: Tuple[float, float] = (0.75, 1.5),
    preserve_pitch: bool = True,
    verbose: bool = False
) -> Tuple[str, float]:
    """
    Advanced audio speed adjustment with pitch preservation (requires ffmpeg).

    This method uses ffmpeg's atempo filter to change speed while preserving pitch.
    This results in more natural-sounding audio but requires ffmpeg to be installed.

    Args:
        audio_path: Path to the input audio file
        target_duration: Target duration in seconds
        output_path: Optional output path. If None, creates a temp file.
        speed_range: Min and max speed adjustment factors (min, max)
        preserve_pitch: Whether to preserve pitch (requires ffmpeg)
        verbose: Whether to print progress

    Returns:
        Tuple of (output_audio_path, speed_factor)

    Raises:
        ImportError: If ffmpeg is not installed
        Exception: If audio processing fails
    """
    if not preserve_pitch:
        # Fall back to basic speed adjustment
        return adjust_audio_speed(
            audio_path,
            target_duration,
            output_path,
            speed_range,
            verbose
        )

    try:
        import subprocess
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio processing. "
            "Install it with: pip install pydub"
        )

    try:
        # Get original duration
        audio = AudioSegment.from_file(audio_path)
        original_duration = len(audio) / 1000.0

        # Calculate speed factor
        speed_factor = original_duration / target_duration

        if verbose:
            print(f"Original duration: {original_duration:.2f}s")
            print(f"Target duration: {target_duration:.2f}s")
            print(f"Speed factor: {speed_factor:.2f}x")

        # Check if speed adjustment is within allowed range
        min_speed, max_speed = speed_range
        if speed_factor < min_speed or speed_factor > max_speed:
            if verbose:
                print(f"Warning: Speed factor {speed_factor:.2f}x is outside allowed range [{min_speed}, {max_speed}]")
            speed_factor = max(min_speed, min(max_speed, speed_factor))

        # Create output path if not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            temp_filename = f"adjusted_audio_{os.getpid()}_{hash(audio_path)}.mp3"
            output_path = os.path.join(temp_dir, temp_filename)

        # Use ffmpeg atempo filter for pitch-preserving speed change
        # atempo only supports 0.5-2.0 range, so chain multiple filters if needed
        if abs(speed_factor - 1.0) > 0.01:
            atempo_filters = []
            remaining_factor = speed_factor

            # Chain atempo filters to achieve desired speed
            while remaining_factor > 2.0:
                atempo_filters.append("atempo=2.0")
                remaining_factor /= 2.0
            while remaining_factor < 0.5:
                atempo_filters.append("atempo=0.5")
                remaining_factor /= 0.5
            if abs(remaining_factor - 1.0) > 0.01:
                atempo_filters.append(f"atempo={remaining_factor:.4f}")

            filter_string = ",".join(atempo_filters)

            # Run ffmpeg command
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-filter:a", filter_string,
                "-y",  # Overwrite output
                output_path
            ]

            if not verbose:
                cmd.extend(["-loglevel", "error"])

            subprocess.run(cmd, check=True)

            if verbose:
                adjusted_audio = AudioSegment.from_file(output_path)
                final_duration = len(adjusted_audio) / 1000.0
                print(f"Adjusted audio duration: {final_duration:.2f}s")
                print(f"Saved to: {output_path}")
        else:
            # No adjustment needed, just copy
            import shutil
            shutil.copy(audio_path, output_path)
            if verbose:
                print("No speed adjustment needed")

        return output_path, speed_factor

    except FileNotFoundError:
        raise Exception(
            "ffmpeg is required for pitch-preserving speed adjustment. "
            "Install it from https://ffmpeg.org/ or use preserve_pitch=False"
        )
    except Exception as e:
        raise Exception(f"Failed to adjust audio speed with pitch preservation: {str(e)}")


def merge_audio_segments(
    audio_files: list,
    output_path: str,
    pause_duration: float = 0.0,
    verbose: bool = False
) -> str:
    """
    Merge multiple audio files into a single file with optional pauses.

    Args:
        audio_files: List of paths to audio files
        output_path: Path to save the merged audio
        pause_duration: Duration of pause between segments in seconds
        verbose: Whether to print progress

    Returns:
        Path to the merged audio file

    Raises:
        ImportError: If pydub is not installed
        Exception: If merging fails
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio merging. "
            "Install it with: pip install pydub"
        )

    try:
        if verbose:
            print(f"Merging {len(audio_files)} audio segments...")

        # Load first audio file
        combined = AudioSegment.from_file(audio_files[0])

        # Add remaining files with pauses
        pause = AudioSegment.silent(duration=int(pause_duration * 1000))  # Convert to ms

        for audio_file in audio_files[1:]:
            if pause_duration > 0:
                combined += pause
            combined += AudioSegment.from_file(audio_file)

        # Export merged audio
        combined.export(output_path, format="mp3")

        if verbose:
            duration = len(combined) / 1000.0
            print(f"Merged audio duration: {duration:.2f}s")
            print(f"Saved to: {output_path}")

        return output_path

    except Exception as e:
        raise Exception(f"Failed to merge audio segments: {str(e)}")
