"""
Audio utility functions for resampling between different sample rates.
Handles conversion between OpenAI (24kHz) and ElevenLabs (16kHz).
"""

import numpy as np
from typing import Optional
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def resample_audio(
    audio_data: bytes,
    source_rate: int,
    target_rate: int,
    dtype: str = 'int16'
) -> bytes:
    """
    Resample audio data from source sample rate to target sample rate.

    Args:
        audio_data: Raw PCM audio bytes
        source_rate: Source sample rate in Hz (e.g., 24000)
        target_rate: Target sample rate in Hz (e.g., 16000)
        dtype: Data type ('int16' or 'float32')

    Returns:
        Resampled audio as bytes

    Example:
        >>> # Resample from 24kHz to 16kHz
        >>> audio_16k = resample_audio(audio_24k, 24000, 16000)

    Note:
        Requires scipy for high-quality resampling.
        Falls back to simple decimation if scipy not available.
    """
    if source_rate == target_rate:
        return audio_data  # No resampling needed

    # Convert bytes to numpy array
    if dtype == 'int16':
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    elif dtype == 'float32':
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Resample
    if SCIPY_AVAILABLE:
        # High-quality resampling using scipy
        num_samples = int(len(audio_array) * target_rate / source_rate)
        resampled = signal.resample(audio_array, num_samples)

        # Convert back to original dtype
        if dtype == 'int16':
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        else:
            resampled = resampled.astype(np.float32)
    else:
        # Simple decimation/interpolation fallback
        ratio = target_rate / source_rate

        if ratio < 1:
            # Downsample (e.g., 24kHz -> 16kHz)
            step = int(1 / ratio)
            resampled = audio_array[::step]
        else:
            # Upsample (e.g., 16kHz -> 24kHz)
            resampled = np.repeat(audio_array, int(ratio))

        if dtype == 'int16':
            resampled = resampled.astype(np.int16)
        else:
            resampled = resampled.astype(np.float32)

    return resampled.tobytes()


def resample_24k_to_16k(audio_data: bytes) -> bytes:
    """
    Convenience function: Resample from 24kHz to 16kHz.

    Args:
        audio_data: PCM16 audio at 24kHz

    Returns:
        PCM16 audio at 16kHz

    Use case: Convert OpenAI audio (24kHz) to ElevenLabs format (16kHz)
    """
    return resample_audio(audio_data, 24000, 16000, dtype='int16')


def resample_16k_to_24k(audio_data: bytes) -> bytes:
    """
    Convenience function: Resample from 16kHz to 24kHz.

    Args:
        audio_data: PCM16 audio at 16kHz

    Returns:
        PCM16 audio at 24kHz

    Use case: Convert ElevenLabs audio (16kHz) to OpenAI format (24kHz)
    """
    return resample_audio(audio_data, 16000, 24000, dtype='int16')


class AudioResampler:
    """
    Stateful audio resampler for streaming applications.

    Maintains internal state for seamless resampling of audio chunks.
    Useful when you need to resample continuous audio streams.

    Example:
        >>> resampler = AudioResampler(source_rate=24000, target_rate=16000)
        >>> for chunk in audio_chunks:
        ...     resampled_chunk = resampler.resample(chunk)
        ...     # Process resampled_chunk
    """

    def __init__(self, source_rate: int, target_rate: int, dtype: str = 'int16'):
        """
        Initialize audio resampler.

        Args:
            source_rate: Source sample rate in Hz
            target_rate: Target sample rate in Hz
            dtype: Data type ('int16' or 'float32')
        """
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.dtype = dtype
        self.ratio = target_rate / source_rate

        # Buffer for leftover samples
        self._buffer = np.array([], dtype=np.int16 if dtype == 'int16' else np.float32)

        if not SCIPY_AVAILABLE:
            print(
                "WARNING: scipy not available. Using simple resampling. "
                "For best quality, install scipy: pip install scipy"
            )

    def resample(self, audio_data: bytes) -> bytes:
        """
        Resample audio chunk.

        Args:
            audio_data: Raw PCM audio bytes

        Returns:
            Resampled audio as bytes
        """
        if self.source_rate == self.target_rate:
            return audio_data

        # Convert to numpy array
        if self.dtype == 'int16':
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_array = np.frombuffer(audio_data, dtype=np.float32)

        # Append to buffer
        audio_array = np.concatenate([self._buffer, audio_array])

        # Resample
        if SCIPY_AVAILABLE:
            num_samples = int(len(audio_array) * self.ratio)
            resampled = signal.resample(audio_array, num_samples)

            if self.dtype == 'int16':
                resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
            else:
                resampled = resampled.astype(np.float32)

            # Clear buffer (scipy handles state internally)
            self._buffer = np.array([], dtype=audio_array.dtype)
        else:
            # Simple resampling
            if self.ratio < 1:
                step = int(1 / self.ratio)
                resampled = audio_array[::step]
                # Save remainder for next chunk
                remainder_start = (len(audio_array) // step) * step
                self._buffer = audio_array[remainder_start:]
            else:
                resampled = np.repeat(audio_array, int(self.ratio))
                self._buffer = np.array([], dtype=audio_array.dtype)

            if self.dtype == 'int16':
                resampled = resampled.astype(np.int16)
            else:
                resampled = resampled.astype(np.float32)

        return resampled.tobytes()

    def reset(self):
        """Reset internal buffer."""
        self._buffer = np.array([], dtype=np.int16 if self.dtype == 'int16' else np.float32)
