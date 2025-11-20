import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import os
from typing import Optional
from pynput import keyboard
from SimplerLLM.utils.custom_verbose import verbose_print


class AudioRecorder:
    """
    Record audio from microphone using sounddevice.

    Supports push-to-talk mode where recording starts when a key is pressed
    and stops when the key is released.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = 'int16',
        verbose: bool = False
    ):
        """
        Initialize AudioRecorder.

        Args:
            sample_rate: Sample rate in Hz (16000 recommended for STT)
            channels: Number of audio channels (1=mono, 2=stereo)
            dtype: Data type for audio samples ('int16' or 'float32')
            verbose: Enable verbose logging
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.verbose = verbose

        # Recording state (for file-based recording)
        self.is_recording = False
        self.recording_data = []
        self.stream = None

        # Streaming state (for realtime streaming)
        self._streaming = False
        self._input_stream = None
        self._stream_sample_rate = None
        self._chunk_size = None

        if self.verbose:
            verbose_print(
                f"AudioRecorder initialized - {sample_rate}Hz, {channels} channel(s)",
                "info"
            )

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream."""
        if status and self.verbose:
            verbose_print(f"Audio callback status: {status}", "warning")

        if self.is_recording:
            # Copy audio data to recording buffer
            self.recording_data.append(indata.copy())

    def record_audio(
        self,
        key: str = 'space',
        max_duration: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Record audio using push-to-talk (press and hold key).

        Args:
            key: Key to use for push-to-talk (e.g., 'space', 'ctrl')
            max_duration: Maximum recording duration in seconds (None = unlimited)
            output_path: Path to save WAV file (None = temp file)

        Returns:
            Path to recorded WAV file

        Example:
            ```python
            recorder = AudioRecorder()
            audio_file = recorder.record_audio(key='space')
            print(f"Recorded to: {audio_file}")
            ```
        """
        if self.verbose:
            verbose_print(f"Ready to record. Press and hold '{key}' to start...", "info")

        # Reset recording state
        self.recording_data = []
        self.is_recording = False
        key_pressed = False

        # Keyboard listener for push-to-talk
        def on_press(k):
            nonlocal key_pressed
            try:
                if hasattr(k, 'name') and k.name == key:
                    key_pressed = True
                    if not self.is_recording:
                        self.is_recording = True
                        if self.verbose:
                            verbose_print("Recording started...", "info")
                elif k == keyboard.KeyCode.from_char(key):
                    key_pressed = True
                    if not self.is_recording:
                        self.is_recording = True
                        if self.verbose:
                            verbose_print("Recording started...", "info")
            except AttributeError:
                pass

        def on_release(k):
            nonlocal key_pressed
            try:
                if hasattr(k, 'name') and k.name == key:
                    key_pressed = False
                    self.is_recording = False
                    if self.verbose:
                        verbose_print("Recording stopped", "info")
                    return False  # Stop listener
                elif k == keyboard.KeyCode.from_char(key):
                    key_pressed = False
                    self.is_recording = False
                    if self.verbose:
                        verbose_print("Recording stopped", "info")
                    return False  # Stop listener
            except AttributeError:
                pass

        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback
        ):
            # Start keyboard listener
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()  # Wait for key release

        if not self.recording_data:
            if self.verbose:
                verbose_print("No audio data recorded", "warning")
            raise ValueError("No audio data recorded. Please press and hold the key while speaking.")

        # Combine all recorded chunks
        audio_data = np.concatenate(self.recording_data, axis=0)

        # Trim to max_duration if specified
        if max_duration:
            max_frames = int(max_duration * self.sample_rate)
            audio_data = audio_data[:max_frames]

        # Save to file
        if output_path is None:
            # Create temp file
            temp_fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='recording_')
            os.close(temp_fd)  # Close file descriptor

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Write WAV file
        wavfile.write(output_path, self.sample_rate, audio_data)

        if self.verbose:
            duration = len(audio_data) / self.sample_rate
            size_kb = os.path.getsize(output_path) / 1024
            verbose_print(
                f"Saved recording: {output_path} ({duration:.2f}s, {size_kb:.1f}KB)",
                "info"
            )

        return output_path

    def record_duration(
        self,
        duration: float,
        output_path: Optional[str] = None
    ) -> str:
        """
        Record audio for a fixed duration (no key press required).

        Args:
            duration: Recording duration in seconds
            output_path: Path to save WAV file (None = temp file)

        Returns:
            Path to recorded WAV file

        Example:
            ```python
            recorder = AudioRecorder()
            audio_file = recorder.record_duration(5.0)  # Record for 5 seconds
            ```
        """
        if self.verbose:
            verbose_print(f"Recording for {duration} seconds...", "info")

        # Record for specified duration
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype
        )
        sd.wait()  # Wait for recording to finish

        # Save to file
        if output_path is None:
            temp_fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='recording_')
            os.close(temp_fd)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        wavfile.write(output_path, self.sample_rate, audio_data)

        if self.verbose:
            size_kb = os.path.getsize(output_path) / 1024
            verbose_print(f"Saved recording: {output_path} ({duration}s, {size_kb:.1f}KB)", "info")

        return output_path

    @staticmethod
    def list_devices():
        """
        List available audio input devices.

        Returns:
            List of device information
        """
        return sd.query_devices()

    @staticmethod
    def get_default_device():
        """Get default input device."""
        return sd.query_devices(kind='input')

    # ========================================================================
    # Streaming methods for Realtime API
    # ========================================================================

    def start_stream(
        self,
        sample_rate: Optional[int] = None,
        chunk_duration: float = 0.2
    ):
        """
        Start continuous audio streaming (for Realtime API).

        This method enables low-latency streaming of audio chunks instead of
        recording to files. Useful for real-time voice applications.

        Args:
            sample_rate: Sample rate in Hz (None = use instance sample_rate)
                        For OpenAI Realtime API, use 24000 Hz
            chunk_duration: Duration of each chunk in seconds (default 0.2 = 200ms)

        Example:
            ```python
            recorder = AudioRecorder(sample_rate=24000)
            recorder.start_stream(chunk_duration=0.2)

            while recording:
                chunk = recorder.read_chunk()  # Get raw audio bytes
                # Send chunk to API

            recorder.stop_stream()
            ```
        """
        if self._streaming:
            if self.verbose:
                verbose_print("Stream already started", "warning")
            return

        # Use provided sample rate or fall back to instance sample rate
        self._stream_sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        self._chunk_size = int(self._stream_sample_rate * chunk_duration)

        if self.verbose:
            verbose_print(
                f"Starting audio stream: {self._stream_sample_rate}Hz, "
                f"{chunk_duration}s chunks ({self._chunk_size} frames)",
                "info"
            )

        # Create input stream
        self._input_stream = sd.InputStream(
            samplerate=self._stream_sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self._chunk_size
        )

        self._input_stream.start()
        self._streaming = True

        if self.verbose:
            verbose_print("Audio stream started successfully", "info")

    def read_chunk(self) -> bytes:
        """
        Read one chunk of audio from the stream.

        Returns:
            Raw audio bytes (PCM16 format)

        Raises:
            RuntimeError: If stream not started

        Example:
            ```python
            recorder.start_stream(sample_rate=24000)
            audio_chunk = recorder.read_chunk()  # bytes object
            ```
        """
        if not self._streaming:
            raise RuntimeError(
                "Stream not started. Call start_stream() first."
            )

        # Read audio data (blocking until data available)
        audio_data, overflowed = self._input_stream.read(self._chunk_size)

        if overflowed and self.verbose:
            verbose_print("Audio input overflow - some data may be lost", "warning")

        # Convert numpy array to bytes
        return audio_data.tobytes()

    def stop_stream(self):
        """
        Stop the audio stream and cleanup resources.

        Example:
            ```python
            recorder.start_stream()
            # ... streaming operations ...
            recorder.stop_stream()
            ```
        """
        if not self._streaming:
            if self.verbose:
                verbose_print("Stream not running", "warning")
            return

        if self.verbose:
            verbose_print("Stopping audio stream...", "info")

        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None

        self._streaming = False
        self._stream_sample_rate = None
        self._chunk_size = None

        if self.verbose:
            verbose_print("Audio stream stopped", "info")

    def is_streaming(self) -> bool:
        """
        Check if audio stream is currently active.

        Returns:
            True if streaming, False otherwise
        """
        return self._streaming
