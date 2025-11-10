import pygame
import os
import time
from typing import Optional
from SimplerLLM.utils.custom_verbose import verbose_print


class AudioPlayer:
    """
    Play audio files using pygame.

    Supports MP3, WAV, OGG formats with volume control and
    both blocking and non-blocking playback modes.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize AudioPlayer.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self._initialized = False

        if self.verbose:
            verbose_print("AudioPlayer initialized", "info")

    def _ensure_initialized(self):
        """Ensure pygame mixer is initialized."""
        if not self._initialized:
            try:
                pygame.mixer.init()
                self._initialized = True
                if self.verbose:
                    verbose_print("Pygame mixer initialized", "debug")
            except pygame.error as e:
                if self.verbose:
                    verbose_print(f"Error initializing pygame mixer: {e}", "error")
                raise

    def play(
        self,
        audio_path: str,
        volume: float = 1.0,
        wait: bool = True
    ):
        """
        Play an audio file.

        Args:
            audio_path: Path to audio file (MP3, WAV, OGG)
            volume: Playback volume (0.0 to 1.0)
            wait: If True, wait for playback to finish. If False, return immediately

        Example:
            ```python
            player = AudioPlayer()

            # Blocking playback
            player.play("response.mp3")

            # Non-blocking playback
            player.play("response.mp3", wait=False)
            ```
        """
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            if self.verbose:
                verbose_print(error_msg, "error")
            raise FileNotFoundError(error_msg)

        self._ensure_initialized()

        try:
            if self.verbose:
                size_kb = os.path.getsize(audio_path) / 1024
                verbose_print(f"Playing: {audio_path} ({size_kb:.1f}KB)", "info")

            # Load and play audio
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
            pygame.mixer.music.play()

            if wait:
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                if self.verbose:
                    verbose_print("Playback finished", "info")

        except pygame.error as e:
            error_msg = f"Error playing audio: {e}"
            if self.verbose:
                verbose_print(error_msg, "error")
            raise Exception(error_msg)

    def play_async(
        self,
        audio_path: str,
        volume: float = 1.0
    ):
        """
        Play audio file asynchronously (non-blocking).

        Args:
            audio_path: Path to audio file
            volume: Playback volume (0.0 to 1.0)
        """
        self.play(audio_path, volume=volume, wait=False)

    def stop(self):
        """Stop currently playing audio."""
        if self._initialized and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            if self.verbose:
                verbose_print("Playback stopped", "info")

    def pause(self):
        """Pause currently playing audio."""
        if self._initialized and pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            if self.verbose:
                verbose_print("Playback paused", "info")

    def unpause(self):
        """Resume paused audio."""
        if self._initialized:
            pygame.mixer.music.unpause()
            if self.verbose:
                verbose_print("Playback resumed", "info")

    def set_volume(self, volume: float):
        """
        Set playback volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self._initialized:
            pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
            if self.verbose:
                verbose_print(f"Volume set to {volume:.2f}", "debug")

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if self._initialized:
            return pygame.mixer.music.get_busy()
        return False

    def get_volume(self) -> float:
        """Get current volume level."""
        if self._initialized:
            return pygame.mixer.music.get_volume()
        return 0.0

    def cleanup(self):
        """Cleanup pygame mixer resources."""
        if self._initialized:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            self._initialized = False
            if self.verbose:
                verbose_print("AudioPlayer cleaned up", "debug")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._initialized:
            try:
                self.cleanup()
            except:
                pass  # Ignore errors during cleanup
