"""
RealtimeVoiceChat - Complete voice chat implementation using OpenAI Realtime API.

Integrates AudioRecorder and AudioPlayer for full microphone-to-speaker voice conversations.
"""

import asyncio
import base64
from typing import Optional
from pydantic import BaseModel

from .base import RealtimeVoice
from ..live_voice_chat.audio_recorder import AudioRecorder
from ..live_voice_chat.audio_player import AudioPlayer
from SimplerLLM.utils.custom_verbose import verbose_print


class RealtimeVoiceChatConfig(BaseModel):
    """
    Configuration for RealtimeVoiceChat.
    """
    # Audio settings
    sample_rate: int = 24000
    chunk_duration: float = 0.2  # 200ms chunks

    # Streaming settings
    enable_microphone: bool = True
    auto_play_audio: bool = True

    # Interaction settings
    push_to_talk: bool = False  # True = hold key, False = continuous
    push_to_talk_key: str = "space"

    # Display settings
    show_transcripts: bool = True
    show_user_transcripts: bool = True
    verbose: bool = False


class RealtimeVoiceChat:
    """
    Complete voice chat using OpenAI Realtime API with audio I/O.

    This class integrates:
    - RealtimeVoice (OpenAI Realtime API)
    - AudioRecorder (microphone input)
    - AudioPlayer (speaker output)

    Example:
        >>> from SimplerLLM import (
        ...     RealtimeVoice, RealtimeVoiceProvider,
        ...     AudioRecorder, AudioPlayer,
        ...     RealtimeVoiceChat, RealtimeVoiceChatConfig
        ... )
        >>>
        >>> # Create components
        >>> realtime = RealtimeVoice.create(
        ...     provider=RealtimeVoiceProvider.OPENAI,
        ...     voice="alloy"
        ... )
        >>> recorder = AudioRecorder(sample_rate=24000)
        >>> player = AudioPlayer()
        >>>
        >>> # Create voice chat
        >>> config = RealtimeVoiceChatConfig(show_transcripts=True)
        >>> chat = RealtimeVoiceChat(realtime, recorder, player, config)
        >>>
        >>> # Start conversation
        >>> await chat.start_conversation()
    """

    def __init__(
        self,
        realtime_voice: RealtimeVoice,
        audio_recorder: AudioRecorder,
        audio_player: AudioPlayer,
        config: Optional[RealtimeVoiceChatConfig] = None
    ):
        """
        Initialize RealtimeVoiceChat.

        Args:
            realtime_voice: RealtimeVoice instance (connected to OpenAI)
            audio_recorder: AudioRecorder instance for microphone
            audio_player: AudioPlayer instance for speakers
            config: Configuration (uses defaults if None)
        """
        self.realtime = realtime_voice
        self.recorder = audio_recorder
        self.player = audio_player
        self.config = config or RealtimeVoiceChatConfig()

        self._running = False
        self._audio_task = None

        if self.config.verbose:
            verbose_print("RealtimeVoiceChat initialized", "info")

    async def start_conversation(self):
        """
        Start a voice conversation session.

        This will:
        1. Connect to the Realtime API
        2. Start audio recorder stream (if enabled)
        3. Start audio player stream (if enabled)
        4. Set up event handlers
        5. Begin bidirectional audio streaming

        The conversation continues until stop_conversation() is called.
        """
        if self._running:
            if self.config.verbose:
                verbose_print("Conversation already running", "warning")
            return

        if self.config.verbose:
            verbose_print("Starting conversation...", "info")

        # Connect to API
        if not self.realtime._provider.connected:
            await self.realtime.connect()

        # Set up event handlers
        self._setup_event_handlers()

        # Start audio streams
        if self.config.auto_play_audio:
            self.player.start_stream(sample_rate=self.config.sample_rate)
            if self.config.verbose:
                verbose_print("Audio player stream started", "info")

        if self.config.enable_microphone:
            self.recorder.start_stream(
                sample_rate=self.config.sample_rate,
                chunk_duration=self.config.chunk_duration
            )
            if self.config.verbose:
                verbose_print("Audio recorder stream started", "info")

            # Start audio streaming task
            self._audio_task = asyncio.create_task(self._audio_streaming_loop())

        self._running = True

        if self.config.verbose:
            verbose_print(
                "Conversation started! Speak into your microphone.",
                "info"
            )

    async def stop_conversation(self):
        """
        Stop the voice conversation session.

        This will:
        1. Stop audio streaming
        2. Stop audio recorder and player
        3. Disconnect from Realtime API (optional)
        """
        if not self._running:
            if self.config.verbose:
                verbose_print("Conversation not running", "warning")
            return

        if self.config.verbose:
            verbose_print("Stopping conversation...", "info")

        self._running = False

        # Cancel audio streaming task
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None

        # Stop audio streams
        if self.recorder.is_streaming():
            self.recorder.stop_stream()
            if self.config.verbose:
                verbose_print("Audio recorder stopped", "info")

        if self.player.is_stream_active():
            self.player.stop_stream()
            if self.config.verbose:
                verbose_print("Audio player stopped", "info")

        if self.config.verbose:
            verbose_print("Conversation stopped", "info")

    async def _audio_streaming_loop(self):
        """
        Main loop for streaming audio from microphone to API.
        Runs in background task.
        """
        try:
            while self._running and self.recorder.is_streaming():
                # Read audio chunk from microphone
                audio_chunk = self.recorder.read_chunk()

                # Send to Realtime API
                await self.realtime.send_audio(audio_chunk)

                # Small delay to prevent busy loop
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            if self.config.verbose:
                verbose_print("Audio streaming loop cancelled", "debug")
            raise

        except Exception as e:
            if self.config.verbose:
                verbose_print(f"Error in audio streaming loop: {e}", "error")
            raise

    def _setup_event_handlers(self):
        """Set up event handlers for Realtime API events."""

        # Audio output
        if self.config.auto_play_audio:
            @self.realtime.on("response.audio.delta")
            async def on_audio_delta(event):
                """Play audio chunks as they arrive."""
                audio_base64 = event.get("delta")
                if audio_base64 and self.player.is_stream_active():
                    audio_bytes = base64.b64decode(audio_base64)
                    self.player.play_chunk(audio_bytes)

        # Assistant transcript (audio output text)
        if self.config.show_transcripts:
            @self.realtime.on("response.audio_transcript.delta")
            async def on_transcript_delta(event):
                """Display assistant's speech as text."""
                delta = event.get("delta", "")
                if delta:
                    print(delta, end="", flush=True)

            @self.realtime.on("response.audio_transcript.done")
            async def on_transcript_done(event):
                """Add newline after assistant finishes speaking."""
                print()  # New line

        # User transcript (audio input text)
        if self.config.show_user_transcripts:
            @self.realtime.on("conversation.item.input_audio_transcription.completed")
            async def on_user_transcript(event):
                """Display user's speech as text."""
                transcript = event.get("transcript", "")
                if transcript and self.config.verbose:
                    verbose_print(f"You: {transcript}", "info")

        # Speech detection events
        if self.config.verbose:
            @self.realtime.on("input_audio_buffer.speech_started")
            async def on_speech_started(event):
                """User started speaking."""
                verbose_print("Speech started", "debug")

            @self.realtime.on("input_audio_buffer.speech_stopped")
            async def on_speech_stopped(event):
                """User stopped speaking."""
                verbose_print("Speech stopped", "debug")

        # Response events
        if self.config.verbose:
            @self.realtime.on("response.created")
            async def on_response_created(event):
                """Assistant started generating response."""
                verbose_print("Assistant responding...", "debug")

            @self.realtime.on("response.done")
            async def on_response_done(event):
                """Assistant finished response."""
                response = event.get("response", {})
                usage = response.get("usage", {})
                tokens = usage.get("total_tokens", 0)
                verbose_print(f"Response complete (tokens: {tokens})", "debug")

        # Error handling
        @self.realtime.on("error")
        async def on_error(event):
            """Handle API errors."""
            error = event.get("error", {})
            error_message = error.get("message", "Unknown error")
            if self.config.verbose:
                verbose_print(f"API Error: {error_message}", "error")

    def is_running(self) -> bool:
        """Check if conversation is currently active."""
        return self._running

    async def __aenter__(self):
        """Context manager support."""
        await self.start_conversation()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        await self.stop_conversation()
