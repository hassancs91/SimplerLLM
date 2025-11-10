import os
from typing import Optional
from ..voice_chat import VoiceChat, VoiceTurnResult
from .models import LiveVoiceChatConfig
from .audio_recorder import AudioRecorder
from .audio_player import AudioPlayer
from SimplerLLM.utils.custom_verbose import verbose_print


class LiveVoiceChat:
    """
    Live voice chat using microphone input and automatic audio playback.

    Wraps VoiceChat to provide real-time voice conversation with
    push-to-talk recording and automatic response playback.

    Example:
        ```python
        from SimplerLLM import STT, LLM, TTS, STTProvider, LLMProvider, TTSProvider
        from SimplerLLM import LiveVoiceChat

        # Create components
        stt = STT.create(provider=STTProvider.OPENAI)
        llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")
        tts = TTS.create(provider=TTSProvider.OPENAI, voice="nova")

        # Create live chat
        chat = LiveVoiceChat(stt, llm, tts)
        chat.start_session()

        print("Press SPACE to talk, release to send. Ctrl+C to quit.")

        # Single interaction
        result = chat.listen_and_respond()
        ```
    """

    def __init__(
        self,
        stt_instance,
        llm_instance,
        tts_instance,
        config: Optional[LiveVoiceChatConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize LiveVoiceChat with microphone support.

        Args:
            stt_instance: STT instance (created via STT.create())
            llm_instance: LLM instance (can be LLM, ReliableLLM, etc.)
            tts_instance: TTS instance (created via TTS.create())
            config: Optional LiveVoiceChatConfig (uses defaults if not provided)
            verbose: Enable verbose logging
        """
        self.config = config or LiveVoiceChatConfig()
        self.verbose = verbose

        # Create VoiceChat instance (wraps existing functionality)
        self.voice_chat = VoiceChat(
            stt_instance,
            llm_instance,
            tts_instance,
            config=self.config,  # LiveVoiceChatConfig extends VoiceChatConfig
            verbose=verbose
        )

        # Create audio recorder
        self.recorder = AudioRecorder(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.audio_dtype,
            verbose=verbose
        )

        # Create audio player
        self.player = AudioPlayer(verbose=verbose)

        if self.verbose:
            verbose_print(f"LiveVoiceChat initialized - Session: {self.voice_chat.session_id}", "info")
            verbose_print(f"Push-to-talk key: {self.config.push_to_talk_key}", "debug")
            verbose_print(f"Auto-play: {self.config.auto_play_response}", "debug")

    def start_session(self):
        """
        Start a new live voice chat session.

        Initializes the conversation with the system prompt.
        """
        self.voice_chat.start_session()

        if self.verbose:
            verbose_print("Live voice chat session started", "info")

    def listen_and_respond(
        self,
        key: Optional[str] = None,
        auto_play: Optional[bool] = None
    ) -> VoiceTurnResult:
        """
        Single voice interaction: record → process → play.

        Workflow:
        1. Press and hold key to record from microphone
        2. Release key to stop recording
        3. Transcribe audio (STT)
        4. Generate response (LLM)
        5. Synthesize speech (TTS)
        6. Play response audio (if auto_play=True)

        Args:
            key: Key for push-to-talk (None = use config default)
            auto_play: Auto-play response (None = use config default)

        Returns:
            VoiceTurnResult with transcription, response, and audio paths

        Example:
            ```python
            chat.start_session()
            result = chat.listen_and_respond()

            print(f"You said: {result.user_text}")
            print(f"AI said: {result.assistant_text}")
            ```
        """
        key = key or self.config.push_to_talk_key
        auto_play = auto_play if auto_play is not None else self.config.auto_play_response

        try:
            # Step 1: Record from microphone
            if self.verbose:
                verbose_print(f"Ready to record. Press and hold '{key}' to speak...", "info")

            audio_file = self.recorder.record_audio(
                key=key,
                max_duration=self.config.max_recording_duration,
                output_path=self._get_temp_audio_path() if self.config.temp_audio_dir else None
            )

            # Step 2: Process via VoiceChat (STT → LLM → TTS)
            result = self.voice_chat.chat_turn(audio_file, save_audio=False)

            # Step 3: Play response audio
            if auto_play and result.assistant_audio_path:
                if self.verbose:
                    verbose_print("Playing response...", "info")

                self.player.play(
                    result.assistant_audio_path,
                    volume=self.config.playback_volume,
                    wait=True
                )

            # Step 4: Cleanup temp recording file
            if self.config.cleanup_temp_files and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    if self.verbose:
                        verbose_print(f"Cleaned up temp file: {audio_file}", "debug")
                except Exception as e:
                    if self.verbose:
                        verbose_print(f"Could not delete temp file: {e}", "warning")

            return result

        except KeyboardInterrupt:
            if self.verbose:
                verbose_print("Recording interrupted", "info")
            raise
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error in listen_and_respond: {e}", "error")
            raise

    async def listen_and_respond_async(
        self,
        key: Optional[str] = None,
        auto_play: Optional[bool] = None
    ) -> VoiceTurnResult:
        """
        Async version of listen_and_respond.

        Args:
            key: Key for push-to-talk
            auto_play: Auto-play response

        Returns:
            VoiceTurnResult
        """
        key = key or self.config.push_to_talk_key
        auto_play = auto_play if auto_play is not None else self.config.auto_play_response

        try:
            # Record from microphone (sync - no async version needed)
            if self.verbose:
                verbose_print(f"Ready to record. Press and hold '{key}' to speak...", "info")

            audio_file = self.recorder.record_audio(
                key=key,
                max_duration=self.config.max_recording_duration,
                output_path=self._get_temp_audio_path() if self.config.temp_audio_dir else None
            )

            # Process via VoiceChat async
            result = await self.voice_chat.chat_turn_async(audio_file, save_audio=False)

            # Play response audio
            if auto_play and result.assistant_audio_path:
                if self.verbose:
                    verbose_print("Playing response...", "info")

                self.player.play(
                    result.assistant_audio_path,
                    volume=self.config.playback_volume,
                    wait=True
                )

            # Cleanup
            if self.config.cleanup_temp_files and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except:
                    pass

            return result

        except KeyboardInterrupt:
            if self.verbose:
                verbose_print("Recording interrupted", "info")
            raise
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error in async listen_and_respond: {e}", "error")
            raise

    def start_live_session(
        self,
        continuous: bool = True,
        stop_phrase: Optional[str] = None,
        key: Optional[str] = None
    ):
        """
        Start continuous conversation loop.

        Keeps listening and responding until interrupted or stop phrase detected.

        Args:
            continuous: If True, loop indefinitely until interrupted
            stop_phrase: Optional phrase to stop (e.g., "goodbye", "exit")
            key: Key for push-to-talk

        Example:
            ```python
            chat.start_live_session(
                continuous=True,
                stop_phrase="goodbye",
                key='space'
            )
            ```
        """
        self.start_session()

        if self.verbose:
            verbose_print("Starting continuous live session", "info")
            verbose_print("Press Ctrl+C to stop", "debug")
            if stop_phrase:
                verbose_print(f"Say '{stop_phrase}' to exit", "debug")

        turn_count = 0

        while continuous:
            try:
                turn_count += 1

                if self.verbose:
                    verbose_print(f"\n--- Turn {turn_count} ---", "info")

                result = self.listen_and_respond(key=key)

                # Check for stop phrase
                if stop_phrase and stop_phrase.lower() in result.user_text.lower():
                    if self.verbose:
                        verbose_print(f"Stop phrase '{stop_phrase}' detected. Ending session.", "info")
                    break

            except KeyboardInterrupt:
                if self.verbose:
                    verbose_print("\nSession interrupted by user", "info")
                break
            except Exception as e:
                if self.verbose:
                    verbose_print(f"Error in live session: {e}", "error")
                # Continue to next turn instead of crashing
                continue

        # End session
        session = self.voice_chat.end_session()

        if self.verbose:
            verbose_print(
                f"Live session ended - {turn_count} turns completed",
                "info"
            )

        return session

    def _get_temp_audio_path(self) -> str:
        """Generate path for temporary audio file."""
        import tempfile
        if self.config.temp_audio_dir:
            os.makedirs(self.config.temp_audio_dir, exist_ok=True)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.wav',
                prefix='recording_',
                dir=self.config.temp_audio_dir
            )
        else:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='recording_')

        os.close(temp_fd)
        return temp_path

    def stop_playback(self):
        """Stop currently playing audio."""
        self.player.stop()

    def set_volume(self, volume: float):
        """
        Set playback volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.config.playback_volume = max(0.0, min(1.0, volume))
        self.player.set_volume(self.config.playback_volume)

        if self.verbose:
            verbose_print(f"Volume set to {self.config.playback_volume:.2f}", "debug")

    def get_conversation_history(self):
        """Get conversation history from underlying VoiceChat."""
        return self.voice_chat.get_conversation_history()

    def clear_history(self):
        """Clear conversation history."""
        self.voice_chat.clear_history()

    def get_turn_count(self) -> int:
        """Get total number of turns completed."""
        return self.voice_chat.get_turn_count()

    def end_session(self):
        """End session and return session data."""
        return self.voice_chat.end_session()

    def save_session(self, filepath: str):
        """Save session to JSON file."""
        self.voice_chat.save_session(filepath)

    def cleanup(self):
        """Cleanup resources."""
        self.player.cleanup()

        if self.verbose:
            verbose_print("LiveVoiceChat cleaned up", "debug")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
