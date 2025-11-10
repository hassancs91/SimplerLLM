import os
import time
import uuid
from typing import Optional, Union, List
from pathlib import Path
from datetime import datetime

from .models import (
    VoiceChatConfig,
    ConversationMessage,
    ConversationRole,
    VoiceTurnResult,
    VoiceChatSession
)
from .conversation import ConversationManager
from SimplerLLM.utils.custom_verbose import verbose_print


class VoiceChat:
    """
    Interactive voice chat combining STT, LLM, and TTS.

    Combines SimplerLLM modular components to create an easy-to-use
    voice conversation system.

    Features:
    - Turn-by-turn voice conversation
    - Automatic conversation history management
    - Audio file management
    - Session persistence
    - Provider-agnostic (works with any STT/LLM/TTS provider)

    Future extensibility:
    - Tool calling
    - RAG integration
    - Multi-provider routing

    Example:
        ```python
        from SimplerLLM import STT, LLM, TTS, STTProvider, LLMProvider, TTSProvider
        from SimplerLLM import VoiceChat

        # Create components
        stt = STT.create(provider=STTProvider.OPENAI)
        llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")
        tts = TTS.create(provider=TTSProvider.OPENAI, voice="nova")

        # Create voice chat
        chat = VoiceChat(stt, llm, tts)

        # Start conversation
        chat.start_session()
        result = chat.chat_turn("my_question.wav")
        print(f"You: {result.user_text}")
        print(f"AI: {result.assistant_text}")
        ```
    """

    def __init__(
        self,
        stt_instance,
        llm_instance,
        tts_instance,
        config: Optional[VoiceChatConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize VoiceChat with required components.

        Args:
            stt_instance: STT instance (created via STT.create())
            llm_instance: LLM instance (can be LLM, ReliableLLM, or any LLM-compatible instance)
            tts_instance: TTS instance (created via TTS.create())
            config: Optional configuration (uses defaults if not provided)
            verbose: Enable verbose logging
        """
        self.stt = stt_instance
        self.llm = llm_instance
        self.tts = tts_instance
        self.config = config or VoiceChatConfig()
        self.verbose = verbose

        # Session management
        self.session_id = str(uuid.uuid4())
        self.started_at = None
        self.conversation = ConversationManager(
            max_length=self.config.max_history_length,
            verbose=verbose
        )
        self.turns = []

        # Setup output directory if saving audio
        if self.config.save_audio:
            os.makedirs(self.config.output_dir, exist_ok=True)

        if self.verbose:
            verbose_print(f"VoiceChat initialized - Session: {self.session_id}", "info")
            verbose_print(
                f"Components - STT: {self.stt.provider.name}, "
                f"LLM: {self.llm.provider.name}, "
                f"TTS: {self.tts.provider.name}",
                "debug"
            )

    def start_session(self):
        """
        Start a new conversation session.

        Initializes the conversation with the system prompt.
        """
        self.started_at = datetime.now()
        self.conversation.add_message(
            role=ConversationRole.SYSTEM,
            content=self.config.system_prompt
        )

        if self.verbose:
            verbose_print("Voice chat session started", "info")
            verbose_print(f"System prompt: {self.config.system_prompt}", "debug")

    def chat_turn(
        self,
        audio_input: str,
        save_audio: Optional[bool] = None
    ) -> VoiceTurnResult:
        """
        Execute a single conversation turn.

        Workflow: Audio Input → STT → LLM → TTS → Audio Output

        Args:
            audio_input: Path to audio file
            save_audio: Override config.save_audio for this turn

        Returns:
            VoiceTurnResult with all turn data including:
            - Transcribed user text
            - Generated assistant text
            - Path to assistant audio response
            - Timing information

        Example:
            ```python
            result = chat.chat_turn("user_question.wav")
            print(f"You said: {result.user_text}")
            print(f"Assistant: {result.assistant_text}")
            # Play audio: result.assistant_audio_path
            ```
        """
        start_time = time.time()
        save_audio = save_audio if save_audio is not None else self.config.save_audio

        try:
            # 1. Speech to Text
            if self.verbose:
                verbose_print(f"Processing audio input: {audio_input}", "info")

            stt_start = time.time()
            user_text = self._transcribe_audio(audio_input)
            stt_duration = time.time() - stt_start

            if self.verbose:
                verbose_print(f"Transcribed [{stt_duration:.2f}s]: {user_text}", "info")

            # Save user audio if requested
            user_audio_path = None
            if save_audio:
                user_audio_path = self._save_turn_audio(audio_input, "user")

            # Add to conversation history
            self.conversation.add_message(
                role=ConversationRole.USER,
                content=user_text,
                audio_file=user_audio_path
            )

            # 2. Generate LLM Response
            if self.verbose:
                verbose_print("Generating assistant response...", "info")

            llm_start = time.time()
            assistant_text = self._generate_response()
            llm_duration = time.time() - llm_start

            if self.verbose:
                verbose_print(f"Generated [{llm_duration:.2f}s]: {assistant_text}", "info")

            # Add to conversation history
            self.conversation.add_message(
                role=ConversationRole.ASSISTANT,
                content=assistant_text
            )

            # 3. Text to Speech
            if self.verbose:
                verbose_print("Synthesizing speech...", "info")

            tts_start = time.time()
            assistant_audio_path = self._synthesize_speech(assistant_text)
            tts_duration = time.time() - tts_start

            if self.verbose:
                verbose_print(f"Synthesized [{tts_duration:.2f}s]: {assistant_audio_path}", "info")

            total_duration = time.time() - start_time

            # Create result
            result = VoiceTurnResult(
                user_audio_path=user_audio_path,
                user_text=user_text,
                assistant_text=assistant_text,
                assistant_audio_path=assistant_audio_path,
                stt_duration=stt_duration,
                llm_duration=llm_duration,
                tts_duration=tts_duration,
                total_duration=total_duration
            )

            self.turns.append(result)

            if self.verbose:
                verbose_print(
                    f"Turn completed in {total_duration:.2f}s "
                    f"(STT: {stt_duration:.2f}s, LLM: {llm_duration:.2f}s, TTS: {tts_duration:.2f}s)",
                    "info"
                )

            return result

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error in chat turn: {str(e)}", "error")

            result = VoiceTurnResult(
                user_text="",
                assistant_text="",
                total_duration=time.time() - start_time,
                error=str(e)
            )
            self.turns.append(result)
            raise

    async def chat_turn_async(
        self,
        audio_input: str,
        save_audio: Optional[bool] = None
    ) -> VoiceTurnResult:
        """
        Async version of chat_turn for concurrent operations.

        Args:
            audio_input: Path to audio file
            save_audio: Override config.save_audio for this turn

        Returns:
            VoiceTurnResult with all turn data

        Example:
            ```python
            result = await chat.chat_turn_async("user_question.wav")
            ```
        """
        start_time = time.time()
        save_audio = save_audio if save_audio is not None else self.config.save_audio

        try:
            # 1. Speech to Text (async)
            if self.verbose:
                verbose_print(f"Processing audio input (async): {audio_input}", "info")

            stt_start = time.time()
            user_text = await self._transcribe_audio_async(audio_input)
            stt_duration = time.time() - stt_start

            if self.verbose:
                verbose_print(f"Transcribed [{stt_duration:.2f}s]: {user_text}", "info")

            # Save user audio if requested
            user_audio_path = None
            if save_audio:
                user_audio_path = self._save_turn_audio(audio_input, "user")

            # Add to conversation history
            self.conversation.add_message(
                role=ConversationRole.USER,
                content=user_text,
                audio_file=user_audio_path
            )

            # 2. Generate LLM Response (async)
            if self.verbose:
                verbose_print("Generating assistant response (async)...", "info")

            llm_start = time.time()
            assistant_text = await self._generate_response_async()
            llm_duration = time.time() - llm_start

            if self.verbose:
                verbose_print(f"Generated [{llm_duration:.2f}s]: {assistant_text}", "info")

            # Add to conversation history
            self.conversation.add_message(
                role=ConversationRole.ASSISTANT,
                content=assistant_text
            )

            # 3. Text to Speech (async)
            if self.verbose:
                verbose_print("Synthesizing speech (async)...", "info")

            tts_start = time.time()
            assistant_audio_path = await self._synthesize_speech_async(assistant_text)
            tts_duration = time.time() - tts_start

            if self.verbose:
                verbose_print(f"Synthesized [{tts_duration:.2f}s]: {assistant_audio_path}", "info")

            total_duration = time.time() - start_time

            # Create result
            result = VoiceTurnResult(
                user_audio_path=user_audio_path,
                user_text=user_text,
                assistant_text=assistant_text,
                assistant_audio_path=assistant_audio_path,
                stt_duration=stt_duration,
                llm_duration=llm_duration,
                tts_duration=tts_duration,
                total_duration=total_duration
            )

            self.turns.append(result)

            if self.verbose:
                verbose_print(f"Async turn completed in {total_duration:.2f}s", "info")

            return result

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error in async chat turn: {str(e)}", "error")

            result = VoiceTurnResult(
                user_text="",
                assistant_text="",
                total_duration=time.time() - start_time,
                error=str(e)
            )
            self.turns.append(result)
            raise

    def _transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio using STT."""
        params = {}
        if self.config.stt_language:
            params['language'] = self.config.stt_language
        if self.config.stt_model:
            params['model'] = self.config.stt_model

        result = self.stt.transcribe(audio_file, **params)
        return result if isinstance(result, str) else result.text

    async def _transcribe_audio_async(self, audio_file: str) -> str:
        """Async transcribe audio using STT."""
        params = {}
        if self.config.stt_language:
            params['language'] = self.config.stt_language
        if self.config.stt_model:
            params['model'] = self.config.stt_model

        result = await self.stt.transcribe_async(audio_file, **params)
        return result if isinstance(result, str) else result.text

    def _generate_response(self) -> str:
        """Generate LLM response with conversation history."""
        messages = self.conversation.get_messages_for_llm()

        params = {
            'messages': messages,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens
        }

        response = self.llm.generate_response(**params)
        return response

    async def _generate_response_async(self) -> str:
        """Async generate LLM response with conversation history."""
        messages = self.conversation.get_messages_for_llm()

        params = {
            'messages': messages,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens
        }

        response = await self.llm.generate_response_async(**params)
        return response

    def _synthesize_speech(self, text: str) -> str:
        """Synthesize speech and return audio file path."""
        # Ensure output directory exists (needed even if save_audio=False)
        os.makedirs(self.config.output_dir, exist_ok=True)

        turn_num = len(self.turns) + 1
        output_filename = f"assistant_{turn_num:03d}.mp3"
        output_path = os.path.join(self.config.output_dir, output_filename)

        params = {
            'text': text,
            'output_path': output_path
        }
        if self.config.tts_voice:
            params['voice'] = self.config.tts_voice
        if self.config.tts_speed:
            params['speed'] = self.config.tts_speed
        if self.config.tts_model:
            params['model'] = self.config.tts_model

        self.tts.generate_speech(**params)
        return output_path

    async def _synthesize_speech_async(self, text: str) -> str:
        """Async synthesize speech and return audio file path."""
        # Ensure output directory exists (needed even if save_audio=False)
        os.makedirs(self.config.output_dir, exist_ok=True)

        turn_num = len(self.turns) + 1
        output_filename = f"assistant_{turn_num:03d}.mp3"
        output_path = os.path.join(self.config.output_dir, output_filename)

        params = {
            'text': text,
            'output_path': output_path
        }
        if self.config.tts_voice:
            params['voice'] = self.config.tts_voice
        if self.config.tts_speed:
            params['speed'] = self.config.tts_speed
        if self.config.tts_model:
            params['model'] = self.config.tts_model

        await self.tts.generate_speech_async(**params)
        return output_path

    def _save_turn_audio(self, audio_path: str, role: str) -> str:
        """Copy input audio to output directory."""
        import shutil
        turn_num = len(self.turns) + 1
        filename = f"{role}_{turn_num:03d}{Path(audio_path).suffix}"
        dest_path = os.path.join(self.config.output_dir, filename)
        shutil.copy2(audio_path, dest_path)
        return dest_path

    def get_conversation_history(self) -> List[ConversationMessage]:
        """
        Get full conversation history.

        Returns:
            List of all conversation messages
        """
        return self.conversation.get_all_messages()

    def clear_history(self):
        """
        Clear conversation history (keeps system prompt).

        Useful for starting a new topic while keeping the same session.
        """
        self.conversation.clear()
        self.conversation.add_message(
            role=ConversationRole.SYSTEM,
            content=self.config.system_prompt
        )

        if self.verbose:
            verbose_print("Conversation history cleared", "info")

    def get_turn_count(self) -> int:
        """Get total number of turns completed."""
        return len(self.turns)

    def end_session(self) -> VoiceChatSession:
        """
        End session and return complete session data.

        Returns:
            VoiceChatSession with all session information
        """
        session = VoiceChatSession(
            session_id=self.session_id,
            config=self.config,
            conversation_history=self.get_conversation_history(),
            turns=self.turns,
            started_at=self.started_at,
            ended_at=datetime.now(),
            total_turns=len(self.turns),
            success=not any(turn.error for turn in self.turns)
        )

        if self.verbose:
            verbose_print(
                f"Session ended - {len(self.turns)} turns, "
                f"Duration: {(session.ended_at - session.started_at).total_seconds():.1f}s",
                "info"
            )

        return session

    def save_session(self, filepath: str):
        """
        Save session to JSON file.

        Args:
            filepath: Path to save session JSON

        Example:
            ```python
            chat.save_session("my_session.json")
            ```
        """
        session = self.end_session()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(session.model_dump_json(indent=2))

        if self.verbose:
            verbose_print(f"Session saved to {filepath}", "info")

    @classmethod
    def load_session(cls, filepath: str) -> VoiceChatSession:
        """
        Load session from JSON file.

        Args:
            filepath: Path to session JSON file

        Returns:
            VoiceChatSession object

        Example:
            ```python
            session = VoiceChat.load_session("my_session.json")
            print(f"Loaded session with {session.total_turns} turns")
            ```
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return VoiceChatSession.model_validate_json(f.read())
