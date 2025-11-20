"""
OpenAI Realtime Voice API user-facing wrapper.
Provides high-level interface for building voice agents.
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable, List
from ..base import RealtimeVoice, RealtimeVoiceProvider
from ..models import RealtimeSessionConfig
from ..providers.openai_realtime import OpenAIRealtimeProvider
from ..providers.realtime_response_models import RealtimeFullResponse
from SimplerLLM.utils.custom_verbose import verbose_print


class OpenAIRealtimeVoice(RealtimeVoice):
    """
    User-facing wrapper for OpenAI Realtime Voice API.
    Provides both low-level event-driven API and high-level conversation API.

    Example (Low-level):
        >>> realtime = OpenAIRealtimeVoice(...)
        >>> @realtime.on("response.audio.delta")
        ... async def on_audio(event):
        ...     audio_chunk = base64.b64decode(event.get("delta"))
        ...     # Play audio chunk
        >>> await realtime.connect()

    Example (High-level):
        >>> realtime = OpenAIRealtimeVoice(...)
        >>> await realtime.start_conversation(
        ...     enable_microphone=True,
        ...     auto_play_audio=True
        ... )
    """

    def __init__(
        self,
        provider: RealtimeVoiceProvider,
        session_config: RealtimeSessionConfig,
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize OpenAI Realtime Voice wrapper.

        Args:
            provider: The voice provider (should be OPENAI)
            session_config: Session configuration
            api_key: OpenAI API key
            verbose: Enable verbose logging
        """
        super().__init__(provider, session_config, api_key, verbose)

        # Initialize the provider
        self._provider = OpenAIRealtimeProvider(
            session_config=session_config,
            api_key=api_key,
            verbose=verbose
        )

        # Conversation state
        self._conversation_active = False
        self._audio_recorder = None
        self._audio_player = None

        # Function registry
        self._functions: Dict[str, Callable] = {}
        self._function_schemas: List[Dict[str, Any]] = []

    async def connect(self):
        """
        Establish connection to the OpenAI Realtime API.

        Raises:
            ConnectionError: If connection fails
        """
        await self._provider.connect()

        if self.verbose:
            verbose_print("Connected to OpenAI Realtime API", "info")

    async def disconnect(self):
        """Close connection to the API."""
        if self._conversation_active:
            await self.stop_conversation()

        await self._provider.disconnect()

        if self.verbose:
            verbose_print("Disconnected from OpenAI Realtime API", "info")

    async def send_audio(self, audio_data: bytes, commit: bool = False):
        """
        Send audio data to the API.

        Args:
            audio_data: Raw audio bytes (PCM16 24kHz by default)
            commit: Whether to commit the audio buffer after sending

        Note:
            In Server VAD mode, audio is automatically committed when speech ends.
            In manual mode, you must set commit=True or call commit_audio() manually.
        """
        await self._provider.send_audio_chunk(audio_data, commit=commit)

    async def send_text(self, text: str):
        """
        Send a text message to the conversation.

        Args:
            text: Text message to send

        Note:
            This will automatically trigger a response from the model.
        """
        await self._provider.send_text_message(text)

    async def commit_audio(self):
        """
        Manually commit the audio buffer.
        Only needed in manual turn detection mode.
        """
        await self._provider.commit_audio_buffer()

    async def clear_audio_buffer(self):
        """Clear the audio input buffer."""
        await self._provider.clear_audio_buffer()

    async def cancel_response(self):
        """Cancel the current in-progress response."""
        await self._provider.cancel_response()

    async def create_response(self, **kwargs):
        """
        Manually trigger a response.
        Only needed in manual turn detection mode.

        Args:
            **kwargs: Optional response config (instructions, temperature, etc.)
        """
        await self._provider.create_response(**kwargs)

    def on(self, event_type: str, callback: Optional[Callable] = None):
        """
        Register an event handler for low-level event access.
        Supports both decorator and direct call patterns.

        Args:
            event_type: Event type to listen for
                Common events:
                - "session.created"
                - "session.updated"
                - "conversation.item.created"
                - "input_audio_buffer.speech_started"
                - "input_audio_buffer.speech_stopped"
                - "response.created"
                - "response.audio.delta"
                - "response.audio_transcript.delta"
                - "response.text.delta"
                - "response.done"
                - "response.function_call_arguments.delta"
                - "error"
            callback: Callback function (sync or async), optional for decorator use

        Example (Decorator):
            >>> @realtime.on("response.audio.delta")
            ... async def on_audio(event):
            ...     audio_base64 = event.get("delta")
            ...     # Process audio...

        Example (Direct call):
            >>> realtime.on("response.audio.delta", my_callback_function)
        """
        if callback is None:
            # Decorator pattern: @realtime.on("event_type")
            def decorator(func: Callable):
                self._provider.on(event_type, func)
                # Also register with base class for consistency
                super(OpenAIRealtimeVoice, self).on(event_type, func)
                return func
            return decorator
        else:
            # Direct call pattern: realtime.on("event_type", callback)
            self._provider.on(event_type, callback)
            # Also register with base class for consistency
            super().on(event_type, callback)

    def off(self, event_type: str, callback: Optional[Callable] = None):
        """
        Unregister an event handler.

        Args:
            event_type: Event type
            callback: Specific callback to remove (if None, removes all)
        """
        self._provider.off(event_type, callback)
        super().off(event_type, callback)

    def add_function(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Add a function that the model can call.

        Args:
            name: Function name
            function: The callable function
            description: Description of what the function does
            parameters: JSON schema for function parameters

        Example:
            >>> def get_weather(location: str):
            ...     return f"Weather in {location}: 72°F, sunny"
            >>>
            >>> realtime.add_function(
            ...     name="get_weather",
            ...     function=get_weather,
            ...     description="Get current weather for a location",
            ...     parameters={
            ...         "type": "object",
            ...         "properties": {
            ...             "location": {
            ...                 "type": "string",
            ...                 "description": "City name"
            ...             }
            ...         },
            ...         "required": ["location"]
            ...     }
            ... )
        """
        self._functions[name] = function

        schema = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters
        }
        self._function_schemas.append(schema)

        # Update session config
        self.session_config.tools = self._function_schemas

        if self.verbose:
            verbose_print(f"Added function: {name}", "info")

    async def start_conversation(
        self,
        enable_microphone: bool = True,
        auto_play_audio: bool = True,
        vad_mode: str = "server"
    ):
        """
        Start a high-level conversation session with automatic audio handling.

        Args:
            enable_microphone: Whether to enable microphone input
            auto_play_audio: Whether to automatically play audio responses
            vad_mode: Voice activity detection mode ("server" or "manual")

        Note:
            This is a simplified API. For more control, use the low-level
            event API with connect() and manual event handlers.
        """
        if not self._provider.connected:
            await self.connect()

        self._conversation_active = True

        # Set up audio handlers if requested
        if auto_play_audio:
            await self._setup_audio_playback()

        if enable_microphone:
            await self._setup_microphone_input()

        # Set up function call handler if functions are registered
        if self._functions:
            self._provider.on("response.function_call_arguments.done", self._handle_function_call)

        if self.verbose:
            verbose_print(
                f"Conversation started (mic: {enable_microphone}, auto_play: {auto_play_audio}, vad: {vad_mode})",
                "info"
            )

    async def stop_conversation(self):
        """Stop the conversation session."""
        self._conversation_active = False

        # Clean up audio components
        if self._audio_recorder:
            # Stop recording (implementation depends on audio library)
            self._audio_recorder = None

        if self._audio_player:
            # Stop playback (implementation depends on audio library)
            self._audio_player = None

        if self.verbose:
            verbose_print("Conversation stopped", "info")

    async def _setup_audio_playback(self):
        """
        Set up automatic audio playback for responses.
        Uses accumulated audio chunks for seamless playback.
        """
        async def on_response_done(event):
            """Play accumulated audio when response completes."""
            try:
                audio_data = self._provider.get_accumulated_audio()
                if audio_data:
                    # Here you would integrate with AudioPlayer or similar
                    # For now, we'll just log
                    if self.verbose:
                        verbose_print(f"Received audio response: {len(audio_data)} bytes", "info")

                    # TODO: Integrate with SimplerLLM AudioPlayer
                    # from SimplerLLM.voice.live_voice_chat.audio_player import AudioPlayer
                    # self._audio_player.play_audio(audio_data)

            except Exception as e:
                if self.verbose:
                    verbose_print(f"Error playing audio: {e}", "error")

        self._provider.on("response.done", on_response_done)

    async def _setup_microphone_input(self):
        """
        Set up microphone input for continuous conversation.
        """
        # TODO: Integrate with SimplerLLM AudioRecorder
        # from SimplerLLM.voice.live_voice_chat.audio_recorder import AudioRecorder
        # This would stream microphone audio to send_audio()

        if self.verbose:
            verbose_print(
                "Microphone input setup (integration with AudioRecorder pending)",
                "warning"
            )

    async def _handle_function_call(self, event: Dict[str, Any]):
        """
        Handle function calls from the model.

        Args:
            event: The function_call_arguments.done event
        """
        try:
            item = event.get("item", {})
            call_id = item.get("call_id")
            function_name = item.get("name")
            arguments_json = item.get("arguments")

            if not all([call_id, function_name, arguments_json]):
                if self.verbose:
                    verbose_print("Incomplete function call data", "warning")
                return

            if function_name not in self._functions:
                if self.verbose:
                    verbose_print(f"Function not found: {function_name}", "error")
                return

            # Parse arguments
            import json
            arguments = json.loads(arguments_json)

            if self.verbose:
                verbose_print(f"Executing function: {function_name}({arguments})", "info")

            # Execute function
            start_time = time.time()
            function = self._functions[function_name]

            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(function):
                result = await function(**arguments)
            else:
                result = function(**arguments)

            execution_time = time.time() - start_time

            if self.verbose:
                verbose_print(
                    f"Function {function_name} completed in {execution_time:.2f}s",
                    "info"
                )

            # Send function output back to the model
            await self._provider.send_event("conversation.item.create", {
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(result)
                }
            })

            # Trigger response with function output
            await self._provider.create_response()

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error handling function call: {e}", "error")

    def get_accumulated_audio(self) -> bytes:
        """
        Get all accumulated audio from the current response.

        Returns:
            Concatenated audio data (PCM16 24kHz by default)
        """
        return self._provider.get_accumulated_audio()

    def get_accumulated_transcript(self) -> str:
        """
        Get the accumulated transcript from the current response.

        Returns:
            Transcript text
        """
        return self._provider.get_accumulated_transcript()

    # Sync wrappers (inherited from base, implemented here)
    def connect_sync(self):
        """Synchronous version of connect()."""
        return asyncio.run(self.connect())

    def disconnect_sync(self):
        """Synchronous version of disconnect()."""
        return asyncio.run(self.disconnect())

    def send_audio_sync(self, audio_data: bytes, commit: bool = False):
        """Synchronous version of send_audio()."""
        return asyncio.run(self.send_audio(audio_data, commit))

    def send_text_sync(self, text: str):
        """Synchronous version of send_text()."""
        return asyncio.run(self.send_text(text))
