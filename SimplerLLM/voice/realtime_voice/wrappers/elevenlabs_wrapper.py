"""
ElevenLabs Conversational AI user-facing wrapper.
Provides high-level interface for building voice agents with custom voices.
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable, List
from ..base import RealtimeVoice, RealtimeVoiceProvider
from ..models import ElevenLabsSessionConfig
from ..providers.elevenlabs_convai import ElevenLabsRealtimeProvider
from SimplerLLM.utils.custom_verbose import verbose_print


class ElevenLabsRealtimeVoice(RealtimeVoice):
    """
    User-facing wrapper for ElevenLabs Conversational AI.
    Supports custom voice cloning and provides OpenAI-compatible API.

    Example (Existing Agent):
        >>> realtime = ElevenLabsRealtimeVoice(
        ...     session_config=ElevenLabsSessionConfig(
        ...         agent_id="your_agent_id_from_dashboard"
        ...     )
        ... )
        >>> await realtime.connect()
        >>> await realtime.send_audio(audio_bytes)

    Example (Dynamic Agent with Custom Voice):
        >>> realtime = ElevenLabsRealtimeVoice(
        ...     session_config=ElevenLabsSessionConfig(
        ...         voice_id="your_cloned_voice_id",
        ...         model="gpt-4o-mini",
        ...         instructions="You are a helpful assistant"
        ...     ),
        ...     api_key="your_elevenlabs_api_key"
        ... )
        >>> await realtime.connect()  # Creates agent automatically
    """

    def __init__(
        self,
        provider: RealtimeVoiceProvider,
        session_config: ElevenLabsSessionConfig,
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize ElevenLabs Realtime Voice wrapper.

        Args:
            provider: The voice provider (should be ELEVENLABS)
            session_config: Session configuration
            api_key: ElevenLabs API key
            verbose: Enable verbose logging
        """
        super().__init__(provider, session_config, api_key, verbose)

        # Initialize the provider
        self._provider = ElevenLabsRealtimeProvider(
            session_config=session_config,
            api_key=api_key,
            verbose=verbose
        )

        # Conversation state
        self._conversation_active = False

        # Function registry (for client tools)
        self._functions: Dict[str, Callable] = {}
        self._function_schemas: List[Dict[str, Any]] = []

        # Setup automatic function call handling
        self._setup_function_handler()

    async def connect(self):
        """
        Establish connection to ElevenLabs Conversational AI.

        If voice_id is provided (instead of agent_id), this will automatically
        create a temporary agent with your custom cloned voice.

        Raises:
            ConnectionError: If connection fails
        """
        await self._provider.connect()

        if self.verbose:
            verbose_print("Connected to ElevenLabs Conversational AI", "info")

    async def disconnect(self):
        """
        Close connection to the API.

        If a temporary agent was created, it will be automatically deleted.
        """
        if self._conversation_active:
            await self.stop_conversation()

        await self._provider.disconnect()

        if self.verbose:
            verbose_print("Disconnected from ElevenLabs", "info")

    async def send_audio(self, audio_data: bytes):
        """
        Send audio data to ElevenLabs.

        Args:
            audio_data: Raw audio bytes (PCM16 16kHz recommended for ElevenLabs)

        Note:
            ElevenLabs uses automatic Voice Activity Detection (VAD) with
            sophisticated turn-taking. Audio is automatically processed when
            the user stops speaking.
        """
        await self._provider.send_audio_chunk(audio_data)

    async def send_text(self, text: str):
        """
        Send a text message to the conversation.

        Args:
            text: Text message to send

        Note:
            This will automatically trigger a response from the agent.
        """
        await self._provider.send_text_message(text)

    def on(self, event_type: str, callback: Optional[Callable] = None):
        """
        Register an event handler for event access.
        Supports both decorator and direct call patterns.

        Args:
            event_type: Event type to listen for (OpenAI-compatible or ElevenLabs-specific)
            callback: Callback function (optional for decorator use)

        OpenAI-Compatible Events:
            - response.audio.delta - Audio response chunks
            - response.audio_transcript.delta/done - AI's text response
            - conversation.item.input_audio_transcription.completed - User's transcript
            - input_audio_buffer.speech_started - User started speaking
            - error - Error events

        ElevenLabs-Specific Events:
            - elevenlabs.interruption - User interrupted the agent
            - elevenlabs.client_tool_call - Agent requests client-side function
            - elevenlabs.agent_response - Full agent text response

        Example:
            >>> @realtime.on("response.audio.delta")
            ... async def on_audio(event):
            ...     audio_bytes = base64.b64decode(event.get("delta"))
            ...     player.play_chunk(audio_bytes)
        """
        if callback is None:
            # Decorator pattern
            def decorator(func: Callable):
                self._provider.on(event_type, func)
                return func
            return decorator
        else:
            # Direct call
            self._provider.on(event_type, callback)

    def add_function(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Register a client-side function that the agent can call.

        Args:
            name: Function name
            function: Callable function to execute
            description: Description of what the function does
            parameters: JSON schema for function parameters

        Note:
            Functions must also be configured in the ElevenLabs agent settings.
            This only handles the client-side execution.

        Example:
            >>> def get_weather(location: str) -> str:
            ...     return f"It's sunny in {location}"
            >>>
            >>> realtime.add_function(
            ...     name="get_weather",
            ...     function=get_weather,
            ...     description="Get current weather",
            ...     parameters={
            ...         "type": "object",
            ...         "properties": {
            ...             "location": {"type": "string"}
            ...         },
            ...         "required": ["location"]
            ...     }
            ... )
        """
        self._functions[name] = function
        self._function_schemas.append({
            "name": name,
            "description": description,
            "parameters": parameters
        })

        if self.verbose:
            verbose_print(f"Registered function: {name}", "info")

    async def start_conversation(
        self,
        enable_microphone: bool = True,
        auto_play_audio: bool = True
    ):
        """
        Start a high-level conversation session.

        Note:
            For full voice chat with microphone and speaker support, use
            RealtimeVoiceChat class instead. This method is for basic setup.

        Args:
            enable_microphone: Enable microphone input
            auto_play_audio: Automatically play audio responses

        Example:
            >>> await realtime.start_conversation()
            >>> # Manually send audio or text
            >>> await realtime.send_text("Hello!")
        """
        if not self._provider.connected:
            await self.connect()

        self._conversation_active = True

        if self.verbose:
            verbose_print("Conversation started", "info")

    async def stop_conversation(self):
        """Stop the conversation session."""
        self._conversation_active = False

        if self.verbose:
            verbose_print("Conversation stopped", "info")

    def _setup_function_handler(self):
        """
        Setup automatic handler for client tool calls from ElevenLabs.
        When the agent calls a function, this executes it and returns the result.
        """
        @self._provider.on("client_tool_call")
        async def handle_function_call(event: Dict[str, Any]):
            """Handle function calls from the agent."""
            tool_call_id = event.get("tool_call_id")
            tool_name = event.get("tool_name")
            parameters = event.get("parameters", {})

            if self.verbose:
                verbose_print(f"Agent called function: {tool_name}", "info")

            # Check if function is registered
            if tool_name not in self._functions:
                if self.verbose:
                    verbose_print(f"Function not found: {tool_name}", "warning")
                result = f"Error: Function '{tool_name}' not registered"
            else:
                # Execute the function
                try:
                    func = self._functions[tool_name]

                    # Call function (support both sync and async)
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**parameters)
                    else:
                        result = func(**parameters)

                    if self.verbose:
                        verbose_print(f"Function '{tool_name}' executed successfully", "info")

                except Exception as e:
                    if self.verbose:
                        verbose_print(f"Error executing '{tool_name}': {e}", "error")
                    result = f"Error: {str(e)}"

            # Send result back to ElevenLabs
            await self._provider.send_client_tool_result(tool_call_id, result)

    @property
    def agent_id(self) -> Optional[str]:
        """Get the current agent ID (useful when dynamically created)."""
        return self._provider._agent_id

    @property
    def is_connected(self) -> bool:
        """Check if connected to ElevenLabs."""
        return self._provider.connected

    @property
    def conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self._provider.conversation_id
