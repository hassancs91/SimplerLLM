"""
Base classes and enums for Realtime Voice API.
Follows SimplerLLM's factory pattern for provider management.
"""

from enum import Enum
from typing import Optional, Callable, Any, Dict
from .models import RealtimeSessionConfig


class RealtimeVoiceProvider(Enum):
    """
    Enum for supported Realtime Voice API providers.
    Supports OpenAI Realtime API and ElevenLabs Conversational AI.
    """
    OPENAI = 1
    ELEVENLABS = 2


class RealtimeVoice:
    """
    Base class for Realtime Voice API implementations.
    Uses factory pattern to create provider-specific instances.

    Example:
        >>> from SimplerLLM import RealtimeVoice, RealtimeVoiceProvider
        >>> realtime = RealtimeVoice.create(
        ...     provider=RealtimeVoiceProvider.OPENAI,
        ...     model="gpt-4o-realtime-preview-2024-10-01",
        ...     voice="alloy"
        ... )
    """

    def __init__(
        self,
        provider: RealtimeVoiceProvider,
        session_config: Optional[RealtimeSessionConfig] = None,
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize RealtimeVoice base class.

        Args:
            provider: The voice provider to use
            session_config: Configuration for the session
            api_key: API key for the provider (or use env var)
            verbose: Enable verbose logging
        """
        self.provider = provider
        self.session_config = session_config or RealtimeSessionConfig()
        self.api_key = api_key
        self.verbose = verbose
        self._event_handlers: Dict[str, list[Callable]] = {}

    @staticmethod
    def create(
        provider: RealtimeVoiceProvider,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = 0.8,
        session_config: Optional[RealtimeSessionConfig] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Factory method to create provider-specific RealtimeVoice instances.

        Args:
            provider: The voice provider to use
            model: Model name (e.g., "gpt-4o-realtime-preview-2024-10-01")
            voice: Voice to use (e.g., "alloy", "echo") OR voice_id for ElevenLabs
            instructions: System instructions for the model
            temperature: Sampling temperature (0.6-1.2)
            session_config: Full session configuration (overrides individual params)
            api_key: API key for the provider
            verbose: Enable verbose logging
            **kwargs: Additional provider-specific parameters
                For ElevenLabs:
                    - agent_id: Existing agent ID
                    - voice_id: Custom cloned voice ID (creates temp agent)

        Returns:
            Provider-specific RealtimeVoice instance

        Example (OpenAI):
            >>> realtime = RealtimeVoice.create(
            ...     provider=RealtimeVoiceProvider.OPENAI,
            ...     model="gpt-4o-realtime-preview-2024-10-01",
            ...     voice="alloy",
            ...     instructions="You are a helpful assistant."
            ... )

        Example (ElevenLabs with custom voice):
            >>> realtime = RealtimeVoice.create(
            ...     provider=RealtimeVoiceProvider.ELEVENLABS,
            ...     voice_id="your_cloned_voice_id",
            ...     model="gpt-4o-mini",
            ...     instructions="You are helpful"
            ... )
        """
        # Build session config from parameters if not provided
        if session_config is None:
            if provider == RealtimeVoiceProvider.ELEVENLABS:
                # ElevenLabs uses different config model
                from .models import ElevenLabsSessionConfig

                config_params = {}

                # Handle voice_id vs voice parameter
                if 'voice_id' in kwargs:
                    config_params['voice_id'] = kwargs.pop('voice_id')
                elif voice:
                    # If voice provided, treat as voice_id for ElevenLabs
                    config_params['voice_id'] = voice

                # Handle agent_id
                if 'agent_id' in kwargs:
                    config_params['agent_id'] = kwargs.pop('agent_id')

                # Common parameters
                if model:
                    config_params['model'] = model
                if instructions:
                    config_params['instructions'] = instructions
                if temperature:
                    config_params['temperature'] = temperature

                # Merge remaining kwargs
                config_params.update(kwargs)

                session_config = ElevenLabsSessionConfig(**config_params)

            else:
                # OpenAI and other providers
                config_params = {}
                if model:
                    config_params['model'] = model
                if voice:
                    config_params['voice'] = voice
                if instructions:
                    config_params['instructions'] = instructions
                if temperature:
                    config_params['temperature'] = temperature

                # Merge with any additional kwargs
                config_params.update(kwargs)
                session_config = RealtimeSessionConfig(**config_params)

        if provider == RealtimeVoiceProvider.OPENAI:
            from .wrappers.openai_wrapper import OpenAIRealtimeVoice
            return OpenAIRealtimeVoice(
                provider=provider,
                session_config=session_config,
                api_key=api_key,
                verbose=verbose
            )

        elif provider == RealtimeVoiceProvider.ELEVENLABS:
            from .wrappers.elevenlabs_wrapper import ElevenLabsRealtimeVoice
            return ElevenLabsRealtimeVoice(
                provider=provider,
                session_config=session_config,
                api_key=api_key,
                verbose=verbose
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def on(self, event_type: str, callback: Optional[Callable] = None):
        """
        Register an event handler for a specific event type.
        Supports both decorator and direct call patterns.

        Args:
            event_type: The event type to listen for (e.g., "response.audio.delta")
            callback: Callback function to invoke when event occurs (optional for decorator use)

        Example (Decorator):
            >>> @realtime.on("response.audio.delta")
            ... async def on_audio(event):
            ...     print(f"Received audio chunk")

        Example (Direct call):
            >>> realtime.on("response.audio.delta", my_callback_function)
        """
        if callback is None:
            # Decorator pattern: @realtime.on("event_type")
            def decorator(func: Callable):
                if event_type not in self._event_handlers:
                    self._event_handlers[event_type] = []
                self._event_handlers[event_type].append(func)
                return func
            return decorator
        else:
            # Direct call pattern: realtime.on("event_type", callback)
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(callback)

    def off(self, event_type: str, callback: Optional[Callable] = None):
        """
        Unregister an event handler.

        Args:
            event_type: The event type to stop listening for
            callback: Specific callback to remove (if None, removes all handlers for this event)
        """
        if callback is None:
            self._event_handlers.pop(event_type, None)
        elif event_type in self._event_handlers:
            self._event_handlers[event_type] = [
                cb for cb in self._event_handlers[event_type] if cb != callback
            ]

    async def _emit_event(self, event_type: str, *args, **kwargs):
        """
        Emit an event to all registered handlers.

        Args:
            event_type: The event type to emit
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        if event_type in self._event_handlers:
            for callback in self._event_handlers[event_type]:
                if callback:
                    # Handle both sync and async callbacks
                    import asyncio
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)

    # Abstract methods to be implemented by provider wrappers
    async def connect(self):
        """Establish connection to the Realtime API."""
        raise NotImplementedError("Subclass must implement connect()")

    async def disconnect(self):
        """Close connection to the Realtime API."""
        raise NotImplementedError("Subclass must implement disconnect()")

    async def send_audio(self, audio_data: bytes):
        """Send audio data to the API."""
        raise NotImplementedError("Subclass must implement send_audio()")

    async def send_text(self, text: str):
        """Send text message to the API."""
        raise NotImplementedError("Subclass must implement send_text()")

    async def start_conversation(
        self,
        enable_microphone: bool = True,
        auto_play_audio: bool = True
    ):
        """Start a conversation session with automatic audio handling."""
        raise NotImplementedError("Subclass must implement start_conversation()")

    async def stop_conversation(self):
        """Stop the current conversation session."""
        raise NotImplementedError("Subclass must implement stop_conversation()")

    # Sync wrappers for convenience
    def connect_sync(self):
        """Synchronous version of connect()."""
        import asyncio
        return asyncio.run(self.connect())

    def disconnect_sync(self):
        """Synchronous version of disconnect()."""
        import asyncio
        return asyncio.run(self.disconnect())

    def send_audio_sync(self, audio_data: bytes):
        """Synchronous version of send_audio()."""
        import asyncio
        return asyncio.run(self.send_audio(audio_data))

    def send_text_sync(self, text: str):
        """Synchronous version of send_text()."""
        import asyncio
        return asyncio.run(self.send_text(text))
