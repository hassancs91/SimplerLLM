"""
ElevenLabs Conversational AI provider implementation.
Handles WebSocket connection, agent management, event translation, and audio streaming.
"""

import asyncio
import json
import base64
import time
import os
from typing import Optional, Dict, Any, Callable, List
import websockets
from websockets.exceptions import ConnectionClosed
import httpx

from ..models import (
    ElevenLabsSessionConfig,
    RealtimeEvent,
    RealtimeError
)
from .realtime_response_models import (
    RealtimeFullResponse,
    RealtimeStreamChunk,
    RealtimeSessionInfo
)
from SimplerLLM.utils.custom_verbose import verbose_print


# Constants
ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


class ElevenLabsRealtimeProvider:
    """
    Core provider implementation for ElevenLabs Conversational AI.
    Manages WebSocket connection, agent lifecycle, and event handling.

    ElevenLabs uses an agent-based architecture:
    - Agents are pre-configured with voice, model, instructions
    - Connect to WebSocket using agent_id
    - Or dynamically create agents with voice_id
    """

    def __init__(
        self,
        session_config: ElevenLabsSessionConfig,
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize ElevenLabs Realtime provider.

        Args:
            session_config: Configuration for the session
            api_key: ElevenLabs API key (or use ELEVENLABS_API_KEY env var)
            verbose: Enable verbose logging
        """
        self.session_config = session_config
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.verbose = verbose

        # Validate configuration
        if not self.session_config.agent_id and not self.session_config.voice_id:
            raise ValueError(
                "Either agent_id or voice_id must be provided. "
                "Use agent_id for existing agents, or voice_id to create a new agent."
            )

        # If using voice_id (dynamic agent creation), API key is required
        if self.session_config.voice_id and not self.api_key:
            raise ValueError(
                "API key is required when using voice_id for dynamic agent creation. "
                "Set ELEVENLABS_API_KEY environment variable or pass api_key parameter."
            )

        # WebSocket connection
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self._receive_task: Optional[asyncio.Task] = None

        # Agent management
        self._created_agent_id: Optional[str] = None  # Track if we created an agent
        self._agent_id: Optional[str] = session_config.agent_id

        # Session state
        self.session_id: Optional[str] = None
        self.conversation_id: Optional[str] = None

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Response tracking
        self._audio_chunks: List[bytes] = []
        self._transcript_chunks: List[str] = []

        if self.verbose:
            verbose_print(
                f"Initialized ElevenLabs Provider with agent_id: {self._agent_id or 'dynamic'}",
                "info"
            )

    async def connect(self):
        """
        Establish WebSocket connection to ElevenLabs Conversational AI.

        Steps:
        1. Create agent if voice_id provided (instead of agent_id)
        2. Get signed URL if needed (private agents)
        3. Connect to WebSocket
        4. Start event listening

        Raises:
            ConnectionError: If connection fails after retries
        """
        if self.connected:
            if self.verbose:
                verbose_print("Already connected", "warning")
            return

        # Step 1: Create agent if needed
        if self.session_config.voice_id and not self._agent_id:
            if self.verbose:
                verbose_print(
                    f"Creating temporary agent with voice_id: {self.session_config.voice_id}",
                    "info"
                )
            self._agent_id = await self._create_agent()
            self._created_agent_id = self._agent_id

        # Step 2: Get WebSocket URL
        ws_url = await self._get_websocket_url()

        # Step 3: Connect with retries
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.verbose:
                    verbose_print(
                        f"Connecting to ElevenLabs (attempt {attempt}/{MAX_RETRIES})...",
                        "info"
                    )

                self.websocket = await websockets.connect(
                    ws_url,
                    extra_headers=self._get_headers() if not self.session_config.use_signed_url else {},
                    ping_interval=20,
                    ping_timeout=10
                )

                self.connected = True

                if self.verbose:
                    verbose_print("Connected to ElevenLabs Conversational AI", "info")

                # Start receiving events
                self._receive_task = asyncio.create_task(self._receive_events())

                return

            except Exception as e:
                if self.verbose:
                    verbose_print(
                        f"Connection attempt {attempt} failed: {e}",
                        "error"
                    )

                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** (attempt - 1))  # Exponential backoff
                    if self.verbose:
                        verbose_print(f"Retrying in {delay}s...", "info")
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to ElevenLabs after {MAX_RETRIES} attempts: {e}"
                    )

    async def disconnect(self):
        """
        Close WebSocket connection and cleanup resources.

        If a temporary agent was created, it will be deleted.
        """
        if not self.connected:
            return

        if self.verbose:
            verbose_print("Disconnecting from ElevenLabs...", "info")

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        self.connected = False

        # Delete temporary agent if we created one
        if self._created_agent_id:
            await self._delete_agent(self._created_agent_id)
            self._created_agent_id = None

        if self.verbose:
            verbose_print("Disconnected from ElevenLabs", "info")

    async def send_audio_chunk(self, audio_data: bytes):
        """
        Send audio chunk to ElevenLabs.

        Args:
            audio_data: Raw PCM16 audio bytes

        ElevenLabs expects:
        {
            "audioBase64": "base64_encoded_audio"
        }
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected. Call connect() first.")

        # Encode to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Send message
        message = {"audioBase64": audio_base64}

        await self.websocket.send(json.dumps(message))

        if self.verbose:
            verbose_print(f"Sent audio chunk ({len(audio_data)} bytes)", "debug")

    async def send_text_message(self, text: str):
        """
        Send text message to ElevenLabs.

        Args:
            text: Text message to send

        ElevenLabs expects:
        {
            "type": "user_message",
            "message": "text content"
        }
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected. Call connect() first.")

        message = {
            "type": "user_message",
            "message": text
        }

        await self.websocket.send(json.dumps(message))

        if self.verbose:
            verbose_print(f"Sent text message: {text[:50]}...", "info")

    async def send_client_tool_result(self, tool_call_id: str, result: Any):
        """
        Send client tool execution result back to ElevenLabs.

        Args:
            tool_call_id: ID of the tool call
            result: Result data to send back

        ElevenLabs expects:
        {
            "type": "client_tool_result",
            "tool_call_id": "call_id",
            "result": "result_data"
        }
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected. Call connect() first.")

        message = {
            "type": "client_tool_result",
            "tool_call_id": tool_call_id,
            "result": str(result)
        }

        await self.websocket.send(json.dumps(message))

        if self.verbose:
            verbose_print(f"Sent tool result for {tool_call_id}", "debug")

    def on(self, event_type: str, callback: Optional[Callable] = None):
        """
        Register event handler. Supports decorator and direct call patterns.

        Args:
            event_type: Event type to listen for
            callback: Callback function (optional for decorator pattern)
        """
        if callback is None:
            # Decorator pattern
            def decorator(func: Callable):
                if event_type not in self._event_handlers:
                    self._event_handlers[event_type] = []
                self._event_handlers[event_type].append(func)
                return func
            return decorator
        else:
            # Direct call
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(callback)

    async def _emit_to_handlers(self, event_type: str, event_data: Dict[str, Any]):
        """
        Emit event to all registered handlers.

        Args:
            event_type: Event type
            event_data: Event payload
        """
        if event_type in self._event_handlers:
            for callback in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    if self.verbose:
                        verbose_print(
                            f"Error in event handler for '{event_type}': {e}",
                            "error"
                        )

    async def _receive_events(self):
        """
        Background task to receive and process events from ElevenLabs.
        Translates ElevenLabs events to OpenAI-compatible format.
        """
        try:
            async for message in self.websocket:
                try:
                    event = json.loads(message)

                    if self.verbose:
                        event_type = event.get("type", "unknown")
                        verbose_print(f"Received event: {event_type}", "debug")

                    # Translate and emit event
                    await self._handle_elevenlabs_event(event)

                except json.JSONDecodeError as e:
                    if self.verbose:
                        verbose_print(f"Failed to parse event: {e}", "error")

        except ConnectionClosed:
            if self.verbose:
                verbose_print("WebSocket connection closed", "warning")
            self.connected = False

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error in receive loop: {e}", "error")
            self.connected = False

    async def _handle_elevenlabs_event(self, event: Dict[str, Any]):
        """
        Handle and translate ElevenLabs events to OpenAI-compatible format.

        ElevenLabs Event Types:
        - audio: Audio response chunks
        - user_transcript: User's speech transcribed
        - agent_response: Agent's text response
        - agent_chat_response_part: Streaming response chunks
        - interruption: User interrupted the agent
        - client_tool_call: Agent requests client-side function
        - conversation_initiation_metadata: Connection established
        - ping: Keep-alive

        Translates to OpenAI-compatible events:
        - response.audio.delta
        - conversation.item.input_audio_transcription.completed
        - response.audio_transcript.delta/done
        - etc.
        """
        event_type = event.get("type", "unknown")

        # Audio output
        if event_type == "audio":
            # Translate to OpenAI format: response.audio.delta
            await self._emit_to_handlers("response.audio.delta", {
                "type": "response.audio.delta",
                "delta": event.get("audio"),  # Already base64
                "event_id": event.get("event_id")
            })

        # User transcript (STT output)
        elif event_type == "user_transcript":
            # Translate to: conversation.item.input_audio_transcription.completed
            await self._emit_to_handlers("conversation.item.input_audio_transcription.completed", {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": event.get("transcript"),
                "event_id": event.get("event_id")
            })

        # Agent text response (full)
        elif event_type == "agent_response":
            # Translate to: response.audio_transcript.done
            response_text = event.get("response", "")
            await self._emit_to_handlers("response.audio_transcript.done", {
                "type": "response.audio_transcript.done",
                "transcript": response_text,
                "event_id": event.get("event_id")
            })

        # Agent response streaming (partial)
        elif event_type == "agent_chat_response_part":
            # Translate to: response.audio_transcript.delta
            partial_text = event.get("partial_response", "")
            await self._emit_to_handlers("response.audio_transcript.delta", {
                "type": "response.audio_transcript.delta",
                "delta": partial_text,
                "event_id": event.get("event_id")
            })

        # Interruption
        elif event_type == "interruption":
            # Translate to: input_audio_buffer.speech_started (similar intent)
            await self._emit_to_handlers("interruption", {
                "type": "interruption",
                "event_id": event.get("event_id")
            })
            # Also emit OpenAI-compatible event
            await self._emit_to_handlers("input_audio_buffer.speech_started", {
                "type": "input_audio_buffer.speech_started",
                "event_id": event.get("event_id")
            })

        # Client tool call
        elif event_type == "client_tool_call":
            # Translate to: response.function_call_arguments.done
            await self._emit_to_handlers("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "call_id": event.get("tool_call_id"),
                "name": event.get("tool_name"),
                "arguments": json.dumps(event.get("parameters", {})),
                "event_id": event.get("event_id")
            })
            # Also emit ElevenLabs-specific event
            await self._emit_to_handlers("client_tool_call", event)

        # Connection metadata
        elif event_type == "conversation_initiation_metadata":
            self.conversation_id = event.get("conversation_id")
            await self._emit_to_handlers("session.created", {
                "type": "session.created",
                "session": {
                    "id": self.conversation_id,
                    "model": self.session_config.model
                }
            })

        # Ping (keep-alive)
        elif event_type == "ping":
            # Respond with pong if needed (ElevenLabs handles this automatically)
            pass

        # Generic error handling
        elif event_type == "error":
            error_data = event.get("error", {})
            await self._emit_to_handlers("error", {
                "type": "error",
                "error": {
                    "type": error_data.get("type", "unknown_error"),
                    "message": error_data.get("message", "Unknown error"),
                    "code": error_data.get("code")
                }
            })

        # Emit raw event for custom handling
        await self._emit_to_handlers(f"elevenlabs.{event_type}", event)

    # ========================================================================
    # Agent Management (REST API)
    # ========================================================================

    async def _create_agent(self) -> str:
        """
        Create a new ElevenLabs agent with the specified voice and configuration.

        Returns:
            agent_id: ID of the created agent

        Raises:
            Exception: If agent creation fails
        """
        url = f"{ELEVENLABS_API_BASE}/convai/agents/create"

        # Build agent configuration
        agent_config = {
            "conversation_config": {
                "agent": {
                    "prompt": {
                        "prompt": self.session_config.instructions
                    },
                    "language": self.session_config.language
                },
                "tts": {
                    "voice_id": self.session_config.voice_id,
                    "model_id": self.session_config.tts_model,
                    "stability": self.session_config.stability,
                    "similarity_boost": self.session_config.similarity_boost,
                    "use_speaker_boost": self.session_config.use_speaker_boost,
                    "optimize_streaming_latency": self.session_config.optimize_streaming_latency
                },
                "llm": {
                    "model": self.session_config.model,
                    "temperature": self.session_config.temperature
                },
                "conversation": {
                    "max_duration_seconds": self.session_config.max_duration_seconds
                },
                "asr": {
                    "quality": "high",
                    "user_input_audio_format": self.session_config.output_format
                }
            },
            "platform_settings": {
                "turn_timeout": self.session_config.turn_timeout_ms
            }
        }

        # Add first message if specified
        if self.session_config.first_message:
            agent_config["conversation_config"]["agent"]["first_message"] = (
                self.session_config.first_message
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=agent_config,
                    timeout=30.0
                )
                response.raise_for_status()

                result = response.json()
                agent_id = result.get("agent_id")

                if not agent_id:
                    raise Exception("Agent creation failed: no agent_id in response")

                if self.verbose:
                    verbose_print(f"Created agent: {agent_id}", "info")

                return agent_id

        except Exception as e:
            if self.verbose:
                verbose_print(f"Failed to create agent: {e}", "error")
            raise

    async def _delete_agent(self, agent_id: str):
        """
        Delete an ElevenLabs agent.

        Args:
            agent_id: ID of the agent to delete
        """
        url = f"{ELEVENLABS_API_BASE}/convai/agents/{agent_id}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    url,
                    headers=self._get_headers(),
                    timeout=10.0
                )
                response.raise_for_status()

                if self.verbose:
                    verbose_print(f"Deleted agent: {agent_id}", "info")

        except Exception as e:
            if self.verbose:
                verbose_print(f"Failed to delete agent {agent_id}: {e}", "warning")

    async def _get_websocket_url(self) -> str:
        """
        Get WebSocket URL for connecting to ElevenLabs.

        If use_signed_url is True, fetches a signed URL from the API.
        Otherwise, returns the public agent URL.

        Returns:
            WebSocket URL
        """
        if self.session_config.use_signed_url:
            # Get signed URL for private agent
            url = f"{ELEVENLABS_API_BASE}/convai/conversation/get-signed-url"
            params = {"agent_id": self._agent_id}

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        url,
                        headers=self._get_headers(),
                        params=params,
                        timeout=10.0
                    )
                    response.raise_for_status()

                    result = response.json()
                    signed_url = result.get("signed_url")

                    if not signed_url:
                        raise Exception("Failed to get signed URL")

                    if self.verbose:
                        verbose_print("Got signed URL for private agent", "info")

                    return signed_url

            except Exception as e:
                if self.verbose:
                    verbose_print(f"Failed to get signed URL: {e}", "error")
                raise
        else:
            # Public agent - use direct URL
            return f"{ELEVENLABS_WS_URL}?agent_id={self._agent_id}"

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
