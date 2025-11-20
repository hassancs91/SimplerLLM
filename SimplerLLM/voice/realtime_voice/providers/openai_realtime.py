"""
OpenAI Realtime Voice API provider implementation.
Handles WebSocket connection, event management, and audio streaming.
"""

import asyncio
import json
import base64
import time
import os
from typing import Optional, Dict, Any, Callable, List
import websockets
from websockets.exceptions import ConnectionClosed

from ..models import (
    RealtimeSessionConfig,
    RealtimeEvent,
    RealtimeError,
    TurnDetectionType
)
from .realtime_response_models import (
    RealtimeFullResponse,
    RealtimeStreamChunk,
    RealtimeSessionInfo
)
from SimplerLLM.utils.custom_verbose import verbose_print


# Constants
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-10-01"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


class OpenAIRealtimeProvider:
    """
    Core provider implementation for OpenAI Realtime Voice API.
    Manages WebSocket connection and event handling.
    """

    def __init__(
        self,
        session_config: RealtimeSessionConfig,
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize OpenAI Realtime provider.

        Args:
            session_config: Configuration for the session
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            verbose: Enable verbose logging
        """
        self.session_config = session_config
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.verbose = verbose

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # WebSocket connection
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self._receive_task: Optional[asyncio.Task] = None

        # Session state
        self.session_id: Optional[str] = None
        self.session_info: Optional[RealtimeSessionInfo] = None

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Response tracking
        self._current_response: Optional[Dict[str, Any]] = None
        self._response_audio_chunks: List[bytes] = []
        self._response_text_chunks: List[str] = []

        # Audio buffer for input
        self._audio_buffer: List[bytes] = []

        if self.verbose:
            verbose_print(
                f"Initialized OpenAI Realtime Provider with model: {session_config.model}",
                "info"
            )

    async def connect(self):
        """
        Establish WebSocket connection to OpenAI Realtime API.

        Raises:
            ConnectionError: If connection fails after retries
        """
        url = f"{OPENAI_REALTIME_URL}?model={self.session_config.model}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        for attempt in range(MAX_RETRIES):
            try:
                if self.verbose:
                    verbose_print(
                        f"Connecting to OpenAI Realtime API (attempt {attempt + 1}/{MAX_RETRIES})...",
                        "info"
                    )

                self.websocket = await websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20
                )

                self.connected = True

                if self.verbose:
                    verbose_print("WebSocket connection established", "info")

                # Start receiving messages
                self._receive_task = asyncio.create_task(self._receive_loop())

                # Wait for session.created event
                await asyncio.sleep(0.5)  # Give time for session.created

                # Send session.update to configure the session
                await self._send_session_update()

                return

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    if self.verbose:
                        verbose_print(
                            f"Connection failed: {e}. Retrying in {wait_time}s...",
                            "warning"
                        )
                    await asyncio.sleep(wait_time)
                else:
                    if self.verbose:
                        verbose_print(f"Connection failed after {MAX_RETRIES} attempts: {e}", "error")
                    raise ConnectionError(f"Failed to connect to OpenAI Realtime API: {e}")

    async def disconnect(self):
        """Close WebSocket connection."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False

            if self.verbose:
                verbose_print("WebSocket connection closed", "info")

    async def _send_session_update(self):
        """Send session.update event to configure the session."""
        session_data = {
            "modalities": [m.value if hasattr(m, 'value') else m for m in self.session_config.modalities],
            "voice": self.session_config.voice.value if hasattr(self.session_config.voice, 'value') else self.session_config.voice,
            "input_audio_format": self.session_config.input_audio_format.value if hasattr(self.session_config.input_audio_format, 'value') else self.session_config.input_audio_format,
            "output_audio_format": self.session_config.output_audio_format.value if hasattr(self.session_config.output_audio_format, 'value') else self.session_config.output_audio_format,
            "temperature": self.session_config.temperature,
        }

        # Add optional fields
        if self.session_config.instructions:
            session_data["instructions"] = self.session_config.instructions

        if self.session_config.turn_detection:
            turn_detection_data = {
                "type": self.session_config.turn_detection.type.value if hasattr(self.session_config.turn_detection.type, 'value') else self.session_config.turn_detection.type
            }
            # Only add VAD parameters if using server_vad
            if self.session_config.turn_detection.type == TurnDetectionType.SERVER_VAD:
                if self.session_config.turn_detection.threshold is not None:
                    turn_detection_data["threshold"] = self.session_config.turn_detection.threshold
                if self.session_config.turn_detection.prefix_padding_ms is not None:
                    turn_detection_data["prefix_padding_ms"] = self.session_config.turn_detection.prefix_padding_ms
                if self.session_config.turn_detection.silence_duration_ms is not None:
                    turn_detection_data["silence_duration_ms"] = self.session_config.turn_detection.silence_duration_ms

            session_data["turn_detection"] = turn_detection_data

        if self.session_config.input_audio_transcription:
            session_data["input_audio_transcription"] = {
                "model": self.session_config.input_audio_transcription.model
            }

        if self.session_config.max_response_output_tokens:
            session_data["max_response_output_tokens"] = self.session_config.max_response_output_tokens

        if self.session_config.tools:
            session_data["tools"] = self.session_config.tools

        if self.session_config.tool_choice:
            session_data["tool_choice"] = self.session_config.tool_choice

        await self.send_event("session.update", {"session": session_data})

        if self.verbose:
            verbose_print("Session configuration sent", "info")

    async def send_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """
        Send a client event to the API.

        Args:
            event_type: Type of event (e.g., "session.update", "input_audio_buffer.append")
            data: Event payload

        Raises:
            RuntimeError: If not connected
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to Realtime API")

        event = {
            "type": event_type
        }

        if data:
            event.update(data)

        await self.websocket.send(json.dumps(event))

        # Only log non-audio events to avoid spam
        if self.verbose and event_type != "input_audio_buffer.append":
            verbose_print(f"Sent event: {event_type}", "debug")

    async def send_audio_chunk(self, audio_data: bytes, commit: bool = False):
        """
        Send audio chunk to the API.

        Args:
            audio_data: Raw audio data (PCM16 24kHz or configured format)
            commit: Whether to commit the audio buffer after sending

        Note:
            Audio must be in the format specified in session config (default: PCM16 24kHz)
        """
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Send input_audio_buffer.append event
        await self.send_event("input_audio_buffer.append", {
            "audio": audio_base64
        })

        if commit:
            await self.commit_audio_buffer()

    async def commit_audio_buffer(self):
        """
        Commit the audio buffer and create a user message.
        In Server VAD mode, this happens automatically.
        """
        await self.send_event("input_audio_buffer.commit")

        if self.verbose:
            verbose_print("Audio buffer committed", "debug")

    async def clear_audio_buffer(self):
        """Clear the audio input buffer."""
        await self.send_event("input_audio_buffer.clear")

        if self.verbose:
            verbose_print("Audio buffer cleared", "debug")

    async def send_text_message(self, text: str):
        """
        Send a text message to the conversation.

        Args:
            text: Text message to send
        """
        await self.send_event("conversation.item.create", {
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        })

        # Trigger response creation
        await self.send_event("response.create")

        if self.verbose:
            verbose_print(f"Sent text message: {text[:50]}...", "info")

    async def create_response(self, **kwargs):
        """
        Manually trigger a response creation.
        In Server VAD mode, responses are created automatically.

        Args:
            **kwargs: Optional response configuration (instructions, temperature, etc.)
        """
        event_data = {}
        if kwargs:
            event_data["response"] = kwargs

        await self.send_event("response.create", event_data)

        if self.verbose:
            verbose_print("Response creation triggered", "debug")

    async def cancel_response(self):
        """Cancel the current in-progress response."""
        await self.send_event("response.cancel")

        if self.verbose:
            verbose_print("Response cancelled", "info")

    async def truncate_conversation_item(self, item_id: str, content_index: int, audio_end_ms: int):
        """
        Truncate a conversation item (used for interruptions).

        Args:
            item_id: ID of the item to truncate
            content_index: Index of the content part to truncate
            audio_end_ms: Millisecond timestamp to truncate at
        """
        await self.send_event("conversation.item.truncate", {
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms
        })

    async def _receive_loop(self):
        """
        Main loop to receive and process server events.
        Runs in background task.
        """
        try:
            while self.connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    event = json.loads(message)
                    await self._handle_server_event(event)

                except ConnectionClosed:
                    if self.verbose:
                        verbose_print("WebSocket connection closed", "warning")
                    self.connected = False
                    break

                except json.JSONDecodeError as e:
                    if self.verbose:
                        verbose_print(f"Failed to decode message: {e}", "error")

        except asyncio.CancelledError:
            if self.verbose:
                verbose_print("Receive loop cancelled", "debug")
            raise

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error in receive loop: {e}", "error")
            self.connected = False

    async def _handle_server_event(self, event: Dict[str, Any]):
        """
        Process a server event and emit to registered handlers.

        Args:
            event: Server event data
        """
        event_type = event.get("type")

        # Log non-audio events
        if self.verbose and not event_type.endswith(".delta"):
            verbose_print(f"Received event: {event_type}", "debug")

        # Handle specific events for state management
        if event_type == "session.created":
            self.session_id = event.get("session", {}).get("id")
            if self.verbose:
                verbose_print(f"Session created: {self.session_id}", "info")

        elif event_type == "session.updated":
            if self.verbose:
                verbose_print("Session updated", "info")

        elif event_type == "error":
            error_data = event.get("error", {})
            if self.verbose:
                verbose_print(f"API Error: {error_data.get('message')}", "error")

        elif event_type == "response.created":
            self._current_response = event.get("response")
            self._response_audio_chunks = []
            self._response_text_chunks = []

        elif event_type == "response.audio.delta":
            # Accumulate audio chunks
            audio_delta = event.get("delta")
            if audio_delta:
                audio_bytes = base64.b64decode(audio_delta)
                self._response_audio_chunks.append(audio_bytes)

        elif event_type == "response.audio_transcript.delta":
            # Accumulate transcript chunks
            text_delta = event.get("delta")
            if text_delta:
                self._response_text_chunks.append(text_delta)

        elif event_type == "response.done":
            # Response completed
            if self.verbose:
                response = event.get("response", {})
                status = response.get("status")
                verbose_print(f"Response completed with status: {status}", "info")

        # Emit event to registered handlers
        await self._emit_to_handlers(event_type, event)

    async def _emit_to_handlers(self, event_type: str, event_data: Dict[str, Any]):
        """
        Emit event to all registered handlers.

        Args:
            event_type: The event type
            event_data: The event data
        """
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                except Exception as e:
                    if self.verbose:
                        verbose_print(
                            f"Error in event handler for {event_type}: {e}",
                            "error"
                        )

    def on(self, event_type: str, handler: Optional[Callable] = None):
        """
        Register an event handler.
        Supports both decorator and direct call patterns.

        Args:
            event_type: Event type to listen for
            handler: Callback function (can be sync or async), optional for decorator use
        """
        if handler is None:
            # Decorator pattern: @provider.on("event_type")
            def decorator(func: Callable):
                if event_type not in self._event_handlers:
                    self._event_handlers[event_type] = []
                self._event_handlers[event_type].append(func)
                return func
            return decorator
        else:
            # Direct call pattern: provider.on("event_type", handler)
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Optional[Callable] = None):
        """
        Unregister an event handler.

        Args:
            event_type: Event type
            handler: Specific handler to remove (if None, removes all)
        """
        if handler is None:
            self._event_handlers.pop(event_type, None)
        elif event_type in self._event_handlers:
            self._event_handlers[event_type] = [
                h for h in self._event_handlers[event_type] if h != handler
            ]

    def get_accumulated_audio(self) -> bytes:
        """
        Get all accumulated audio chunks from current response.

        Returns:
            Concatenated audio data
        """
        return b''.join(self._response_audio_chunks)

    def get_accumulated_transcript(self) -> str:
        """
        Get accumulated transcript from current response.

        Returns:
            Concatenated transcript text
        """
        return ''.join(self._response_text_chunks)
