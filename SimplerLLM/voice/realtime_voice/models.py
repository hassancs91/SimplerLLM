"""
Pydantic models for OpenAI Realtime Voice API.
These models define the structure for sessions, messages, events, and configurations.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from enum import Enum


class TurnDetectionType(str, Enum):
    """Voice Activity Detection (VAD) mode."""
    SERVER_VAD = "server_vad"
    NONE = "none"  # Manual turn detection


class Modality(str, Enum):
    """Supported modalities for the session."""
    TEXT = "text"
    AUDIO = "audio"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    PCM16 = "pcm16"  # 16-bit PCM at 24kHz
    G711_ULAW = "g711_ulaw"  # G.711 u-law at 8kHz
    G711_ALAW = "g711_alaw"  # G.711 a-law at 8kHz


class Voice(str, Enum):
    """Available OpenAI voices."""
    ALLOY = "alloy"
    ECHO = "echo"
    SHIMMER = "shimmer"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    SAGE = "sage"
    VERSE = "verse"


class TurnDetectionConfig(BaseModel):
    """Configuration for turn detection."""
    type: TurnDetectionType = TurnDetectionType.SERVER_VAD
    threshold: Optional[float] = Field(default=0.5, description="Activation threshold (0.0-1.0)")
    prefix_padding_ms: Optional[int] = Field(default=300, description="Audio before speech start (ms)")
    silence_duration_ms: Optional[int] = Field(default=500, description="Silence to detect turn end (ms)")


class InputAudioTranscriptionConfig(BaseModel):
    """Configuration for input audio transcription."""
    model: str = Field(default="whisper-1", description="Transcription model")


class RealtimeSessionConfig(BaseModel):
    """
    Configuration for a Realtime Voice session.
    This defines the session parameters including model, voice, instructions, etc.
    """
    model: str = Field(
        default="gpt-4o-realtime-preview-2024-10-01",
        description="The Realtime model to use"
    )
    modalities: List[Modality] = Field(
        default=[Modality.TEXT, Modality.AUDIO],
        description="List of modalities the model can respond with"
    )
    voice: Voice = Field(default=Voice.ALLOY, description="The voice to use for audio responses")
    instructions: Optional[str] = Field(
        default="You are a helpful assistant.",
        description="System instructions for the model"
    )
    input_audio_format: AudioFormat = Field(
        default=AudioFormat.PCM16,
        description="Format of input audio"
    )
    output_audio_format: AudioFormat = Field(
        default=AudioFormat.PCM16,
        description="Format of output audio"
    )
    input_audio_transcription: Optional[InputAudioTranscriptionConfig] = Field(
        default=None,
        description="Configuration for input audio transcription"
    )
    turn_detection: Optional[TurnDetectionConfig] = Field(
        default_factory=lambda: TurnDetectionConfig(type=TurnDetectionType.SERVER_VAD),
        description="Turn detection configuration (Server VAD or manual)"
    )
    temperature: float = Field(default=0.8, ge=0.6, le=1.2, description="Sampling temperature")
    max_response_output_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens for response (default: inf)"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of tools/functions available to the model"
    )
    tool_choice: Optional[str] = Field(
        default="auto",
        description="Tool choice setting ('auto', 'none', 'required', or specific tool)"
    )

    class Config:
        use_enum_values = True


class ElevenLabsSessionConfig(BaseModel):
    """
    Configuration for ElevenLabs Conversational AI session.

    ElevenLabs uses an agent-based architecture where voice and configuration
    are pre-configured in an agent. You can either:
    1. Use an existing agent_id (created via dashboard or API)
    2. Provide voice_id and let SimplerLLM create a temporary agent

    Example (existing agent):
        >>> config = ElevenLabsSessionConfig(agent_id="your_agent_id")

    Example (dynamic agent with custom voice):
        >>> config = ElevenLabsSessionConfig(
        ...     voice_id="your_cloned_voice_id",
        ...     model="gpt-4o-mini",
        ...     instructions="You are helpful"
        ... )
    """
    # Agent configuration
    agent_id: Optional[str] = Field(
        default=None,
        description="Existing ElevenLabs agent ID (use this OR voice_id)"
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for dynamic agent creation (use this OR agent_id)"
    )

    # LLM settings (for dynamic agent creation)
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model (gpt-4o-mini, gpt-4o, claude-3-5-sonnet, etc.)"
    )
    instructions: Optional[str] = Field(
        default="You are a helpful assistant.",
        description="System instructions for the agent"
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )
    first_message: Optional[str] = Field(
        default=None,
        description="Initial greeting from the agent"
    )
    language: str = Field(
        default="en",
        description="Agent language code"
    )

    # TTS settings (for dynamic agent creation)
    tts_model: str = Field(
        default="eleven_turbo_v2_5",
        description="TTS model (eleven_turbo_v2_5, eleven_multilingual_v2)"
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability (0.0-1.0)"
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Voice similarity boost (0.0-1.0)"
    )
    use_speaker_boost: bool = Field(
        default=True,
        description="Enable speaker boost for clarity"
    )
    optimize_streaming_latency: int = Field(
        default=3,
        ge=0,
        le=4,
        description="Latency optimization (0=quality, 4=speed)"
    )

    # Audio settings
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz (ElevenLabs default: 16000)"
    )
    output_format: str = Field(
        default="pcm_16000",
        description="Audio output format"
    )

    # Authentication
    use_signed_url: bool = Field(
        default=False,
        description="Use signed URL for private agents"
    )

    # Session settings
    max_duration_seconds: int = Field(
        default=600,
        description="Maximum conversation duration in seconds"
    )
    turn_timeout_ms: int = Field(
        default=10000,
        description="Milliseconds before auto-hang-up on silence"
    )

    class Config:
        use_enum_values = True


class RealtimeMessage(BaseModel):
    """
    Represents a message in the Realtime conversation.
    Can contain text, audio, or both.
    """
    id: Optional[str] = None
    type: Literal["message", "function_call", "function_call_output"] = "message"
    role: Literal["user", "assistant", "system"] = "user"
    content: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Message content (text and/or audio)"
    )
    text: Optional[str] = Field(default=None, description="Text content (convenience field)")
    audio: Optional[bytes] = Field(default=None, description="Audio data (convenience field)")
    transcript: Optional[str] = Field(default=None, description="Audio transcript if available")
    timestamp: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class RealtimeFunctionCall(BaseModel):
    """Represents a function call from the model."""
    call_id: str = Field(description="Unique ID for this function call")
    name: str = Field(description="Function name")
    arguments: str = Field(description="JSON string of function arguments")


class RealtimeUsage(BaseModel):
    """Token usage information for a response."""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_token_details: Optional[Dict[str, int]] = None


class RealtimeResponse(BaseModel):
    """
    Full response from the Realtime API.
    Contains the response content, metadata, and usage information.
    """
    id: Optional[str] = None
    status: Literal["in_progress", "completed", "failed", "cancelled"] = "in_progress"
    status_details: Optional[Dict[str, Any]] = None
    output: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Response output items"
    )
    text: Optional[str] = Field(default=None, description="Text response (convenience)")
    audio: Optional[bytes] = Field(default=None, description="Audio response (convenience)")
    transcript: Optional[str] = Field(default=None, description="Audio transcript")
    function_calls: Optional[List[RealtimeFunctionCall]] = Field(
        default=None,
        description="Function calls made by the model"
    )
    usage: Optional[RealtimeUsage] = None
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration: Optional[float] = Field(
        default=None,
        description="Response duration in seconds"
    )

    class Config:
        arbitrary_types_allowed = True


class RealtimeEvent(BaseModel):
    """
    Wrapper for Realtime API events.
    Used for both client and server events.
    """
    event_id: Optional[str] = None
    type: str = Field(description="Event type (e.g., 'response.audio.delta')")
    event_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload"
    )
    timestamp: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class RealtimeError(BaseModel):
    """Error information from the Realtime API."""
    type: str = Field(description="Error type")
    code: Optional[str] = None
    message: str = Field(description="Error message")
    param: Optional[str] = None
    event_id: Optional[str] = None
