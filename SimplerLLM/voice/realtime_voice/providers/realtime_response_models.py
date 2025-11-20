"""
Response models for OpenAI Realtime Voice API.
These models wrap the full response data from the provider.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class RealtimeFullResponse(BaseModel):
    """
    Complete response wrapper for OpenAI Realtime Voice API.
    Contains the response data plus metadata about the request.

    Similar to other SimplerLLM response models (LLMFullResponse, TTSFullResponse, etc.).
    """
    response_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this response"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier"
    )
    model: str = Field(
        description="Model used for this response"
    )
    voice: Optional[str] = Field(
        default=None,
        description="Voice used for audio output"
    )
    status: str = Field(
        default="completed",
        description="Response status (in_progress, completed, failed, cancelled)"
    )
    text_output: Optional[str] = Field(
        default=None,
        description="Text content of the response"
    )
    audio_output: Optional[bytes] = Field(
        default=None,
        description="Audio data of the response (PCM16 24kHz)"
    )
    transcript: Optional[str] = Field(
        default=None,
        description="Transcript of the audio output"
    )
    input_transcript: Optional[str] = Field(
        default=None,
        description="Transcript of the audio input (if transcription enabled)"
    )
    function_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Function calls made by the model"
    )
    usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage information"
    )
    process_time: float = Field(
        description="Time taken to process the request (seconds)"
    )
    created_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when response was created"
    )
    completed_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when response completed"
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error information if response failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )
    provider_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw response from the provider API"
    )

    class Config:
        arbitrary_types_allowed = True


class RealtimeStreamChunk(BaseModel):
    """
    Represents a streaming chunk from the Realtime API.
    Used for incremental audio/text delivery.
    """
    chunk_id: Optional[str] = None
    type: str = Field(description="Chunk type (audio, text, transcript, etc.)")
    data: Any = Field(description="Chunk data (audio bytes, text string, etc.)")
    delta: bool = Field(
        default=True,
        description="Whether this is a delta (incremental) chunk"
    )
    done: bool = Field(
        default=False,
        description="Whether this is the final chunk"
    )
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class RealtimeSessionInfo(BaseModel):
    """
    Information about a Realtime API session.
    """
    session_id: str = Field(description="Unique session identifier")
    model: str = Field(description="Model being used")
    voice: Optional[str] = None
    modalities: List[str] = Field(description="Active modalities (text, audio)")
    instructions: Optional[str] = None
    turn_detection_type: Optional[str] = Field(
        default="server_vad",
        description="Turn detection mode"
    )
    input_audio_format: str = Field(default="pcm16")
    output_audio_format: str = Field(default="pcm16")
    temperature: float = Field(default=0.8)
    tools: Optional[List[Dict[str, Any]]] = None
    created_at: float = Field(description="Unix timestamp of session creation")
    expires_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when session expires (15 min max)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RealtimeConversationItem(BaseModel):
    """
    Represents an item in the conversation history.
    """
    item_id: str = Field(description="Unique item identifier")
    type: str = Field(description="Item type (message, function_call, function_call_output)")
    role: Optional[str] = Field(default=None, description="Role (user, assistant, system)")
    content: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = Field(
        default="completed",
        description="Item status (in_progress, completed, incomplete)"
    )
    created_at: float = Field(description="Unix timestamp of item creation")


class RealtimeFunctionCallResult(BaseModel):
    """
    Result from a function call execution.
    """
    call_id: str = Field(description="Function call ID")
    name: str = Field(description="Function name")
    arguments: Dict[str, Any] = Field(description="Parsed function arguments")
    output: Optional[Any] = Field(
        default=None,
        description="Function execution result"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if function failed"
    )
    execution_time: Optional[float] = Field(
        default=None,
        description="Time taken to execute function (seconds)"
    )

    class Config:
        arbitrary_types_allowed = True
