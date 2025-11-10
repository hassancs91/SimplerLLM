from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class ConversationRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: ConversationRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    audio_file: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What's the weather today?",
                "timestamp": "2025-01-15T10:30:00",
                "audio_file": "input_001.wav"
            }
        }


class VoiceChatConfig(BaseModel):
    """Configuration for VoiceChat session."""

    # LLM settings
    system_prompt: str = "You are a helpful voice assistant."
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=150, ge=1, le=4000)

    # Voice settings
    tts_voice: Optional[str] = None
    tts_speed: float = Field(default=1.0, ge=0.25, le=4.0)
    tts_model: Optional[str] = None
    stt_language: Optional[str] = None
    stt_model: Optional[str] = None

    # Conversation settings
    max_history_length: int = Field(default=20, ge=0, le=100)
    save_audio: bool = False
    output_dir: str = "voice_chat_output"

    # Advanced features (for future extensibility)
    enable_tools: bool = False
    enable_rag: bool = False
    enable_routing: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "system_prompt": "You are a friendly cooking assistant",
                "temperature": 0.8,
                "max_tokens": 200,
                "tts_voice": "nova",
                "tts_speed": 1.1,
                "max_history_length": 10,
                "save_audio": True,
                "output_dir": "cooking_chat"
            }
        }


class VoiceTurnResult(BaseModel):
    """Result of a single voice interaction turn."""
    user_audio_path: Optional[str] = None
    user_text: str
    assistant_text: str
    assistant_audio_path: Optional[str] = None

    stt_duration: Optional[float] = None
    llm_duration: Optional[float] = None
    tts_duration: Optional[float] = None
    total_duration: float

    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_text": "What's the capital of France?",
                "assistant_text": "The capital of France is Paris.",
                "assistant_audio_path": "output/assistant_001.mp3",
                "stt_duration": 1.2,
                "llm_duration": 0.8,
                "tts_duration": 1.5,
                "total_duration": 3.5
            }
        }


class VoiceChatSession(BaseModel):
    """Complete voice chat session data."""
    session_id: str
    config: VoiceChatConfig
    conversation_history: List[ConversationMessage]
    turns: List[VoiceTurnResult]

    started_at: datetime
    ended_at: Optional[datetime] = None
    total_turns: int = 0
    success: bool = True

    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "total_turns": 5,
                "started_at": "2025-01-15T10:00:00",
                "ended_at": "2025-01-15T10:15:00",
                "success": True
            }
        }
