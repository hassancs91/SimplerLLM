from pydantic import BaseModel, Field
from typing import Optional
from ..voice_chat.models import VoiceChatConfig


class LiveVoiceChatConfig(VoiceChatConfig):
    """
    Configuration for LiveVoiceChat with microphone input.

    Extends VoiceChatConfig with additional settings for audio recording
    and playback.
    """

    # Audio recording settings
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    """Audio sample rate in Hz (16000 recommended for STT)"""

    channels: int = Field(default=1, ge=1, le=2)
    """Number of audio channels (1=mono, 2=stereo)"""

    audio_dtype: str = 'int16'
    """Audio data type ('int16' or 'float32')"""

    # Push-to-talk settings
    push_to_talk_key: str = 'space'
    """Key to use for push-to-talk recording"""

    max_recording_duration: Optional[float] = None
    """Maximum recording duration in seconds (None = unlimited)"""

    # Playback settings
    auto_play_response: bool = True
    """Automatically play TTS response audio"""

    playback_volume: float = Field(default=1.0, ge=0.0, le=1.0)
    """Playback volume (0.0 to 1.0)"""

    # File handling
    cleanup_temp_files: bool = True
    """Automatically delete temporary audio files after processing"""

    temp_audio_dir: Optional[str] = None
    """Directory for temporary audio files (None = system temp)"""

    class Config:
        json_schema_extra = {
            "example": {
                "system_prompt": "You are a helpful voice assistant",
                "temperature": 0.7,
                "tts_voice": "nova",
                "sample_rate": 16000,
                "channels": 1,
                "push_to_talk_key": "space",
                "auto_play_response": True,
                "playback_volume": 0.8,
                "cleanup_temp_files": True,
                "max_history_length": 10
            }
        }
