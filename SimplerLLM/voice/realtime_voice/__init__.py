"""
Realtime Voice API integration for SimplerLLM.

This module provides a unified interface for building AI voice agents using
realtime voice APIs including OpenAI Realtime API and ElevenLabs Conversational AI.

Example (OpenAI):
    >>> from SimplerLLM import RealtimeVoice, RealtimeVoiceProvider
    >>> realtime = RealtimeVoice.create(
    ...     provider=RealtimeVoiceProvider.OPENAI,
    ...     model="gpt-4o-realtime-preview-2024-10-01",
    ...     voice="alloy"
    ... )
    >>> await realtime.connect()
    >>> await realtime.send_text("Hello!")

Example (ElevenLabs with Custom Voice):
    >>> realtime = RealtimeVoice.create(
    ...     provider=RealtimeVoiceProvider.ELEVENLABS,
    ...     voice_id="your_cloned_voice_id",
    ...     model="gpt-4o-mini"
    ... )
    >>> await realtime.connect()
    >>> await realtime.send_audio(audio_bytes)
"""

from .base import RealtimeVoice, RealtimeVoiceProvider
from .models import (
    RealtimeSessionConfig,
    ElevenLabsSessionConfig,
    RealtimeMessage,
    RealtimeResponse,
    RealtimeEvent,
    RealtimeError,
    RealtimeFunctionCall,
    RealtimeUsage,
    TurnDetectionType,
    TurnDetectionConfig,
    InputAudioTranscriptionConfig,
    Modality,
    AudioFormat,
    Voice
)
from .providers import (
    RealtimeFullResponse,
    RealtimeStreamChunk,
    RealtimeSessionInfo,
    RealtimeConversationItem,
    RealtimeFunctionCallResult
)
from .wrappers import OpenAIRealtimeVoice, ElevenLabsRealtimeVoice
from .realtime_voice_chat import RealtimeVoiceChat, RealtimeVoiceChatConfig
from .audio_utils import (
    resample_audio,
    resample_24k_to_16k,
    resample_16k_to_24k,
    AudioResampler
)

__all__ = [
    # Base classes
    'RealtimeVoice',
    'RealtimeVoiceProvider',

    # Configuration models
    'RealtimeSessionConfig',
    'ElevenLabsSessionConfig',
    'TurnDetectionConfig',
    'InputAudioTranscriptionConfig',

    # Message and response models
    'RealtimeMessage',
    'RealtimeResponse',
    'RealtimeEvent',
    'RealtimeError',
    'RealtimeFunctionCall',
    'RealtimeUsage',

    # Enums
    'TurnDetectionType',
    'Modality',
    'AudioFormat',
    'Voice',

    # Provider response models
    'RealtimeFullResponse',
    'RealtimeStreamChunk',
    'RealtimeSessionInfo',
    'RealtimeConversationItem',
    'RealtimeFunctionCallResult',

    # Wrappers
    'OpenAIRealtimeVoice',
    'ElevenLabsRealtimeVoice',

    # Voice chat
    'RealtimeVoiceChat',
    'RealtimeVoiceChatConfig',

    # Audio utilities
    'resample_audio',
    'resample_24k_to_16k',
    'resample_16k_to_24k',
    'AudioResampler'
]
