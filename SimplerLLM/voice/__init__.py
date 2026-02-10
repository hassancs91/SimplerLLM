"""
SimplerLLM Voice Module

Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.
"""

# TTS exports
from .tts import (
    TTS,
    TTSBase,
    TTSProvider,
    TTSResponse,
    Voice,
    TTSError,
    TTSValidationError,
    TTSProviderError,
    TTSVoiceNotFoundError,
    OpenAITTS,
    OPENAI_VOICES,
    OPENAI_MODELS,
    OPENAI_FORMATS,
    ELEVENLABS_MODELS,
    ELEVENLABS_FORMATS,
)

# Optional ElevenLabs TTS
try:
    from .tts import ElevenLabsTTS
    _has_elevenlabs = True
except (ImportError, TypeError):
    _has_elevenlabs = False
    ElevenLabsTTS = None

# STT exports
from .stt import STT, STTProvider, OpenAISTT, STTFullResponse

__all__ = [
    # TTS - Factory
    'TTS',
    'TTSBase',
    # TTS - Models
    'TTSProvider',
    'TTSResponse',
    'Voice',
    # TTS - Exceptions
    'TTSError',
    'TTSValidationError',
    'TTSProviderError',
    'TTSVoiceNotFoundError',
    # TTS - Providers
    'OpenAITTS',
    # TTS - Constants
    'OPENAI_VOICES',
    'OPENAI_MODELS',
    'OPENAI_FORMATS',
    'ELEVENLABS_MODELS',
    'ELEVENLABS_FORMATS',
    # STT
    'STT',
    'STTProvider',
    'OpenAISTT',
    'STTFullResponse',
]

if _has_elevenlabs:
    __all__.append('ElevenLabsTTS')
