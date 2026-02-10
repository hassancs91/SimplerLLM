"""
SimplerLLM TTS Module

Production-ready Text-to-Speech with a unified provider interface.

Example:
    from SimplerLLM.voice.tts import TTS, TTSProvider

    # Create provider instance
    tts = TTS.create(TTSProvider.OPENAI)

    # Generate speech
    response = tts.generate_speech("Hello world")

    # List available voices
    voices = tts.list_voices()
"""

from .factory import TTS
from .base import TTSBase
from .models import (
    TTSProvider,
    TTSResponse,
    Voice,
    TTSError,
    TTSValidationError,
    TTSProviderError,
    TTSVoiceNotFoundError,
    OPENAI_VOICES,
    OPENAI_MODELS,
    OPENAI_FORMATS,
    ELEVENLABS_MODELS,
    ELEVENLABS_FORMATS,
    LAHAJATI_FORMATS,
    LAHAJATI_INPUT_MODES,
    Dialect,
    Performance,
)
from .providers import OpenAITTS, LahajatiTTS

# Optional provider - only import if elevenlabs package is installed
try:
    from .providers import ElevenLabsTTS
    _has_elevenlabs = True
except (ImportError, TypeError):
    _has_elevenlabs = False
    ElevenLabsTTS = None

__all__ = [
    # Factory
    'TTS',
    # Base class
    'TTSBase',
    # Models
    'TTSProvider',
    'TTSResponse',
    'Voice',
    'Dialect',
    'Performance',
    # Exceptions
    'TTSError',
    'TTSValidationError',
    'TTSProviderError',
    'TTSVoiceNotFoundError',
    # Providers
    'OpenAITTS',
    'LahajatiTTS',
    # Constants
    'OPENAI_VOICES',
    'OPENAI_MODELS',
    'OPENAI_FORMATS',
    'ELEVENLABS_MODELS',
    'ELEVENLABS_FORMATS',
    'LAHAJATI_FORMATS',
    'LAHAJATI_INPUT_MODES',
]

if _has_elevenlabs:
    __all__.append('ElevenLabsTTS')
