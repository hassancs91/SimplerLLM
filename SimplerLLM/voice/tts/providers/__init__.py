"""
TTS Providers

Provider implementations for different TTS services.
"""

from .openai_tts import OpenAITTS
from .lahajati_tts import LahajatiTTS

# Optional provider - only import if elevenlabs package is installed
try:
    from .elevenlabs_tts import ElevenLabsTTS
    _has_elevenlabs = True
except ImportError:
    _has_elevenlabs = False
    ElevenLabsTTS = None

__all__ = [
    'OpenAITTS',
    'LahajatiTTS',
]

if _has_elevenlabs:
    __all__.append('ElevenLabsTTS')
