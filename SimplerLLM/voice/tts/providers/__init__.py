from .tts_response_models import TTSFullResponse
from . import openai_tts

# Optional provider - only import if elevenlabs package is installed
try:
    from . import elevenlabs_tts
    _has_elevenlabs = True
except ImportError:
    _has_elevenlabs = False
    elevenlabs_tts = None

__all__ = [
    'TTSFullResponse',
    'openai_tts',
]

if _has_elevenlabs:
    __all__.append('elevenlabs_tts')
