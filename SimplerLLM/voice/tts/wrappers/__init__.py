from .openai_wrapper import OpenAITTS

# Optional wrapper - only import if elevenlabs is available
try:
    from .elevenlabs_wrapper import ElevenLabsTTS
    _has_elevenlabs = True
except ImportError:
    _has_elevenlabs = False
    ElevenLabsTTS = None

__all__ = [
    'OpenAITTS',
]

if _has_elevenlabs:
    __all__.append('ElevenLabsTTS')
