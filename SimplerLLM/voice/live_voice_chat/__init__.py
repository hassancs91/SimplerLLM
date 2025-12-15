from .models import LiveVoiceChatConfig

# These imports may fail if sounddevice/pynput not available (requires PortAudio)
try:
    from .live_voice_chat import LiveVoiceChat
    from .audio_recorder import AudioRecorder
    from .audio_player import AudioPlayer
    _LIVE_VOICE_AVAILABLE = True
except (ImportError, OSError):
    LiveVoiceChat = None
    AudioRecorder = None
    AudioPlayer = None
    _LIVE_VOICE_AVAILABLE = False

__all__ = ['LiveVoiceChatConfig', '_LIVE_VOICE_AVAILABLE']
if _LIVE_VOICE_AVAILABLE:
    __all__.extend(['LiveVoiceChat', 'AudioRecorder', 'AudioPlayer'])
