from .tts import TTS, TTSProvider, OpenAITTS, ElevenLabsTTS, TTSFullResponse
from .stt import STT, STTProvider, OpenAISTT, STTFullResponse
from .voice_chat import (
    VoiceChat,
    VoiceChatConfig,
    ConversationMessage,
    ConversationRole,
    VoiceTurnResult,
    VoiceChatSession,
    ConversationManager
)
from .live_voice_chat import LiveVoiceChatConfig

# LiveVoiceChat requires sounddevice/pynput which need PortAudio
try:
    from .live_voice_chat import (
        LiveVoiceChat,
        AudioRecorder,
        AudioPlayer
    )
    _LIVE_VOICE_AVAILABLE = True
except (ImportError, OSError):
    LiveVoiceChat = None
    AudioRecorder = None
    AudioPlayer = None
    _LIVE_VOICE_AVAILABLE = False
from .dialogue_generator import (
    DialogueGenerator,
    Dialogue,
    DialogueLine,
    SpeakerConfig,
    DialogueGenerationConfig,
    AudioDialogueResult,
    DialogueStyle
)
from .video_transcription import (
    VideoTranscriber,
    MultiLanguageCaptionGenerator,
    VideoTranscriptionResult,
    CaptionSegment,
    LanguageCaptions,
    MultiLanguageCaptionsResult
)
from .video_dubbing import (
    VideoDubber,
    DubbedSegment,
    DubbingConfig,
    VideoDubbingResult
)
from .realtime_voice import (
    RealtimeVoice,
    RealtimeVoiceProvider,
    RealtimeSessionConfig,
    OpenAIRealtimeVoice,
    TurnDetectionType,
    Voice,
    AudioFormat,
    Modality,
    RealtimeVoiceChat,
    RealtimeVoiceChatConfig
)

__all__ = [
    # TTS
    'TTS',
    'TTSProvider',
    'OpenAITTS',
    'ElevenLabsTTS',
    'TTSFullResponse',
    # STT
    'STT',
    'STTProvider',
    'OpenAISTT',
    'STTFullResponse',
    # VoiceChat
    'VoiceChat',
    'VoiceChatConfig',
    'ConversationMessage',
    'ConversationRole',
    'VoiceTurnResult',
    'VoiceChatSession',
    'ConversationManager',
    # LiveVoiceChat (config always available, others require PortAudio)
    'LiveVoiceChatConfig',
    '_LIVE_VOICE_AVAILABLE',
    # Dialogue Generator
    'DialogueGenerator',
    'Dialogue',
    'DialogueLine',
    'SpeakerConfig',
    'DialogueGenerationConfig',
    'AudioDialogueResult',
    'DialogueStyle',
    # Video Transcription
    'VideoTranscriber',
    'MultiLanguageCaptionGenerator',
    'VideoTranscriptionResult',
    'CaptionSegment',
    'LanguageCaptions',
    'MultiLanguageCaptionsResult',
    # Video Dubbing
    'VideoDubber',
    'DubbedSegment',
    'DubbingConfig',
    'VideoDubbingResult',
    # Realtime Voice
    'RealtimeVoice',
    'RealtimeVoiceProvider',
    'RealtimeSessionConfig',
    'OpenAIRealtimeVoice',
    'TurnDetectionType',
    'Voice',
    'AudioFormat',
    'Modality',
    'RealtimeVoiceChat',
    'RealtimeVoiceChatConfig',
]

# Conditionally add LiveVoiceChat exports if available
if _LIVE_VOICE_AVAILABLE:
    __all__.extend(['LiveVoiceChat', 'AudioRecorder', 'AudioPlayer'])
