# SimplerLLM - Simplified interface for LLMs and Voice APIs

# Language module exports
from .language import (
    LLM,
    LLMProvider,
    ReliableLLM,
    OpenAILLM,
    GeminiLLM,
    AnthropicLLM,
    OllamaLLM,
    DeepSeekLLM,
    LLMJudge,
    JudgeMode,
    JudgeResult,
    ProviderResponse,
    ProviderEvaluation,
    EvaluationReport,
    LLMFeedbackLoop,
    FeedbackResult,
    IterationResult,
    Critique,
    EmbeddingsLLM,
    EmbeddingsProvider,
    OpenAIEmbeddings,
    VoyageEmbeddings,
    CohereEmbeddings,
)

# Voice module exports
from .voice import (
    # TTS
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
    # STT
    STT,
    STTProvider,
    OpenAISTT,
    STTFullResponse,
)

# Optional ElevenLabs TTS
try:
    from .voice import ElevenLabsTTS
except (ImportError, TypeError):
    ElevenLabsTTS = None

# Image module exports
from .image import (
    ImageGenerator,
    ImageProvider,
    ImageSize,
    OpenAIImageGenerator,
    StabilityImageGenerator,
    GoogleImageGenerator,
    ImageGenerationResponse,
)

__all__ = [
    # Language module
    'LLM',
    'LLMProvider',
    'ReliableLLM',
    'OpenAILLM',
    'GeminiLLM',
    'AnthropicLLM',
    'OllamaLLM',
    'DeepSeekLLM',
    'LLMJudge',
    'JudgeMode',
    'JudgeResult',
    'ProviderResponse',
    'ProviderEvaluation',
    'EvaluationReport',
    'LLMFeedbackLoop',
    'FeedbackResult',
    'IterationResult',
    'Critique',
    # Embeddings
    'EmbeddingsLLM',
    'EmbeddingsProvider',
    'OpenAIEmbeddings',
    'VoyageEmbeddings',
    'CohereEmbeddings',
    # Voice module - TTS
    'TTS',
    'TTSBase',
    'TTSProvider',
    'TTSResponse',
    'Voice',
    'TTSError',
    'TTSValidationError',
    'TTSProviderError',
    'TTSVoiceNotFoundError',
    'OpenAITTS',
    'ElevenLabsTTS',
    'OPENAI_VOICES',
    'OPENAI_MODELS',
    'OPENAI_FORMATS',
    'ELEVENLABS_MODELS',
    'ELEVENLABS_FORMATS',
    # Voice module - STT
    'STT',
    'STTProvider',
    'OpenAISTT',
    'STTFullResponse',
    # Image module
    'ImageGenerator',
    'ImageProvider',
    'ImageSize',
    'OpenAIImageGenerator',
    'StabilityImageGenerator',
    'GoogleImageGenerator',
    'ImageGenerationResponse',
]
