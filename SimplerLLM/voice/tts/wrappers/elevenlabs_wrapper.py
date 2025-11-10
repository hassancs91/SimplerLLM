import SimplerLLM.voice.tts.providers.elevenlabs_tts as elevenlabs_tts
import os
from ..base import TTS
from SimplerLLM.utils.custom_verbose import verbose_print


class ElevenLabsTTS(TTS):
    """
    ElevenLabs Text-to-Speech wrapper.
    Provides a unified interface for ElevenLabs TTS models.
    """

    def __init__(self, provider, model_name, voice, api_key, verbose=False):
        """
        Initialize ElevenLabs TTS instance.

        Args:
            provider: TTSProvider.ELEVENLABS
            model_name: Model to use (e.g., "eleven_turbo_v2", "eleven_multilingual_v2")
            voice: Default voice ID (e.g., "21m00Tcm4TlvDq8ikWAM" for Rachel)
            api_key: ElevenLabs API key (uses ELEVENLABS_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        super().__init__(provider, model_name, voice, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")

    def generate_speech(
        self,
        text: str,
        voice: str = None,
        model: str = None,
        output_path: str = None,
        full_response: bool = False,
        # ElevenLabs-specific parameters
        stability: float = None,
        similarity_boost: float = None,
        style: float = None,
        use_speaker_boost: bool = None,
    ):
        """
        Generate speech from text using ElevenLabs TTS.

        Args:
            text: Text to convert to speech (required)
            voice: Voice ID to use (None = use instance default)
            model: Model to use (None = use instance default)
                   Options: eleven_turbo_v2, eleven_multilingual_v2, eleven_monolingual_v1
            output_path: File path to save audio (if None, returns bytes)
            full_response: If True, returns TTSFullResponse with metadata
            stability: Voice stability (0.0-1.0, default: 0.5)
            similarity_boost: Voice similarity boost (0.0-1.0, default: 0.75)
            style: Style exaggeration (0.0-1.0, default: 0.0)
            use_speaker_boost: Enable speaker boost (bool, default: True)

        Returns:
            If full_response=True: TTSFullResponse object with metadata
            If output_path provided: file path string
            Otherwise: audio bytes

        Example:
            >>> tts = TTS.create(provider=TTSProvider.ELEVENLABS, ...)
            >>> # Save to file
            >>> tts.generate_speech("Hello world", output_path="hello.mp3")
            >>> # Get bytes
            >>> audio = tts.generate_speech("Hello world")
            >>> # Get full response with custom voice settings
            >>> response = tts.generate_speech(
            ...     "Hello",
            ...     full_response=True,
            ...     stability=0.7,
            ...     similarity_boost=0.8
            ... )
        """
        # Validate input
        if not text:
            if self.verbose:
                verbose_print("Error: Text parameter is required", "error")
            raise ValueError("Text parameter is required for speech generation")

        # Prepare base parameters using base class method
        # Note: prepare_params includes 'speed' which ElevenLabs doesn't use, we'll ignore it
        params = self.prepare_params(voice, model, speed=None)

        if self.verbose:
            verbose_print(
                f"Generating speech - Model: {params['model_name']}, Voice: {params['voice']}",
                "info"
            )
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")
            if any([stability, similarity_boost, style, use_speaker_boost is not None]):
                verbose_print(
                    f"Voice settings - Stability: {stability}, Similarity: {similarity_boost}, Style: {style}, Speaker boost: {use_speaker_boost}",
                    "debug"
                )

        # Build parameters for provider call
        provider_params = {
            "model_name": params['model_name'],
            "voice": params['voice'],
            "text": text,
            "output_path": output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add ElevenLabs-specific parameters if provided
        if stability is not None:
            provider_params["stability"] = stability
        if similarity_boost is not None:
            provider_params["similarity_boost"] = similarity_boost
        if style is not None:
            provider_params["style"] = style
        if use_speaker_boost is not None:
            provider_params["use_speaker_boost"] = use_speaker_boost

        try:
            response = elevenlabs_tts.generate_speech(**provider_params)
            if self.verbose:
                verbose_print("Speech generated successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating speech: {str(e)}", "error")
            raise

    async def generate_speech_async(
        self,
        text: str,
        voice: str = None,
        model: str = None,
        output_path: str = None,
        full_response: bool = False,
        # ElevenLabs-specific parameters
        stability: float = None,
        similarity_boost: float = None,
        style: float = None,
        use_speaker_boost: bool = None,
    ):
        """
        Asynchronously generate speech from text using ElevenLabs TTS.

        Args:
            text: Text to convert to speech (required)
            voice: Voice ID to use (None = use instance default)
            model: Model to use (None = use instance default)
                   Options: eleven_turbo_v2, eleven_multilingual_v2, eleven_monolingual_v1
            output_path: File path to save audio (if None, returns bytes)
            full_response: If True, returns TTSFullResponse with metadata
            stability: Voice stability (0.0-1.0, default: 0.5)
            similarity_boost: Voice similarity boost (0.0-1.0, default: 0.75)
            style: Style exaggeration (0.0-1.0, default: 0.0)
            use_speaker_boost: Enable speaker boost (bool, default: True)

        Returns:
            If full_response=True: TTSFullResponse object with metadata
            If output_path provided: file path string
            Otherwise: audio bytes

        Example:
            >>> tts = TTS.create(provider=TTSProvider.ELEVENLABS, ...)
            >>> audio = await tts.generate_speech_async("Hello world")
        """
        # Validate input
        if not text:
            if self.verbose:
                verbose_print("Error: Text parameter is required", "error")
            raise ValueError("Text parameter is required for speech generation")

        # Prepare base parameters using base class method
        params = self.prepare_params(voice, model, speed=None)

        if self.verbose:
            verbose_print(
                f"Generating speech (async) - Model: {params['model_name']}, Voice: {params['voice']}",
                "info"
            )
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")
            if any([stability, similarity_boost, style, use_speaker_boost is not None]):
                verbose_print(
                    f"Voice settings - Stability: {stability}, Similarity: {similarity_boost}, Style: {style}, Speaker boost: {use_speaker_boost}",
                    "debug"
                )

        # Build parameters for provider call
        provider_params = {
            "model_name": params['model_name'],
            "voice": params['voice'],
            "text": text,
            "output_path": output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add ElevenLabs-specific parameters if provided
        if stability is not None:
            provider_params["stability"] = stability
        if similarity_boost is not None:
            provider_params["similarity_boost"] = similarity_boost
        if style is not None:
            provider_params["style"] = style
        if use_speaker_boost is not None:
            provider_params["use_speaker_boost"] = use_speaker_boost

        try:
            response = await elevenlabs_tts.generate_speech_async(**provider_params)
            if self.verbose:
                verbose_print("Speech generated successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating speech: {str(e)}", "error")
            raise
