import SimplerLLM.voice.tts.providers.openai_tts as openai_tts
import os
from ..base import TTS
from SimplerLLM.utils.custom_verbose import verbose_print


class OpenAITTS(TTS):
    """
    OpenAI Text-to-Speech wrapper.
    Provides a unified interface for OpenAI TTS models.
    """

    def __init__(self, provider, model_name, voice, api_key, verbose=False):
        """
        Initialize OpenAI TTS instance.

        Args:
            provider: TTSProvider.OPENAI
            model_name: Model to use ("tts-1" or "tts-1-hd")
            voice: Default voice (alloy, echo, fable, onyx, nova, shimmer)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        super().__init__(provider, model_name, voice, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def generate_speech(
        self,
        text: str,
        voice: str = None,
        model: str = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        output_path: str = None,
        full_response: bool = False,
    ):
        """
        Generate speech from text using OpenAI TTS.

        Args:
            text: Text to convert to speech (required)
            voice: Voice to use (None = use instance default)
                   Options: alloy, echo, fable, onyx, nova, shimmer
            model: Model to use (None = use instance default)
                   Options: tts-1, tts-1-hd
            speed: Speech speed from 0.25 to 4.0 (default: 1.0)
            response_format: Audio format (default: mp3)
                            Options: mp3, opus, aac, flac, wav, pcm
            output_path: File path to save audio (if None, returns bytes)
            full_response: If True, returns TTSFullResponse with metadata

        Returns:
            If full_response=True: TTSFullResponse object with metadata
            If output_path provided: file path string
            Otherwise: audio bytes

        Example:
            >>> tts = OpenAITTS.create(...)
            >>> # Save to file
            >>> tts.generate_speech("Hello world", output_path="hello.mp3")
            >>> # Get bytes
            >>> audio = tts.generate_speech("Hello world")
            >>> # Get full response
            >>> response = tts.generate_speech("Hello", full_response=True)
        """
        # Validate input
        if not text:
            if self.verbose:
                verbose_print("Error: Text parameter is required", "error")
            raise ValueError("Text parameter is required for speech generation")

        # Prepare parameters using base class method
        params = self.prepare_params(voice, model, speed)

        if self.verbose:
            verbose_print(
                f"Generating speech - Model: {params['model_name']}, Voice: {params['voice']}, Format: {response_format}",
                "info"
            )
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Update parameters with additional options
        params.update({
            "api_key": self.api_key,
            "text": text,
            "response_format": response_format,
            "output_path": output_path,
            "full_response": full_response,
            "verbose": self.verbose,
        })

        try:
            response = openai_tts.generate_speech(**params)
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
        speed: float = 1.0,
        response_format: str = "mp3",
        output_path: str = None,
        full_response: bool = False,
    ):
        """
        Asynchronously generate speech from text using OpenAI TTS.

        Args:
            text: Text to convert to speech (required)
            voice: Voice to use (None = use instance default)
                   Options: alloy, echo, fable, onyx, nova, shimmer
            model: Model to use (None = use instance default)
                   Options: tts-1, tts-1-hd
            speed: Speech speed from 0.25 to 4.0 (default: 1.0)
            response_format: Audio format (default: mp3)
                            Options: mp3, opus, aac, flac, wav, pcm
            output_path: File path to save audio (if None, returns bytes)
            full_response: If True, returns TTSFullResponse with metadata

        Returns:
            If full_response=True: TTSFullResponse object with metadata
            If output_path provided: file path string
            Otherwise: audio bytes

        Example:
            >>> tts = OpenAITTS.create(...)
            >>> audio = await tts.generate_speech_async("Hello world")
        """
        # Validate input
        if not text:
            if self.verbose:
                verbose_print("Error: Text parameter is required", "error")
            raise ValueError("Text parameter is required for speech generation")

        # Prepare parameters using base class method
        params = self.prepare_params(voice, model, speed)

        if self.verbose:
            verbose_print(
                f"Generating speech (async) - Model: {params['model_name']}, Voice: {params['voice']}, Format: {response_format}",
                "info"
            )
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Update parameters with additional options
        params.update({
            "api_key": self.api_key,
            "text": text,
            "response_format": response_format,
            "output_path": output_path,
            "full_response": full_response,
            "verbose": self.verbose,
        })

        try:
            response = await openai_tts.generate_speech_async(**params)
            if self.verbose:
                verbose_print("Speech generated successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating speech: {str(e)}", "error")
            raise
