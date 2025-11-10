import SimplerLLM.voice.stt.providers.openai_stt as openai_stt
import os
from ..base import STT
from SimplerLLM.utils.custom_verbose import verbose_print


class OpenAISTT(STT):
    """
    OpenAI Speech-to-Text wrapper using Whisper.
    Provides a unified interface for OpenAI Whisper models.
    """

    def __init__(self, provider, model_name, api_key, verbose=False):
        """
        Initialize OpenAI STT instance.

        Args:
            provider: STTProvider.OPENAI
            model_name: Model to use (e.g., "whisper-1")
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        super().__init__(provider, model_name, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def transcribe(
        self,
        audio_file: str,
        model: str = None,
        language: str = None,
        prompt: str = None,
        response_format: str = "text",
        temperature: float = 0.0,
        full_response: bool = False,
    ):
        """
        Transcribe audio to text using OpenAI Whisper.

        Args:
            audio_file: Path to the audio file (required)
                       Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
            model: Model to use (None = use instance default)
                   Currently only "whisper-1" is available
            language: Language code (e.g., "en", "es", "fr") - auto-detect if None
                     Using ISO-639-1 format
            prompt: Optional text to guide the model's style or continue from previous segment
            response_format: Response format (default: text)
                            Options: text, json, verbose_json, srt, vtt
            temperature: Sampling temperature between 0 and 1 (default: 0.0)
                        Higher values make output more random
            full_response: If True, returns STTFullResponse with metadata

        Returns:
            If full_response=True: STTFullResponse object with metadata
            Otherwise: transcribed text string (or json/srt/vtt based on response_format)

        Example:
            >>> stt = STT.create(provider=STTProvider.OPENAI)
            >>> # Basic transcription
            >>> text = stt.transcribe("audio.mp3")
            >>> # With language and full response
            >>> response = stt.transcribe("audio.mp3", language="en", full_response=True)
            >>> print(response.text)
            >>> print(f"Processed in {response.process_time}s")
        """
        # Validate input
        if not audio_file:
            if self.verbose:
                verbose_print("Error: audio_file parameter is required", "error")
            raise ValueError("audio_file parameter is required for transcription")

        # Prepare parameters using base class method
        params = self.prepare_params(model, language, temperature)

        if self.verbose:
            verbose_print(
                f"Transcribing audio - Model: {params['model_name']}, Language: {params['language'] or 'auto'}, Format: {response_format}",
                "info"
            )
            verbose_print(f"Audio file: {audio_file}", "debug")

        # Update parameters with additional options
        params.update({
            "api_key": self.api_key,
            "audio_file": audio_file,
            "prompt": prompt,
            "response_format": response_format,
            "full_response": full_response,
            "verbose": self.verbose,
        })

        try:
            response = openai_stt.transcribe(**params)
            if self.verbose:
                verbose_print("Transcription completed successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error during transcription: {str(e)}", "error")
            raise

    async def transcribe_async(
        self,
        audio_file: str,
        model: str = None,
        language: str = None,
        prompt: str = None,
        response_format: str = "text",
        temperature: float = 0.0,
        full_response: bool = False,
    ):
        """
        Asynchronously transcribe audio to text using OpenAI Whisper.

        Args:
            audio_file: Path to the audio file (required)
                       Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
            model: Model to use (None = use instance default)
                   Currently only "whisper-1" is available
            language: Language code (e.g., "en", "es", "fr") - auto-detect if None
                     Using ISO-639-1 format
            prompt: Optional text to guide the model's style or continue from previous segment
            response_format: Response format (default: text)
                            Options: text, json, verbose_json, srt, vtt
            temperature: Sampling temperature between 0 and 1 (default: 0.0)
                        Higher values make output more random
            full_response: If True, returns STTFullResponse with metadata

        Returns:
            If full_response=True: STTFullResponse object with metadata
            Otherwise: transcribed text string (or json/srt/vtt based on response_format)

        Example:
            >>> stt = STT.create(provider=STTProvider.OPENAI)
            >>> text = await stt.transcribe_async("audio.mp3")
            >>> print(text)
        """
        # Validate input
        if not audio_file:
            if self.verbose:
                verbose_print("Error: audio_file parameter is required", "error")
            raise ValueError("audio_file parameter is required for transcription")

        # Prepare parameters using base class method
        params = self.prepare_params(model, language, temperature)

        if self.verbose:
            verbose_print(
                f"Transcribing audio (async) - Model: {params['model_name']}, Language: {params['language'] or 'auto'}, Format: {response_format}",
                "info"
            )
            verbose_print(f"Audio file: {audio_file}", "debug")

        # Update parameters with additional options
        params.update({
            "api_key": self.api_key,
            "audio_file": audio_file,
            "prompt": prompt,
            "response_format": response_format,
            "full_response": full_response,
            "verbose": self.verbose,
        })

        try:
            response = await openai_stt.transcribe_async(**params)
            if self.verbose:
                verbose_print("Transcription completed successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error during transcription: {str(e)}", "error")
            raise
