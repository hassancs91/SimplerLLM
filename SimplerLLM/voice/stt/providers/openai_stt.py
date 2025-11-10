from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os
import time
from .stt_response_models import STTFullResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


def transcribe(
    model_name,
    audio_file,
    api_key=None,
    language=None,
    prompt=None,
    response_format="text",
    temperature=0.0,
    full_response=False,
    verbose=False,
):
    """
    Transcribe audio to text using OpenAI Whisper API.

    Args:
        model_name: STT model to use (e.g., "whisper-1")
        audio_file: Path to the audio file or file-like object
        api_key: OpenAI API key (uses env var if not provided)
        language: Language code (e.g., "en", "es", "fr") - auto-detect if None
        prompt: Optional text to guide the model's style
        response_format: Response format (text, json, verbose_json, srt, vtt)
        temperature: Sampling temperature between 0 and 1 (default: 0.0)
        full_response: If True, returns STTFullResponse with metadata
        verbose: If True, prints progress information

    Returns:
        If full_response=True: STTFullResponse object
        Otherwise: transcribed text string (or json/srt/vtt based on response_format)
    """
    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key)

    if verbose:
        print(f"[OpenAI STT] Transcribing audio with model={model_name}, language={language or 'auto'}, format={response_format}")

    for attempt in range(MAX_RETRIES):
        try:
            # Open audio file and create transcription
            with open(audio_file, "rb") as audio:
                # Build API call parameters
                api_params = {
                    "model": model_name,
                    "file": audio,
                    "response_format": response_format,
                    "temperature": temperature,
                }

                # Add optional parameters only if provided
                if language:
                    api_params["language"] = language
                if prompt:
                    api_params["prompt"] = prompt

                # Create transcription using OpenAI API
                response = openai_client.audio.transcriptions.create(**api_params)

            # Extract text from response based on format
            if response_format == "text":
                transcribed_text = response
            elif response_format in ["json", "verbose_json"]:
                transcribed_text = response.text if hasattr(response, 'text') else response
            else:
                # For srt, vtt formats
                transcribed_text = response

            if verbose:
                preview = str(transcribed_text)[:100] + "..." if len(str(transcribed_text)) > 100 else str(transcribed_text)
                print(f"[OpenAI STT] Transcription completed: {preview}")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time

                # Extract language from verbose response if available
                detected_language = language
                if response_format == "verbose_json" and hasattr(response, 'language'):
                    detected_language = response.language

                # Extract duration if available
                duration = None
                if response_format == "verbose_json" and hasattr(response, 'duration'):
                    duration = response.duration

                return STTFullResponse(
                    text=str(transcribed_text),
                    model=model_name,
                    language=detected_language,
                    process_time=process_time,
                    provider="OPENAI",
                    audio_file=audio_file,
                    duration=duration,
                    response_format=response_format,
                    llm_provider_response=response,
                )

            return transcribed_text

        except FileNotFoundError:
            error_msg = f"Audio file not found: {audio_file}"
            raise FileNotFoundError(error_msg)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[OpenAI STT] Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to transcribe audio after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def transcribe_async(
    model_name,
    audio_file,
    api_key=None,
    language=None,
    prompt=None,
    response_format="text",
    temperature=0.0,
    full_response=False,
    verbose=False,
):
    """
    Async version: Transcribe audio to text using OpenAI Whisper API.

    Args:
        model_name: STT model to use (e.g., "whisper-1")
        audio_file: Path to the audio file or file-like object
        api_key: OpenAI API key (uses env var if not provided)
        language: Language code (e.g., "en", "es", "fr") - auto-detect if None
        prompt: Optional text to guide the model's style
        response_format: Response format (text, json, verbose_json, srt, vtt)
        temperature: Sampling temperature between 0 and 1 (default: 0.0)
        full_response: If True, returns STTFullResponse with metadata
        verbose: If True, prints progress information

    Returns:
        If full_response=True: STTFullResponse object
        Otherwise: transcribed text string (or json/srt/vtt based on response_format)
    """
    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key)

    if verbose:
        print(f"[OpenAI STT] Transcribing audio with model={model_name}, language={language or 'auto'}, format={response_format}")

    for attempt in range(MAX_RETRIES):
        try:
            # Open audio file and create transcription
            with open(audio_file, "rb") as audio:
                # Build API call parameters
                api_params = {
                    "model": model_name,
                    "file": audio,
                    "response_format": response_format,
                    "temperature": temperature,
                }

                # Add optional parameters only if provided
                if language:
                    api_params["language"] = language
                if prompt:
                    api_params["prompt"] = prompt

                # Create transcription using OpenAI API
                response = await async_openai_client.audio.transcriptions.create(**api_params)

            # Extract text from response based on format
            if response_format == "text":
                transcribed_text = response
            elif response_format in ["json", "verbose_json"]:
                transcribed_text = response.text if hasattr(response, 'text') else response
            else:
                # For srt, vtt formats
                transcribed_text = response

            if verbose:
                preview = str(transcribed_text)[:100] + "..." if len(str(transcribed_text)) > 100 else str(transcribed_text)
                print(f"[OpenAI STT] Transcription completed: {preview}")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time

                # Extract language from verbose response if available
                detected_language = language
                if response_format == "verbose_json" and hasattr(response, 'language'):
                    detected_language = response.language

                # Extract duration if available
                duration = None
                if response_format == "verbose_json" and hasattr(response, 'duration'):
                    duration = response.duration

                return STTFullResponse(
                    text=str(transcribed_text),
                    model=model_name,
                    language=detected_language,
                    process_time=process_time,
                    provider="OPENAI",
                    audio_file=audio_file,
                    duration=duration,
                    response_format=response_format,
                    llm_provider_response=response,
                )

            return transcribed_text

        except FileNotFoundError:
            error_msg = f"Audio file not found: {audio_file}"
            raise FileNotFoundError(error_msg)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[OpenAI STT] Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to transcribe audio after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
