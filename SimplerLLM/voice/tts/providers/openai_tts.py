from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os
import time
from .tts_response_models import TTSFullResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


def generate_speech(
    model_name,
    voice,
    text,
    speed=1.0,
    response_format="mp3",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Generate speech from text using OpenAI TTS API.

    Args:
        model_name: TTS model to use (e.g., "tts-1", "tts-1-hd")
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        text: Text to convert to speech
        speed: Speech speed from 0.25 to 4.0 (default: 1.0)
        response_format: Audio format (mp3, opus, aac, flac, wav, pcm)
        output_path: Optional file path to save audio (if None, returns bytes)
        full_response: If True, returns TTSFullResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: TTSFullResponse object
        If output_path provided: file path string
        Otherwise: audio bytes
    """
    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key)

    if verbose:
        print(f"[OpenAI TTS] Generating speech with model={model_name}, voice={voice}, format={response_format}")

    for attempt in range(MAX_RETRIES):
        try:
            # Create speech using OpenAI API
            response = openai_client.audio.speech.create(
                model=model_name,
                voice=voice,
                input=text,
                speed=speed,
                response_format=response_format,
            )

            # Handle output - save to file or get bytes
            if output_path:
                response.stream_to_file(output_path)
                audio_data = output_path
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else None
                if verbose:
                    print(f"[OpenAI TTS] Audio saved to: {output_path} ({file_size} bytes)")
            else:
                audio_data = response.content
                file_size = len(audio_data) if audio_data else None
                if verbose:
                    print(f"[OpenAI TTS] Audio generated in memory ({file_size} bytes)")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return TTSFullResponse(
                    audio_data=audio_data,
                    model=model_name,
                    voice=voice,
                    format=response_format,
                    process_time=process_time,
                    speed=speed,
                    file_size=file_size,
                    output_path=output_path,
                    provider="OPENAI",
                    llm_provider_response=response,
                )

            return audio_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[OpenAI TTS] Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate speech after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_speech_async(
    model_name,
    voice,
    text,
    speed=1.0,
    response_format="mp3",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Async version: Generate speech from text using OpenAI TTS API.

    Args:
        model_name: TTS model to use (e.g., "tts-1", "tts-1-hd")
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        text: Text to convert to speech
        speed: Speech speed from 0.25 to 4.0 (default: 1.0)
        response_format: Audio format (mp3, opus, aac, flac, wav, pcm)
        output_path: Optional file path to save audio (if None, returns bytes)
        full_response: If True, returns TTSFullResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: TTSFullResponse object
        If output_path provided: file path string
        Otherwise: audio bytes
    """
    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key)

    if verbose:
        print(f"[OpenAI TTS] Generating speech with model={model_name}, voice={voice}, format={response_format}")

    for attempt in range(MAX_RETRIES):
        try:
            # Create speech using OpenAI API
            response = await async_openai_client.audio.speech.create(
                model=model_name,
                voice=voice,
                input=text,
                speed=speed,
                response_format=response_format,
            )

            # Handle output - save to file or get bytes
            if output_path:
                response.stream_to_file(output_path)
                audio_data = output_path
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else None
                if verbose:
                    print(f"[OpenAI TTS] Audio saved to: {output_path} ({file_size} bytes)")
            else:
                audio_data = response.content
                file_size = len(audio_data) if audio_data else None
                if verbose:
                    print(f"[OpenAI TTS] Audio generated in memory ({file_size} bytes)")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return TTSFullResponse(
                    audio_data=audio_data,
                    model=model_name,
                    voice=voice,
                    format=response_format,
                    process_time=process_time,
                    speed=speed,
                    file_size=file_size,
                    output_path=output_path,
                    provider="OPENAI",
                    llm_provider_response=response,
                )

            return audio_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[OpenAI TTS] Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate speech after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
