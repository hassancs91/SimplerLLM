from elevenlabs import ElevenLabs, VoiceSettings
from elevenlabs.client import AsyncElevenLabs
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
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    # ElevenLabs-specific parameters
    stability=None,
    similarity_boost=None,
    style=None,
    use_speaker_boost=None,
):
    """
    Generate speech from text using ElevenLabs TTS API.

    Args:
        model_name: TTS model to use (e.g., "eleven_turbo_v2", "eleven_multilingual_v2")
        voice: Voice ID to use (e.g., "21m00Tcm4TlvDq8ikWAM" for Rachel)
        text: Text to convert to speech
        output_path: Optional file path to save audio (if None, returns bytes)
        full_response: If True, returns TTSFullResponse with metadata
        api_key: ElevenLabs API key (uses env var if not provided)
        verbose: If True, prints progress information
        stability: Voice stability setting (0.0-1.0, default: 0.5)
        similarity_boost: Voice similarity boost (0.0-1.0, default: 0.75)
        style: Style exaggeration (0.0-1.0, default: 0.0)
        use_speaker_boost: Enable speaker boost (bool, default: True)

    Returns:
        If full_response=True: TTSFullResponse object
        If output_path provided: file path string
        Otherwise: audio bytes
    """
    start_time = time.time() if full_response else None
    client = ElevenLabs(api_key=api_key)

    if verbose:
        print(f"[ElevenLabs TTS] Generating speech with model={model_name}, voice={voice}")

    for attempt in range(MAX_RETRIES):
        try:
            # Prepare voice settings if any are specified
            voice_settings = None
            if any([stability is not None, similarity_boost is not None, style is not None, use_speaker_boost is not None]):
                voice_settings = VoiceSettings(
                    stability=stability if stability is not None else 0.5,
                    similarity_boost=similarity_boost if similarity_boost is not None else 0.75,
                    style=style if style is not None else 0.0,
                    use_speaker_boost=use_speaker_boost if use_speaker_boost is not None else True,
                )

            # Create speech using ElevenLabs API
            response = client.text_to_speech.convert(
                voice_id=voice,
                text=text,
                model_id=model_name,
                voice_settings=voice_settings,
            )

            # Collect audio data from generator
            audio_bytes = b"".join(chunk for chunk in response)

            # Handle output - save to file or return bytes
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                audio_data = output_path
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else None
                if verbose:
                    print(f"[ElevenLabs TTS] Audio saved to: {output_path} ({file_size} bytes)")
            else:
                audio_data = audio_bytes
                file_size = len(audio_data) if audio_data else None
                if verbose:
                    print(f"[ElevenLabs TTS] Audio generated in memory ({file_size} bytes)")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return TTSFullResponse(
                    audio_data=audio_data,
                    model=model_name,
                    voice=voice,
                    format="mp3",  # ElevenLabs default format
                    process_time=process_time,
                    speed=None,  # ElevenLabs doesn't use speed parameter
                    file_size=file_size,
                    output_path=output_path,
                    provider="ELEVENLABS",
                    llm_provider_response=None,  # Response is a generator, not a structured object
                )

            return audio_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[ElevenLabs TTS] Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate speech after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_speech_async(
    model_name,
    voice,
    text,
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    # ElevenLabs-specific parameters
    stability=None,
    similarity_boost=None,
    style=None,
    use_speaker_boost=None,
):
    """
    Async version: Generate speech from text using ElevenLabs TTS API.

    Args:
        model_name: TTS model to use (e.g., "eleven_turbo_v2", "eleven_multilingual_v2")
        voice: Voice ID to use (e.g., "21m00Tcm4TlvDq8ikWAM" for Rachel)
        text: Text to convert to speech
        output_path: Optional file path to save audio (if None, returns bytes)
        full_response: If True, returns TTSFullResponse with metadata
        api_key: ElevenLabs API key (uses env var if not provided)
        verbose: If True, prints progress information
        stability: Voice stability setting (0.0-1.0, default: 0.5)
        similarity_boost: Voice similarity boost (0.0-1.0, default: 0.75)
        style: Style exaggeration (0.0-1.0, default: 0.0)
        use_speaker_boost: Enable speaker boost (bool, default: True)

    Returns:
        If full_response=True: TTSFullResponse object
        If output_path provided: file path string
        Otherwise: audio bytes
    """
    start_time = time.time() if full_response else None
    async_client = AsyncElevenLabs(api_key=api_key)

    if verbose:
        print(f"[ElevenLabs TTS] Generating speech with model={model_name}, voice={voice}")

    for attempt in range(MAX_RETRIES):
        try:
            # Prepare voice settings if any are specified
            voice_settings = None
            if any([stability is not None, similarity_boost is not None, style is not None, use_speaker_boost is not None]):
                voice_settings = VoiceSettings(
                    stability=stability if stability is not None else 0.5,
                    similarity_boost=similarity_boost if similarity_boost is not None else 0.75,
                    style=style if style is not None else 0.0,
                    use_speaker_boost=use_speaker_boost if use_speaker_boost is not None else True,
                )

            # Create speech using ElevenLabs API (async)
            response = async_client.text_to_speech.convert(
                voice_id=voice,
                text=text,
                model_id=model_name,
                voice_settings=voice_settings,
            )

            # Collect audio data from async generator
            audio_bytes = b"".join([chunk async for chunk in response])

            # Handle output - save to file or return bytes
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                audio_data = output_path
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else None
                if verbose:
                    print(f"[ElevenLabs TTS] Audio saved to: {output_path} ({file_size} bytes)")
            else:
                audio_data = audio_bytes
                file_size = len(audio_data) if audio_data else None
                if verbose:
                    print(f"[ElevenLabs TTS] Audio generated in memory ({file_size} bytes)")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return TTSFullResponse(
                    audio_data=audio_data,
                    model=model_name,
                    voice=voice,
                    format="mp3",  # ElevenLabs default format
                    process_time=process_time,
                    speed=None,  # ElevenLabs doesn't use speed parameter
                    file_size=file_size,
                    output_path=output_path,
                    provider="ELEVENLABS",
                    llm_provider_response=None,  # Response is a generator, not a structured object
                )

            return audio_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[ElevenLabs TTS] Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate speech after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
