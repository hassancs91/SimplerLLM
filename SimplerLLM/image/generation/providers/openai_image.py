from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os
import time
import base64
import requests
from .image_response_models import ImageGenerationResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


def generate_image(
    prompt,
    model_name="dall-e-3",
    size="1024x1024",
    quality="standard",
    style="vivid",
    n=1,
    response_format="url",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Generate image from text prompt using OpenAI DALL-E API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (dall-e-3, dall-e-2)
        size: Image dimensions (1024x1024, 1792x1024, 1024x1792 for DALL-E 3)
        quality: Quality setting for DALL-E 3 (standard, hd)
        style: Style setting for DALL-E 3 (vivid, natural)
        n: Number of images to generate (1-10 for DALL-E 2, only 1 for DALL-E 3)
        response_format: Response format (url or b64_json)
        output_path: Optional file path to save image (if None, returns URL or bytes)
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: URL string or image bytes (depending on response_format)
    """
    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key)

    if verbose:
        print(f"[OpenAI Image] Generating image with model={model_name}, size={size}, quality={quality}, style={style}")

    for attempt in range(MAX_RETRIES):
        try:
            # Build API parameters
            params = {
                "model": model_name,
                "prompt": prompt,
                "size": size,
                "n": n,
                "response_format": response_format,
            }

            # Add DALL-E 3 specific parameters
            if model_name == "dall-e-3":
                params["quality"] = quality
                params["style"] = style

            # Create image using OpenAI API
            response = openai_client.images.generate(**params)

            # Extract image data from response (first image if multiple)
            image_response = response.data[0]
            revised_prompt = getattr(image_response, 'revised_prompt', None)

            # Handle different response formats
            if response_format == "b64_json":
                # Base64 encoded image
                image_bytes = base64.b64decode(image_response.b64_json)

                if output_path:
                    # Save to file
                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Image saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_bytes
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Image generated in memory ({file_size} bytes)")
            else:
                # URL format
                image_url = image_response.url

                if output_path:
                    # Download and save image from URL
                    img_response = requests.get(image_url)
                    img_response.raise_for_status()
                    image_bytes = img_response.content

                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Image downloaded and saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_url
                    file_size = None
                    if verbose:
                        print(f"[OpenAI Image] Image URL generated: {image_url}")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return ImageGenerationResponse(
                    image_data=image_data,
                    model=model_name,
                    prompt=prompt,
                    revised_prompt=revised_prompt,
                    size=size,
                    quality=quality if model_name == "dall-e-3" else None,
                    style=style if model_name == "dall-e-3" else None,
                    process_time=process_time,
                    provider="OPENAI_DALL_E",
                    file_size=file_size,
                    output_path=output_path,
                    llm_provider_response=response,
                )

            return image_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[OpenAI Image] Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate image after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_image_async(
    prompt,
    model_name="dall-e-3",
    size="1024x1024",
    quality="standard",
    style="vivid",
    n=1,
    response_format="url",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Async version: Generate image from text prompt using OpenAI DALL-E API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (dall-e-3, dall-e-2)
        size: Image dimensions (1024x1024, 1792x1024, 1024x1792 for DALL-E 3)
        quality: Quality setting for DALL-E 3 (standard, hd)
        style: Style setting for DALL-E 3 (vivid, natural)
        n: Number of images to generate (1-10 for DALL-E 2, only 1 for DALL-E 3)
        response_format: Response format (url or b64_json)
        output_path: Optional file path to save image (if None, returns URL or bytes)
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: URL string or image bytes (depending on response_format)
    """
    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key)

    if verbose:
        print(f"[OpenAI Image] Generating image with model={model_name}, size={size}, quality={quality}, style={style}")

    for attempt in range(MAX_RETRIES):
        try:
            # Build API parameters
            params = {
                "model": model_name,
                "prompt": prompt,
                "size": size,
                "n": n,
                "response_format": response_format,
            }

            # Add DALL-E 3 specific parameters
            if model_name == "dall-e-3":
                params["quality"] = quality
                params["style"] = style

            # Create image using OpenAI API
            response = await async_openai_client.images.generate(**params)

            # Extract image data from response (first image if multiple)
            image_response = response.data[0]
            revised_prompt = getattr(image_response, 'revised_prompt', None)

            # Handle different response formats
            if response_format == "b64_json":
                # Base64 encoded image
                image_bytes = base64.b64decode(image_response.b64_json)

                if output_path:
                    # Save to file
                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Image saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_bytes
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Image generated in memory ({file_size} bytes)")
            else:
                # URL format
                image_url = image_response.url

                if output_path:
                    # Download and save image from URL (using requests in sync mode)
                    # Note: For production, consider using aiohttp for async downloads
                    img_response = requests.get(image_url)
                    img_response.raise_for_status()
                    image_bytes = img_response.content

                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Image downloaded and saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_url
                    file_size = None
                    if verbose:
                        print(f"[OpenAI Image] Image URL generated: {image_url}")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return ImageGenerationResponse(
                    image_data=image_data,
                    model=model_name,
                    prompt=prompt,
                    revised_prompt=revised_prompt,
                    size=size,
                    quality=quality if model_name == "dall-e-3" else None,
                    style=style if model_name == "dall-e-3" else None,
                    process_time=process_time,
                    provider="OPENAI_DALL_E",
                    file_size=file_size,
                    output_path=output_path,
                    llm_provider_response=response,
                )

            return image_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[OpenAI Image] Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate image after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
