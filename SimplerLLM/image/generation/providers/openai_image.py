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


def _is_gpt_image_model(model_name):
    """Check if the model is a GPT image model (not DALL-E)."""
    return model_name.startswith("gpt-image")


def generate_image(
    prompt,
    model_name="dall-e-3",
    size="1024x1024",
    quality="standard",
    style="vivid",
    n=1,
    response_format="url",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Generate image from text prompt using OpenAI Image API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (dall-e-3, dall-e-2, gpt-image-1, gpt-image-1.5)
        size: Image dimensions
              - DALL-E 3: 1024x1024, 1792x1024, 1024x1792
              - GPT Image: 1024x1024, 1536x1024, 1024x1536, auto
        quality: Quality setting
                 - DALL-E 3: "standard", "hd"
                 - GPT Image: "low", "medium", "high", "auto"
        style: Style setting for DALL-E 3 only (vivid, natural)
        n: Number of images to generate (1-10 for DALL-E 2, only 1 for others)
        response_format: For DALL-E models only - "url" or "b64_json"
        output_format: For GPT Image models only - "png", "jpeg", or "webp"
        output_path: Optional file path to save image (if None, returns URL or bytes)
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: URL string (DALL-E) or image bytes (GPT Image)
    """
    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key)
    is_gpt_image = _is_gpt_image_model(model_name)

    if verbose:
        if is_gpt_image:
            print(f"[OpenAI Image] Generating image with model={model_name}, size={size}, quality={quality}, output_format={output_format}")
        else:
            print(f"[OpenAI Image] Generating image with model={model_name}, size={size}, quality={quality}, style={style}")

    for attempt in range(MAX_RETRIES):
        try:
            # Build API parameters based on model type
            params = {
                "model": model_name,
                "prompt": prompt,
                "size": size,
                "n": n,
            }

            if is_gpt_image:
                # GPT Image models use output_format and always return base64
                params["output_format"] = output_format
                params["quality"] = quality
            else:
                # DALL-E models use response_format
                params["response_format"] = response_format
                if model_name == "dall-e-3":
                    params["quality"] = quality
                    params["style"] = style

            # Create image using OpenAI API
            response = openai_client.images.generate(**params)

            # Extract image data from response (first image if multiple)
            image_response = response.data[0]
            revised_prompt = getattr(image_response, 'revised_prompt', None)

            # Handle response based on model type
            if is_gpt_image or response_format == "b64_json":
                # GPT Image models always return base64, DALL-E with b64_json also
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
                # DALL-E URL format
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
                    quality=quality,
                    style=style if not is_gpt_image and model_name == "dall-e-3" else None,
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
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Async version: Generate image from text prompt using OpenAI Image API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (dall-e-3, dall-e-2, gpt-image-1, gpt-image-1.5)
        size: Image dimensions
              - DALL-E 3: 1024x1024, 1792x1024, 1024x1792
              - GPT Image: 1024x1024, 1536x1024, 1024x1536, auto
        quality: Quality setting
                 - DALL-E 3: "standard", "hd"
                 - GPT Image: "low", "medium", "high", "auto"
        style: Style setting for DALL-E 3 only (vivid, natural)
        n: Number of images to generate (1-10 for DALL-E 2, only 1 for others)
        response_format: For DALL-E models only - "url" or "b64_json"
        output_format: For GPT Image models only - "png", "jpeg", or "webp"
        output_path: Optional file path to save image (if None, returns URL or bytes)
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: URL string (DALL-E) or image bytes (GPT Image)
    """
    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key)
    is_gpt_image = _is_gpt_image_model(model_name)

    if verbose:
        if is_gpt_image:
            print(f"[OpenAI Image] Generating image with model={model_name}, size={size}, quality={quality}, output_format={output_format}")
        else:
            print(f"[OpenAI Image] Generating image with model={model_name}, size={size}, quality={quality}, style={style}")

    for attempt in range(MAX_RETRIES):
        try:
            # Build API parameters based on model type
            params = {
                "model": model_name,
                "prompt": prompt,
                "size": size,
                "n": n,
            }

            if is_gpt_image:
                # GPT Image models use output_format and always return base64
                params["output_format"] = output_format
                params["quality"] = quality
            else:
                # DALL-E models use response_format
                params["response_format"] = response_format
                if model_name == "dall-e-3":
                    params["quality"] = quality
                    params["style"] = style

            # Create image using OpenAI API
            response = await async_openai_client.images.generate(**params)

            # Extract image data from response (first image if multiple)
            image_response = response.data[0]
            revised_prompt = getattr(image_response, 'revised_prompt', None)

            # Handle response based on model type
            if is_gpt_image or response_format == "b64_json":
                # GPT Image models always return base64, DALL-E with b64_json also
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
                # DALL-E URL format
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
                    quality=quality,
                    style=style if not is_gpt_image and model_name == "dall-e-3" else None,
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


def edit_image(
    image_source,
    edit_prompt,
    mask_source=None,
    model_name="dall-e-2",
    size="1024x1024",
    n=1,
    response_format="url",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Edit an existing image using OpenAI's image edit API.

    Note: Only DALL-E 2 supports image editing. DALL-E 3 and GPT Image models do NOT support editing.

    Args:
        image_source: Source image to edit. Can be:
                     - str: File path to PNG image (must be square, < 4MB)
                     - bytes: Raw PNG image data
        edit_prompt: Text description of the desired edit
        mask_source: Optional mask image. Transparent areas indicate where to edit.
                    Can be str (file path) or bytes (raw data). Must be same size as image.
        model_name: Model to use (only "dall-e-2" supported for editing)
        size: Output size - "256x256", "512x512", or "1024x1024"
        n: Number of images to generate (1-10)
        response_format: Response format - "url" or "b64_json"
        output_path: Optional file path to save edited image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: URL string or image bytes (depending on response_format)

    Raises:
        ValueError: If model doesn't support editing (only dall-e-2 supported)
    """
    # Validate model supports editing
    if model_name != "dall-e-2":
        raise ValueError(f"Image editing is only supported by dall-e-2. Model '{model_name}' does not support editing.")

    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key)

    if verbose:
        print(f"[OpenAI Image] Editing image with model={model_name}, size={size}")
        print(f"[OpenAI Image] Edit prompt: {edit_prompt}")

    for attempt in range(MAX_RETRIES):
        try:
            # Prepare image file
            if isinstance(image_source, str):
                image_file = open(image_source, "rb")
            else:
                # Bytes - wrap in a file-like object
                import io
                image_file = io.BytesIO(image_source)
                image_file.name = "image.png"

            # Prepare mask file if provided
            mask_file = None
            if mask_source:
                if isinstance(mask_source, str):
                    mask_file = open(mask_source, "rb")
                else:
                    import io
                    mask_file = io.BytesIO(mask_source)
                    mask_file.name = "mask.png"

            # Build API parameters
            params = {
                "model": model_name,
                "image": image_file,
                "prompt": edit_prompt,
                "size": size,
                "n": n,
                "response_format": response_format,
            }

            if mask_file:
                params["mask"] = mask_file

            # Call edit API
            response = openai_client.images.edit(**params)

            # Close files
            if isinstance(image_source, str):
                image_file.close()
            if mask_source and isinstance(mask_source, str):
                mask_file.close()

            # Extract image data from response
            image_response = response.data[0]

            # Handle different response formats
            if response_format == "b64_json":
                image_bytes = base64.b64decode(image_response.b64_json)

                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Edited image saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_bytes
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Edited image generated in memory ({file_size} bytes)")
            else:
                image_url = image_response.url

                if output_path:
                    img_response = requests.get(image_url)
                    img_response.raise_for_status()
                    image_bytes = img_response.content

                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Edited image downloaded and saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_url
                    file_size = None
                    if verbose:
                        print(f"[OpenAI Image] Edited image URL generated: {image_url}")

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return ImageGenerationResponse(
                    image_data=image_data,
                    model=model_name,
                    prompt=edit_prompt,
                    revised_prompt=None,
                    size=size,
                    quality=None,
                    style=None,
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
                    print(f"[OpenAI Image] Edit attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to edit image after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def edit_image_async(
    image_source,
    edit_prompt,
    mask_source=None,
    model_name="dall-e-2",
    size="1024x1024",
    n=1,
    response_format="url",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Async version: Edit an existing image using OpenAI's image edit API.

    Note: Only DALL-E 2 supports image editing. DALL-E 3 and GPT Image models do NOT support editing.

    Args:
        image_source: Source image to edit. Can be:
                     - str: File path to PNG image (must be square, < 4MB)
                     - bytes: Raw PNG image data
        edit_prompt: Text description of the desired edit
        mask_source: Optional mask image. Transparent areas indicate where to edit.
                    Can be str (file path) or bytes (raw data). Must be same size as image.
        model_name: Model to use (only "dall-e-2" supported for editing)
        size: Output size - "256x256", "512x512", or "1024x1024"
        n: Number of images to generate (1-10)
        response_format: Response format - "url" or "b64_json"
        output_path: Optional file path to save edited image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: URL string or image bytes (depending on response_format)

    Raises:
        ValueError: If model doesn't support editing (only dall-e-2 supported)
    """
    # Validate model supports editing
    if model_name != "dall-e-2":
        raise ValueError(f"Image editing is only supported by dall-e-2. Model '{model_name}' does not support editing.")

    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key)

    if verbose:
        print(f"[OpenAI Image] Editing image (async) with model={model_name}, size={size}")
        print(f"[OpenAI Image] Edit prompt: {edit_prompt}")

    for attempt in range(MAX_RETRIES):
        try:
            # Prepare image file
            if isinstance(image_source, str):
                image_file = open(image_source, "rb")
            else:
                import io
                image_file = io.BytesIO(image_source)
                image_file.name = "image.png"

            # Prepare mask file if provided
            mask_file = None
            if mask_source:
                if isinstance(mask_source, str):
                    mask_file = open(mask_source, "rb")
                else:
                    import io
                    mask_file = io.BytesIO(mask_source)
                    mask_file.name = "mask.png"

            # Build API parameters
            params = {
                "model": model_name,
                "image": image_file,
                "prompt": edit_prompt,
                "size": size,
                "n": n,
                "response_format": response_format,
            }

            if mask_file:
                params["mask"] = mask_file

            # Call edit API
            response = await async_openai_client.images.edit(**params)

            # Close files
            if isinstance(image_source, str):
                image_file.close()
            if mask_source and isinstance(mask_source, str):
                mask_file.close()

            # Extract image data from response
            image_response = response.data[0]

            # Handle different response formats
            if response_format == "b64_json":
                image_bytes = base64.b64decode(image_response.b64_json)

                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Edited image saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_bytes
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Edited image generated in memory ({file_size} bytes)")
            else:
                image_url = image_response.url

                if output_path:
                    img_response = requests.get(image_url)
                    img_response.raise_for_status()
                    image_bytes = img_response.content

                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[OpenAI Image] Edited image downloaded and saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_url
                    file_size = None
                    if verbose:
                        print(f"[OpenAI Image] Edited image URL generated: {image_url}")

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return ImageGenerationResponse(
                    image_data=image_data,
                    model=model_name,
                    prompt=edit_prompt,
                    revised_prompt=None,
                    size=size,
                    quality=None,
                    style=None,
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
                    print(f"[OpenAI Image] Edit attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to edit image after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)
