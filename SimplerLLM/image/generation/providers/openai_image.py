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


def _load_image_data(image_source):
    """
    Load image data from various source types.

    Args:
        image_source: Can be:
            - str: File path to image
            - bytes: Raw image data

    Returns:
        tuple: (image_data_bytes, filename)

    Raises:
        ValueError: If image source is invalid
        FileNotFoundError: If file path doesn't exist
    """
    if isinstance(image_source, bytes):
        return image_source, "image.png"

    elif isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")

        with open(image_source, 'rb') as f:
            image_data = f.read()

        filename = os.path.basename(image_source)
        return image_data, filename

    else:
        raise ValueError(f"Invalid image source type: {type(image_source)}. Must be str (path) or bytes")


def _convert_to_png(image_source, verbose=False):
    """
    Convert image to PNG format for DALL-E 2 compatibility.

    DALL-E 2 only accepts PNG format for image editing.
    This function converts JPEG and other formats to PNG.

    Args:
        image_source: Image file path (str) or raw bytes
        verbose: If True, prints conversion info

    Returns:
        bytes: PNG image data
    """
    import io
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL/Pillow is required for image format conversion. "
            "Install it with: pip install Pillow"
        )

    if isinstance(image_source, str):
        img = Image.open(image_source)
    else:
        img = Image.open(io.BytesIO(image_source))

    # Convert to RGBA for transparency support (required by DALL-E 2)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    output = io.BytesIO()
    img.save(output, format='PNG')
    png_bytes = output.getvalue()

    if verbose:
        print(f"[OpenAI Image] Converted image to PNG format ({len(png_bytes)} bytes)")

    return png_bytes


def _is_jpeg(image_source):
    """Check if image source is JPEG format."""
    if isinstance(image_source, str):
        return image_source.lower().endswith(('.jpg', '.jpeg'))
    elif isinstance(image_source, bytes):
        # JPEG files start with FFD8
        return len(image_source) >= 2 and image_source[:2] == b'\xff\xd8'
    return False


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
    reference_images=None,
    input_fidelity="low",
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
        reference_images: Optional list of reference images for GPT Image models.
                         Up to 16 images for compositing and style reference.
                         Each item can be str (file path) or bytes.
                         Note: gpt-image-1 preserves first image with higher fidelity,
                         gpt-image-1.5 preserves first 5 images with higher fidelity.
        input_fidelity: For GPT Image models - "low" (default) or "high".
                       Controls how precisely the model preserves details from input images.

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
            if reference_images:
                print(f"[OpenAI Image] Using {len(reference_images)} reference image(s), input_fidelity={input_fidelity}")
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

                # Note: Reference images are NOT supported in images.generate()
                # They must be used with images.edit() instead
                if reference_images:
                    print("[OpenAI Image] WARNING: reference_images parameter is not supported for generate_image().")
                    print("[OpenAI Image] To use reference images, use edit_image() instead with the image parameter.")
                    print("[OpenAI Image] The reference images will be ignored for this generation.")
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
    reference_images=None,
    input_fidelity="low",
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
        reference_images: Optional list of reference images for GPT Image models.
                         Up to 16 images for compositing and style reference.
                         Each item can be str (file path) or bytes.
        input_fidelity: For GPT Image models - "low" (default) or "high".
                       Controls how precisely the model preserves details from input images.

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
            print(f"[OpenAI Image] Generating image (async) with model={model_name}, size={size}, quality={quality}, output_format={output_format}")
            if reference_images:
                print(f"[OpenAI Image] Using {len(reference_images)} reference image(s), input_fidelity={input_fidelity}")
        else:
            print(f"[OpenAI Image] Generating image (async) with model={model_name}, size={size}, quality={quality}, style={style}")

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

                # Note: Reference images are NOT supported in images.generate()
                # They must be used with images.edit() instead
                if reference_images:
                    print("[OpenAI Image] WARNING: reference_images parameter is not supported for generate_image_async().")
                    print("[OpenAI Image] To use reference images, use edit_image_async() instead with the image parameter.")
                    print("[OpenAI Image] The reference images will be ignored for this generation.")
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
    output_format="png",
    quality="medium",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    input_fidelity="low",
):
    """
    Edit an existing image using OpenAI's image edit API.

    Supports both DALL-E 2 and GPT Image models for editing.

    Args:
        image_source: Source image to edit. Can be:
                     - str: File path to image
                     - bytes: Raw image data
                     For DALL-E 2: Must be PNG, square, < 4MB
                     For GPT Image: Supports various formats
        edit_prompt: Text description of the desired edit
        mask_source: Optional mask image (DALL-E 2 only). Transparent areas indicate where to edit.
                    Can be str (file path) or bytes (raw data). Must be same size as image.
        model_name: Model to use - "dall-e-2", "gpt-image-1", or "gpt-image-1.5"
        size: Output size
              - DALL-E 2: "256x256", "512x512", or "1024x1024"
              - GPT Image: "1024x1024", "1536x1024", "1024x1536", or "auto"
        n: Number of images to generate (1-10 for DALL-E 2, 1 for GPT Image)
        response_format: For DALL-E 2 only - "url" or "b64_json"
        output_format: For GPT Image only - "png", "jpeg", or "webp"
        quality: For GPT Image only - "low", "medium", "high", or "auto"
        output_path: Optional file path to save edited image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information
        input_fidelity: For GPT Image only - "low" (default) or "high".
                       Controls how precisely the model preserves details from input images.

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: URL string (DALL-E 2) or image bytes (GPT Image)

    Raises:
        ValueError: If model doesn't support editing
    """
    is_gpt_image = _is_gpt_image_model(model_name)

    # Validate model supports editing
    if not is_gpt_image and model_name != "dall-e-2":
        raise ValueError(f"Image editing is only supported by dall-e-2 and gpt-image models. Model '{model_name}' does not support editing.")

    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key)

    if verbose:
        if is_gpt_image:
            print(f"[OpenAI Image] Editing image with model={model_name}, size={size}, quality={quality}, input_fidelity={input_fidelity}")
        else:
            print(f"[OpenAI Image] Editing image with model={model_name}, size={size}")
        print(f"[OpenAI Image] Edit prompt: {edit_prompt}")

    for attempt in range(MAX_RETRIES):
        try:
            import io

            # Prepare image file
            # For DALL-E 2, convert JPEG to PNG (required format)
            if not is_gpt_image and _is_jpeg(image_source):
                if verbose:
                    print("[OpenAI Image] Converting JPEG to PNG for DALL-E 2 compatibility...")
                png_data = _convert_to_png(image_source, verbose=verbose)
                image_file = io.BytesIO(png_data)
                image_file.name = "image.png"
            elif isinstance(image_source, str):
                image_file = open(image_source, "rb")
            else:
                # Bytes - wrap in a file-like object
                image_file = io.BytesIO(image_source)
                image_file.name = "image.png"

            # Build API parameters based on model type
            if is_gpt_image:
                # GPT Image models
                # Note: The OpenAI SDK's images.edit() has limited parameter support.
                # Parameters like output_format, quality, and input_fidelity are NOT
                # supported in the current SDK version, even though the API docs mention them.
                # See: https://community.openai.com/t/error-using-gpt-image-1-api-with-quality-parameter/1239987
                params = {
                    "model": model_name,
                    "image": image_file,
                    "prompt": edit_prompt,
                    "size": size,
                    "n": 1,  # GPT Image only supports 1
                }
                if verbose and (quality or input_fidelity):
                    print(f"[OpenAI Image] Note: quality and input_fidelity parameters are not supported by the SDK for images.edit()")
            else:
                # DALL-E 2
                params = {
                    "model": model_name,
                    "image": image_file,
                    "prompt": edit_prompt,
                    "size": size,
                    "n": n,
                    "response_format": response_format,
                }

                # Prepare mask file if provided (DALL-E 2 only)
                # Also convert mask to PNG if needed
                mask_file = None
                if mask_source:
                    if _is_jpeg(mask_source):
                        png_mask = _convert_to_png(mask_source, verbose=verbose)
                        mask_file = io.BytesIO(png_mask)
                        mask_file.name = "mask.png"
                    elif isinstance(mask_source, str):
                        mask_file = open(mask_source, "rb")
                    else:
                        mask_file = io.BytesIO(mask_source)
                        mask_file.name = "mask.png"
                    params["mask"] = mask_file

            # Call edit API
            response = openai_client.images.edit(**params)

            # Close files
            if isinstance(image_source, str) and not _is_jpeg(image_source):
                image_file.close()
            if not is_gpt_image and mask_source and isinstance(mask_source, str) and not _is_jpeg(mask_source):
                mask_file.close()

            # Extract image data from response
            image_response = response.data[0]

            # Handle different response formats
            if is_gpt_image or response_format == "b64_json":
                # GPT Image always returns base64
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
                    revised_prompt=getattr(image_response, 'revised_prompt', None),
                    size=size,
                    quality=quality if is_gpt_image else None,
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
    output_format="png",
    quality="medium",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    input_fidelity="low",
):
    """
    Async version: Edit an existing image using OpenAI's image edit API.

    Supports both DALL-E 2 and GPT Image models for editing.

    Args:
        image_source: Source image to edit. Can be:
                     - str: File path to image
                     - bytes: Raw image data
        edit_prompt: Text description of the desired edit
        mask_source: Optional mask image (DALL-E 2 only). Transparent areas indicate where to edit.
                    Can be str (file path) or bytes (raw data). Must be same size as image.
        model_name: Model to use - "dall-e-2", "gpt-image-1", or "gpt-image-1.5"
        size: Output size
              - DALL-E 2: "256x256", "512x512", or "1024x1024"
              - GPT Image: "1024x1024", "1536x1024", "1024x1536", or "auto"
        n: Number of images to generate (1-10 for DALL-E 2, 1 for GPT Image)
        response_format: For DALL-E 2 only - "url" or "b64_json"
        output_format: For GPT Image only - "png", "jpeg", or "webp"
        quality: For GPT Image only - "low", "medium", "high", or "auto"
        output_path: Optional file path to save edited image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: OpenAI API key (uses env var if not provided)
        verbose: If True, prints progress information
        input_fidelity: For GPT Image only - "low" (default) or "high".

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: URL string (DALL-E 2) or image bytes (GPT Image)

    Raises:
        ValueError: If model doesn't support editing
    """
    is_gpt_image = _is_gpt_image_model(model_name)

    # Validate model supports editing
    if not is_gpt_image and model_name != "dall-e-2":
        raise ValueError(f"Image editing is only supported by dall-e-2 and gpt-image models. Model '{model_name}' does not support editing.")

    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key)

    if verbose:
        if is_gpt_image:
            print(f"[OpenAI Image] Editing image (async) with model={model_name}, size={size}, quality={quality}, input_fidelity={input_fidelity}")
        else:
            print(f"[OpenAI Image] Editing image (async) with model={model_name}, size={size}")
        print(f"[OpenAI Image] Edit prompt: {edit_prompt}")

    for attempt in range(MAX_RETRIES):
        try:
            import io

            # Prepare image file
            # For DALL-E 2, convert JPEG to PNG (required format)
            if not is_gpt_image and _is_jpeg(image_source):
                if verbose:
                    print("[OpenAI Image] Converting JPEG to PNG for DALL-E 2 compatibility...")
                png_data = _convert_to_png(image_source, verbose=verbose)
                image_file = io.BytesIO(png_data)
                image_file.name = "image.png"
            elif isinstance(image_source, str):
                image_file = open(image_source, "rb")
            else:
                image_file = io.BytesIO(image_source)
                image_file.name = "image.png"

            # Build API parameters based on model type
            if is_gpt_image:
                # GPT Image models
                # Note: The OpenAI SDK's images.edit() has limited parameter support.
                # Parameters like output_format, quality, and input_fidelity are NOT
                # supported in the current SDK version, even though the API docs mention them.
                # See: https://community.openai.com/t/error-using-gpt-image-1-api-with-quality-parameter/1239987
                params = {
                    "model": model_name,
                    "image": image_file,
                    "prompt": edit_prompt,
                    "size": size,
                    "n": 1,  # GPT Image only supports 1
                }
                if verbose and (quality or input_fidelity):
                    print(f"[OpenAI Image] Note: quality and input_fidelity parameters are not supported by the SDK for images.edit()")
            else:
                # DALL-E 2
                params = {
                    "model": model_name,
                    "image": image_file,
                    "prompt": edit_prompt,
                    "size": size,
                    "n": n,
                    "response_format": response_format,
                }

                # Prepare mask file if provided (DALL-E 2 only)
                # Also convert mask to PNG if needed
                mask_file = None
                if mask_source:
                    if _is_jpeg(mask_source):
                        png_mask = _convert_to_png(mask_source, verbose=verbose)
                        mask_file = io.BytesIO(png_mask)
                        mask_file.name = "mask.png"
                    elif isinstance(mask_source, str):
                        mask_file = open(mask_source, "rb")
                    else:
                        mask_file = io.BytesIO(mask_source)
                        mask_file.name = "mask.png"
                    params["mask"] = mask_file

            # Call edit API
            response = await async_openai_client.images.edit(**params)

            # Close files
            if isinstance(image_source, str) and not _is_jpeg(image_source):
                image_file.close()
            if not is_gpt_image and mask_source and isinstance(mask_source, str) and not _is_jpeg(mask_source):
                mask_file.close()

            # Extract image data from response
            image_response = response.data[0]

            # Handle different response formats
            if is_gpt_image or response_format == "b64_json":
                # GPT Image always returns base64
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
                    revised_prompt=getattr(image_response, 'revised_prompt', None),
                    size=size,
                    quality=quality if is_gpt_image else None,
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
