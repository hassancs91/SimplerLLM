from dotenv import load_dotenv
import os
import time
import mimetypes
import google.genai as genai
from google.genai import types
from .image_response_models import ImageGenerationResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


def _get_image_mime_type(image_source):
    """
    Determine MIME type from image source.

    Args:
        image_source: File path (str) or bytes

    Returns:
        MIME type string (e.g., 'image/jpeg', 'image/png')
    """
    if isinstance(image_source, str):
        # Guess from file extension
        mime_type, _ = mimetypes.guess_type(image_source)
        if mime_type and mime_type.startswith('image/'):
            return mime_type

    # Default to JPEG if unknown
    return 'image/jpeg'


def _load_image_data(image_source):
    """
    Load image data from various source types.

    Args:
        image_source: Can be:
            - str: File path to image
            - bytes: Raw image data
            - dict: {'data': bytes, 'mime_type': str}

    Returns:
        tuple: (image_data_bytes, mime_type)

    Raises:
        ValueError: If image source is invalid
        FileNotFoundError: If file path doesn't exist
    """
    if isinstance(image_source, dict):
        # Dict with data and mime_type
        return image_source.get('data'), image_source.get('mime_type', 'image/jpeg')

    elif isinstance(image_source, bytes):
        # Raw bytes - try to guess mime type
        return image_source, 'image/jpeg'

    elif isinstance(image_source, str):
        # File path
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")

        with open(image_source, 'rb') as f:
            image_data = f.read()

        mime_type = _get_image_mime_type(image_source)
        return image_data, mime_type

    else:
        raise ValueError(f"Invalid image source type: {type(image_source)}. Must be str (path), bytes, or dict")


def generate_image(
    prompt,
    model_name="gemini-2.5-flash-image",
    aspect_ratio="1:1",
    resolution="1K",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    reference_images=None,
):
    """
    Generate image from text prompt using Google Gemini API SDK.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (default: gemini-2.5-flash-image)
        aspect_ratio: Aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)
        resolution: Image resolution - "1K", "2K", or "4K" (default: "1K")
        output_format: Image format (png, jpeg, webp) - for metadata only
        output_path: Optional file path to save image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: Google API key (uses env var if not provided)
        verbose: If True, prints progress information
        reference_images: Optional list of reference images for character consistency.
                         Each item can be:
                         - str: File path to image
                         - bytes: Raw image data
                         - dict: {'data': bytes, 'mime_type': str}

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: image bytes

    Example with reference images:
        >>> img_gen = ImageGenerator.create(provider=ImageProvider.GOOGLE_GEMINI)
        >>> image_bytes = img_gen.generate_image(
        ...     prompt="A portrait of the same character in a different pose",
        ...     reference_images=["character_ref.jpg"],
        ...     size=ImageSize.SQUARE
        ... )
    """
    start_time = time.time() if full_response else None

    # Get API key from parameter or environment
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables or parameters")

    # Initialize client with API key
    client = genai.Client(api_key=api_key)

    if verbose:
        print(f"[Google Gemini] Generating image with model={model_name}, aspect_ratio={aspect_ratio}, resolution={resolution}")

    # Always print which model is being used for verification
    print(f"[Google Gemini API] Using model: {model_name}")

    retry_delay = RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            # Build content parts - start with reference images, then text prompt
            parts = []

            # Add reference images if provided
            if reference_images:
                if verbose:
                    print(f"[Google Gemini] Adding {len(reference_images)} reference image(s)")

                for idx, ref_image in enumerate(reference_images):
                    try:
                        image_data, mime_type = _load_image_data(ref_image)
                        parts.append(
                            types.Part.from_bytes(
                                data=image_data,
                                mime_type=mime_type
                            )
                        )
                        if verbose:
                            print(f"[Google Gemini] Added reference image {idx + 1}: {mime_type}, {len(image_data)} bytes")
                    except Exception as e:
                        if verbose:
                            print(f"[Google Gemini] Warning: Could not load reference image {idx + 1}: {e}")
                        # Continue without this reference image
                        continue

            # Add text prompt after reference images
            parts.append(types.Part.from_text(text=prompt))

            # Create contents with proper structure
            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                ),
            ]

            # Configure generation with both IMAGE and TEXT modalities
            # image_size only supported by Gemini 3+ models
            image_config_params = {"aspect_ratio": aspect_ratio}
            if "gemini-3" in model_name.lower():
                image_config_params["image_size"] = resolution

            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=1.0,
                image_config=types.ImageConfig(**image_config_params),
            )

            image_data = None
            mime_type = None
            text_output = []

            # Use streaming
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            ):
                # Check if chunk has valid content
                if (
                    chunk.candidates is None
                    or len(chunk.candidates) == 0
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue

                # Process each part in the chunk
                for part in chunk.candidates[0].content.parts:
                    # Handle image data
                    if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                        if verbose:
                            print(f"[Google Gemini] Found image data, MIME type: {part.inline_data.mime_type}")
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type

                    # Handle text data
                    elif hasattr(part, 'text') and part.text:
                        text_output.append(part.text)

            if not image_data:
                raise Exception("No image data found in response")

            # Combine text output
            text_description = "".join(text_output) if text_output else None

            if verbose:
                print(f"[Google Gemini] Image decoded: {len(image_data)} bytes")
                if text_description:
                    desc_preview = text_description[:100] + "..." if len(text_description) > 100 else text_description
                    print(f"[Google Gemini] Text description: {desc_preview}")

            # Handle output - save to file or return bytes
            if output_path:
                # Save to file
                os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                result_data = output_path
                file_size = len(image_data)
                if verbose:
                    print(f"[Google Gemini] Image saved to: {output_path} ({file_size} bytes)")
            else:
                result_data = image_data
                file_size = len(image_data)
                if verbose:
                    print(f"[Google Gemini] Image generated in memory ({file_size} bytes)")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time

                # Determine file extension from mime type
                file_extension = mimetypes.guess_extension(mime_type) if mime_type else None
                if not file_extension:
                    file_extension = ".png"

                # Create response with Gemini-specific fields
                response_obj = ImageGenerationResponse(
                    image_data=result_data,
                    model=model_name,
                    prompt=prompt,
                    revised_prompt=text_description,  # Store text description as revised_prompt
                    size=aspect_ratio,  # Store aspect ratio in size field
                    quality=resolution,  # Store resolution (1K, 2K, 4K)
                    style=None,  # Gemini doesn't have style presets like Stability
                    process_time=process_time,
                    provider="GOOGLE_GEMINI",
                    file_size=file_size,
                    output_path=output_path,
                    llm_provider_response=None,  # SDK doesn't expose raw JSON
                )

                return response_obj

            return result_data

        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"[Google Gemini] Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_msg}")

            # Check if it's a 500 internal error
            if "500" in error_msg or "INTERNAL" in error_msg:
                if attempt < MAX_RETRIES - 1:
                    if verbose:
                        print(f"[Google Gemini] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue

            # If not a 500 error or last attempt, raise exception
            if attempt == MAX_RETRIES - 1:
                raise Exception(f"Failed to generate image after {MAX_RETRIES} attempts due to: {error_msg}")


async def generate_image_async(
    prompt,
    model_name="gemini-2.5-flash-image",
    aspect_ratio="1:1",
    resolution="1K",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    reference_images=None,
):
    """
    Async version: Generate image from text prompt using Google Gemini API SDK.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (default: gemini-2.5-flash-image)
        aspect_ratio: Aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)
        resolution: Image resolution - "1K", "2K", or "4K" (default: "1K")
        output_format: Image format (png, jpeg, webp) - for metadata only
        output_path: Optional file path to save image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: Google API key (uses env var if not provided)
        verbose: If True, prints progress information
        reference_images: Optional list of reference images for character consistency.
                         Each item can be:
                         - str: File path to image
                         - bytes: Raw image data
                         - dict: {'data': bytes, 'mime_type': str}

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: image bytes
    """
    import asyncio

    # Run the sync version in an executor since the SDK doesn't provide async methods
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: generate_image(
            prompt=prompt,
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            output_path=output_path,
            full_response=full_response,
            api_key=api_key,
            verbose=verbose,
            reference_images=reference_images,
        )
    )


def edit_image(
    image_source,
    edit_prompt,
    model_name="gemini-2.5-flash-image",
    aspect_ratio="1:1",
    resolution="1K",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Edit an existing image using text instructions with Google Gemini API SDK.

    Args:
        image_source: Source image to edit. Can be:
                     - str: File path to image
                     - bytes: Raw image data
                     - dict: {'data': bytes, 'mime_type': str}
        edit_prompt: Text instructions for how to edit the image
        model_name: Model to use (default: gemini-2.5-flash-image)
        aspect_ratio: Aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)
        resolution: Image resolution - "1K", "2K", or "4K" (default: "1K")
        output_format: Image format (png, jpeg, webp) - for metadata only
        output_path: Optional file path to save edited image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: Google API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: edited image bytes

    Example:
        >>> img_gen = ImageGenerator.create(provider=ImageProvider.GOOGLE_GEMINI)
        >>> edited_bytes = img_gen.edit_image(
        ...     image_source="original.jpg",
        ...     edit_prompt="Change the background to a sunset sky",
        ...     size=ImageSize.SQUARE
        ... )
    """
    start_time = time.time() if full_response else None

    # Get API key from parameter or environment
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables or parameters")

    # Initialize client with API key
    client = genai.Client(api_key=api_key)

    if verbose:
        print(f"[Google Gemini] Editing image with model={model_name}, aspect_ratio={aspect_ratio}, resolution={resolution}")
        print(f"[Google Gemini] Edit prompt: {edit_prompt}")

    # Always print which model is being used for verification
    print(f"[Google Gemini API] Using model: {model_name}")

    retry_delay = RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            # Load the image to edit
            image_data, mime_type = _load_image_data(image_source)

            if verbose:
                print(f"[Google Gemini] Loaded image to edit: {mime_type}, {len(image_data)} bytes")

            # Build content parts - image first, then edit instructions
            parts = [
                types.Part.from_bytes(
                    data=image_data,
                    mime_type=mime_type
                ),
                types.Part.from_text(text=edit_prompt)
            ]

            # Create contents with proper structure
            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                ),
            ]

            # Configure generation with lower temperature for consistency
            # image_size only supported by Gemini 3+ models
            image_config_params = {"aspect_ratio": aspect_ratio}
            if "gemini-3" in model_name.lower():
                image_config_params["image_size"] = resolution

            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.8,  # Lower temperature for editing to maintain consistency
                image_config=types.ImageConfig(**image_config_params),
            )

            edited_image_data = None
            edited_mime_type = None
            text_output = []

            # Use streaming
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            ):
                # Check if chunk has valid content
                if (
                    chunk.candidates is None
                    or len(chunk.candidates) == 0
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue

                # Process each part in the chunk
                for part in chunk.candidates[0].content.parts:
                    # Handle image data
                    if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                        if verbose:
                            print(f"[Google Gemini] Found edited image data, MIME type: {part.inline_data.mime_type}")
                        edited_image_data = part.inline_data.data
                        edited_mime_type = part.inline_data.mime_type

                    # Handle text data
                    elif hasattr(part, 'text') and part.text:
                        text_output.append(part.text)

            if not edited_image_data:
                raise Exception("No edited image data found in response")

            # Combine text output
            text_description = "".join(text_output) if text_output else None

            if verbose:
                print(f"[Google Gemini] Edited image decoded: {len(edited_image_data)} bytes")
                if text_description:
                    desc_preview = text_description[:100] + "..." if len(text_description) > 100 else text_description
                    print(f"[Google Gemini] Text description: {desc_preview}")

            # Handle output - save to file or return bytes
            if output_path:
                # Save to file
                os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                with open(output_path, 'wb') as f:
                    f.write(edited_image_data)
                result_data = output_path
                file_size = len(edited_image_data)
                if verbose:
                    print(f"[Google Gemini] Edited image saved to: {output_path} ({file_size} bytes)")
            else:
                result_data = edited_image_data
                file_size = len(edited_image_data)
                if verbose:
                    print(f"[Google Gemini] Edited image generated in memory ({file_size} bytes)")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time

                # Determine file extension from mime type
                file_extension = mimetypes.guess_extension(edited_mime_type) if edited_mime_type else None
                if not file_extension:
                    file_extension = ".png"

                # Create response with Gemini-specific fields
                response_obj = ImageGenerationResponse(
                    image_data=result_data,
                    model=model_name,
                    prompt=edit_prompt,
                    revised_prompt=text_description,  # Store text description as revised_prompt
                    size=aspect_ratio,  # Store aspect ratio in size field
                    quality=resolution,  # Store resolution (1K, 2K, 4K)
                    style=None,  # Gemini doesn't have style presets
                    process_time=process_time,
                    provider="GOOGLE_GEMINI",
                    file_size=file_size,
                    output_path=output_path,
                    llm_provider_response=None,  # SDK doesn't expose raw JSON
                )

                return response_obj

            return result_data

        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"[Google Gemini] Edit attempt {attempt + 1}/{MAX_RETRIES} failed: {error_msg}")

            # Check if it's a 500 internal error
            if "500" in error_msg or "INTERNAL" in error_msg:
                if attempt < MAX_RETRIES - 1:
                    if verbose:
                        print(f"[Google Gemini] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue

            # If not a 500 error or last attempt, raise exception
            if attempt == MAX_RETRIES - 1:
                raise Exception(f"Failed to edit image after {MAX_RETRIES} attempts due to: {error_msg}")


async def edit_image_async(
    image_source,
    edit_prompt,
    model_name="gemini-2.5-flash-image",
    aspect_ratio="1:1",
    resolution="1K",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Async version: Edit an existing image using text instructions with Google Gemini API SDK.

    Args:
        image_source: Source image to edit. Can be:
                     - str: File path to image
                     - bytes: Raw image data
                     - dict: {'data': bytes, 'mime_type': str}
        edit_prompt: Text instructions for how to edit the image
        model_name: Model to use (default: gemini-2.5-flash-image)
        aspect_ratio: Aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)
        resolution: Image resolution - "1K", "2K", or "4K" (default: "1K")
        output_format: Image format (png, jpeg, webp) - for metadata only
        output_path: Optional file path to save edited image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: Google API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: edited image bytes
    """
    import asyncio

    # Run the sync version in an executor since the SDK doesn't provide async methods
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: edit_image(
            image_source=image_source,
            edit_prompt=edit_prompt,
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            output_path=output_path,
            full_response=full_response,
            api_key=api_key,
            verbose=verbose,
        )
    )
