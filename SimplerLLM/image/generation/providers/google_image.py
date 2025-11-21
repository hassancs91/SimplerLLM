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


def generate_image(
    prompt,
    model_name="gemini-2.5-flash-image-preview",
    aspect_ratio="1:1",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Generate image from text prompt using Google Gemini API SDK.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (default: gemini-2.5-flash-image-preview)
        aspect_ratio: Aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)
        output_format: Image format (png, jpeg, webp) - for metadata only
        output_path: Optional file path to save image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: Google API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: image bytes
    """
    start_time = time.time() if full_response else None

    # Get API key from parameter or environment
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables or parameters")

    # Initialize client with API key
    client = genai.Client(api_key=api_key)

    if verbose:
        print(f"[Google Gemini] Generating image with model={model_name}, aspect_ratio={aspect_ratio}")

    # Always print which model is being used for verification
    print(f"[Google Gemini API] Using model: {model_name}")

    retry_delay = RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            # Build content parts
            parts = [types.Part.from_text(text=prompt)]

            # Create contents with proper structure
            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                ),
            ]

            # Configure generation with both IMAGE and TEXT modalities
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=1.0,
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
                    quality=None,  # Gemini doesn't have quality parameter
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
    model_name="gemini-2.5-flash-image-preview",
    aspect_ratio="1:1",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Async version: Generate image from text prompt using Google Gemini API SDK.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (default: gemini-2.5-flash-image-preview)
        aspect_ratio: Aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)
        output_format: Image format (png, jpeg, webp) - for metadata only
        output_path: Optional file path to save image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: Google API key (uses env var if not provided)
        verbose: If True, prints progress information

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
            output_format=output_format,
            output_path=output_path,
            full_response=full_response,
            api_key=api_key,
            verbose=verbose,
        )
    )
