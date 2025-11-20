from dotenv import load_dotenv
import os
import time
import requests
import base64
from .image_response_models import ImageGenerationResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

# Google Gemini API base URL
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def generate_image(
    prompt,
    model_name="gemini-2.5-flash-image",
    aspect_ratio="1:1",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Generate image from text prompt using Google Gemini API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (default: gemini-2.5-flash-image)
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

    # Build endpoint URL
    endpoint = f"{GEMINI_API_BASE}/models/{model_name}:generateContent"

    if verbose:
        print(f"[Google Gemini] Generating image with model={model_name}, aspect_ratio={aspect_ratio}")
        print(f"[Google Gemini] Using endpoint: {endpoint}")

    for attempt in range(MAX_RETRIES):
        try:
            # Build request headers
            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            }

            # Build request payload
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "responseModalities": ["Image"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio
                    }
                }
            }

            # Make API request
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )

            # Check for errors
            if response.status_code != 200:
                error_msg = f"Google Gemini API error (status {response.status_code})"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data}"
                except:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)

            # Parse response
            response_json = response.json()
            candidates = response_json.get("candidates", [])

            if not candidates:
                raise Exception("No candidates in response")

            parts = candidates[0].get("content", {}).get("parts", [])

            # Extract image and text from parts
            image_data_base64 = None
            text_description = None

            for part in parts:
                if "inline_data" in part:
                    # Found image data
                    inline_data = part["inline_data"]
                    image_data_base64 = inline_data.get("data")
                    # mime_type = inline_data.get("mime_type")  # e.g., "image/png"
                elif "text" in part:
                    # Found text description
                    text_description = part["text"]

            if not image_data_base64:
                raise Exception("No image data found in response")

            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data_base64)

            if verbose:
                print(f"[Google Gemini] Image decoded: {len(image_bytes)} bytes")
                if text_description:
                    desc_preview = text_description[:100] + "..." if len(text_description) > 100 else text_description
                    print(f"[Google Gemini] Text description: {desc_preview}")

            # Handle output - save to file or return bytes
            if output_path:
                # Save to file
                os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                with open(output_path, 'wb') as f:
                    f.write(image_bytes)
                image_data = output_path
                file_size = len(image_bytes)
                if verbose:
                    print(f"[Google Gemini] Image saved to: {output_path} ({file_size} bytes)")
            else:
                image_data = image_bytes
                file_size = len(image_bytes)
                if verbose:
                    print(f"[Google Gemini] Image generated in memory ({file_size} bytes)")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time

                # Create response with Gemini-specific fields
                response_obj = ImageGenerationResponse(
                    image_data=image_data,
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
                    llm_provider_response=response_json,
                )

                return response_obj

            return image_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[Google Gemini] Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate image after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_image_async(
    prompt,
    model_name="gemini-2.5-flash-image",
    aspect_ratio="1:1",
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Async version: Generate image from text prompt using Google Gemini API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (default: gemini-2.5-flash-image)
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

    # Note: For a truly async implementation, you would use aiohttp instead of requests
    # For now, we'll run the sync version in an executor
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
