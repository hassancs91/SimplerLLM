from dotenv import load_dotenv
import os
import time
import requests
from .image_response_models import ImageGenerationResponse

# Load environment variables
load_dotenv(override=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

# BytePlus Seedream API endpoint
SEEDREAM_API_ENDPOINT = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"


def generate_image(
    prompt,
    model_name="seedream-4-5-251128",
    size="1024x1024",
    watermark=False,
    response_format="url",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    image=None,
    quality=None,
    seed=None,
    negative_prompt=None,
    n=1,
):
    """
    Generate image from text prompt using BytePlus Seedream API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (default: seedream-4-5-251128)
        size: Image size as dimensions (e.g., "1024x1024", "1536x1024", "2048x2048")
              Supported sizes: 1024x1024, 1536x1536, 2048x2048 (square)
                              1024x1536, 1024x2048 (portrait)
                              1536x1024, 2048x1024 (landscape)
        watermark: If True, adds "AI generated" watermark to bottom-right corner
        response_format: "url" for image URL, "b64_json" for base64 encoded data
        output_path: Optional file path to save image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: BytePlus ARK API key (uses ARK_API_KEY env var if not provided)
        verbose: If True, prints progress information
        image: Optional URL of reference image for image-to-image generation
        quality: Image quality - "standard" or "high" (default: None = API default)
        seed: Random seed for reproducibility (integer)
        negative_prompt: Elements to exclude from the generated image
        n: Number of images to generate (default: 1)

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        If response_format="url": image URL string
        Otherwise: image bytes
    """
    start_time = time.time() if full_response else None

    # Get API key from parameter or environment
    api_key = api_key or os.getenv("ARK_API_KEY", "")
    if not api_key:
        raise ValueError("ARK_API_KEY not found in environment variables or parameters")

    if verbose:
        print(f"[Seedream] Generating image with model={model_name}, size={size}")
        if image:
            print(f"[Seedream] Using reference image: {image}")

    for attempt in range(MAX_RETRIES):
        try:
            # Build request headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            # Build request data
            data = {
                "model": model_name,
                "prompt": prompt,
                "size": size,
                "watermark": watermark,
                "n": n,
            }

            # Add optional parameters if provided
            if quality:
                data["quality"] = quality
            if seed is not None:
                data["seed"] = seed
            if negative_prompt:
                data["negative_prompt"] = negative_prompt

            # Add reference image for image-to-image generation
            if image:
                data["image"] = image

            # Make API request
            response = requests.post(
                SEEDREAM_API_ENDPOINT,
                headers=headers,
                json=data,
                timeout=120
            )

            # Check for errors
            if response.status_code != 200:
                error_msg = f"Seedream API error (status {response.status_code})"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data}"
                except:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)

            # Parse response
            response_data = response.json()

            if verbose:
                print(f"[Seedream] Response received: {response_data.get('model', 'unknown model')}")

            # Extract image data from response
            if "data" not in response_data or len(response_data["data"]) == 0:
                raise Exception("No image data found in Seedream response")

            image_info = response_data["data"][0]
            image_url = image_info.get("url")
            image_size_str = image_info.get("size", size)

            if not image_url:
                raise Exception("No image URL found in Seedream response")

            # Handle output based on format and output_path
            if output_path or response_format == "b64_json":
                # Download the image
                img_response = requests.get(image_url, timeout=60)
                img_response.raise_for_status()
                image_bytes = img_response.content

                if output_path:
                    # Save to file
                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    image_data = output_path
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[Seedream] Image saved to: {output_path} ({file_size} bytes)")
                else:
                    image_data = image_bytes
                    file_size = len(image_bytes)
                    if verbose:
                        print(f"[Seedream] Image downloaded ({file_size} bytes)")
            else:
                # Return URL
                image_data = image_url
                file_size = None
                if verbose:
                    print(f"[Seedream] Image URL received: {image_url[:50]}...")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time

                return ImageGenerationResponse(
                    image_data=image_data,
                    model=model_name,
                    prompt=prompt,
                    revised_prompt=None,
                    size=image_size_str,
                    quality=quality,
                    style=None,
                    process_time=process_time,
                    provider="SEEDREAM",
                    file_size=file_size,
                    output_path=output_path,
                    llm_provider_response=response_data,
                )

            return image_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[Seedream] Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate image after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_image_async(
    prompt,
    model_name="seedream-4-5-251128",
    size="1024x1024",
    watermark=False,
    response_format="url",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    image=None,
    quality=None,
    seed=None,
    negative_prompt=None,
    n=1,
):
    """
    Async version: Generate image from text prompt using BytePlus Seedream API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (default: seedream-4-5-251128)
        size: Image size as dimensions (e.g., "1024x1024", "1536x1024", "2048x2048")
        watermark: If True, adds "AI generated" watermark to bottom-right corner
        response_format: "url" for image URL, "b64_json" for base64 encoded data
        output_path: Optional file path to save image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: BytePlus ARK API key (uses ARK_API_KEY env var if not provided)
        verbose: If True, prints progress information
        image: Optional URL of reference image for image-to-image generation
        quality: Image quality - "standard" or "high" (default: None = API default)
        seed: Random seed for reproducibility (integer)
        negative_prompt: Elements to exclude from the generated image
        n: Number of images to generate (default: 1)

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        If response_format="url": image URL string
        Otherwise: image bytes
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: generate_image(
            prompt=prompt,
            model_name=model_name,
            size=size,
            watermark=watermark,
            response_format=response_format,
            output_path=output_path,
            full_response=full_response,
            api_key=api_key,
            verbose=verbose,
            image=image,
            quality=quality,
            seed=seed,
            negative_prompt=negative_prompt,
            n=n,
        )
    )


def edit_image(
    image_source,
    edit_prompt,
    model_name="seedream-4-5-251128",
    size="1024x1024",
    watermark=False,
    response_format="url",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    quality=None,
    seed=None,
    negative_prompt=None,
):
    """
    Edit an existing image using text instructions with BytePlus Seedream API.

    Args:
        image_source: URL of the source image to edit
        edit_prompt: Text instructions for how to edit the image
        model_name: Model to use (default: seedream-4-5-251128)
        size: Image size as dimensions (e.g., "1024x1024", "1536x1024", "2048x2048")
        watermark: If True, adds "AI generated" watermark to bottom-right corner
        response_format: "url" for image URL, "b64_json" for base64 encoded data
        output_path: Optional file path to save edited image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: BytePlus ARK API key (uses ARK_API_KEY env var if not provided)
        verbose: If True, prints progress information
        quality: Image quality - "standard" or "high" (default: None = API default)
        seed: Random seed for reproducibility (integer)
        negative_prompt: Elements to exclude from the generated image

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        If response_format="url": image URL string
        Otherwise: edited image bytes

    Example:
        >>> edited_url = edit_image(
        ...     image_source="https://example.com/original.jpg",
        ...     edit_prompt="Change the background to a sunset sky",
        ...     size="1024x1024"
        ... )
    """
    if verbose:
        print(f"[Seedream] Editing image with prompt: {edit_prompt[:50]}...")

    # Use generate_image with the image parameter for image-to-image editing
    return generate_image(
        prompt=edit_prompt,
        model_name=model_name,
        size=size,
        watermark=watermark,
        response_format=response_format,
        output_path=output_path,
        full_response=full_response,
        api_key=api_key,
        verbose=verbose,
        image=image_source,
        quality=quality,
        seed=seed,
        negative_prompt=negative_prompt,
        n=1,  # Edit always produces single image
    )


async def edit_image_async(
    image_source,
    edit_prompt,
    model_name="seedream-4-5-251128",
    size="1024x1024",
    watermark=False,
    response_format="url",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
    quality=None,
    seed=None,
    negative_prompt=None,
):
    """
    Async version: Edit an existing image using text instructions with BytePlus Seedream API.

    Args:
        image_source: URL of the source image to edit
        edit_prompt: Text instructions for how to edit the image
        model_name: Model to use (default: seedream-4-5-251128)
        size: Image size as dimensions (e.g., "1024x1024", "1536x1024", "2048x2048")
        watermark: If True, adds "AI generated" watermark to bottom-right corner
        response_format: "url" for image URL, "b64_json" for base64 encoded data
        output_path: Optional file path to save edited image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: BytePlus ARK API key (uses ARK_API_KEY env var if not provided)
        verbose: If True, prints progress information
        quality: Image quality - "standard" or "high" (default: None = API default)
        seed: Random seed for reproducibility (integer)
        negative_prompt: Elements to exclude from the generated image

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        If response_format="url": image URL string
        Otherwise: edited image bytes
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: edit_image(
            image_source=image_source,
            edit_prompt=edit_prompt,
            model_name=model_name,
            size=size,
            watermark=watermark,
            response_format=response_format,
            output_path=output_path,
            full_response=full_response,
            api_key=api_key,
            verbose=verbose,
            quality=quality,
            seed=seed,
            negative_prompt=negative_prompt,
        )
    )
