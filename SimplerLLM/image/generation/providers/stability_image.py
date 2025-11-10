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

# Stability AI API base URL
STABILITY_API_BASE = "https://api.stability.ai"


def _get_endpoint(model_name):
    """
    Determine the correct API endpoint based on the model name.

    Args:
        model_name: The model to use

    Returns:
        str: The full API endpoint URL
    """
    model_lower = model_name.lower()

    if "ultra" in model_lower:
        return f"{STABILITY_API_BASE}/v2beta/stable-image/generate/ultra"
    elif "core" in model_lower:
        return f"{STABILITY_API_BASE}/v2beta/stable-image/generate/core"
    elif "sd3" in model_lower or "stable-diffusion" in model_lower:
        return f"{STABILITY_API_BASE}/v2beta/stable-image/generate/sd3"
    else:
        # Default to core
        return f"{STABILITY_API_BASE}/v2beta/stable-image/generate/core"


def generate_image(
    prompt,
    model_name="stable-image-core",
    aspect_ratio="1:1",
    negative_prompt=None,
    style_preset=None,
    seed=0,
    cfg_scale=None,
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Generate image from text prompt using Stability AI API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (stable-image-ultra, stable-image-core,
                   sd3.5-large, sd3.5-large-turbo, sd3.5-medium, sd3.5-flash)
        aspect_ratio: Aspect ratio (1:1, 16:9, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21)
        negative_prompt: What you do NOT want to see in the output
        style_preset: Style to guide the generation (3d-model, analog-film, anime,
                     cinematic, comic-book, digital-art, enhance, fantasy-art,
                     isometric, line-art, low-poly, modeling-compound, neon-punk,
                     origami, photographic, pixel-art, tile-texture)
        seed: Randomness seed (0 for random)
        cfg_scale: Prompt adherence (1-10, higher = stricter)
        output_format: Image format (png, jpeg, webp)
        output_path: Optional file path to save image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: Stability API key (uses env var if not provided)
        verbose: If True, prints progress information

    Returns:
        If full_response=True: ImageGenerationResponse object
        If output_path provided: file path string
        Otherwise: image bytes
    """
    start_time = time.time() if full_response else None

    # Get API key from parameter or environment
    api_key = api_key or os.getenv("STABILITY_API_KEY", "")
    if not api_key:
        raise ValueError("STABILITY_API_KEY not found in environment variables or parameters")

    # Determine endpoint
    endpoint = _get_endpoint(model_name)

    if verbose:
        print(f"[Stability AI] Generating image with model={model_name}, aspect_ratio={aspect_ratio}")
        print(f"[Stability AI] Using endpoint: {endpoint}")

    for attempt in range(MAX_RETRIES):
        try:
            # Build request headers
            headers = {
                "authorization": f"Bearer {api_key}",
                "accept": "image/*"  # Get image bytes directly
            }

            # Build request data
            data = {
                "prompt": prompt,
                "output_format": output_format,
            }

            # Add aspect_ratio (for text-to-image)
            if aspect_ratio:
                data["aspect_ratio"] = aspect_ratio

            # Add optional parameters
            if negative_prompt:
                data["negative_prompt"] = negative_prompt

            if style_preset:
                data["style_preset"] = style_preset

            if seed and seed != 0:
                data["seed"] = seed

            if cfg_scale is not None:
                data["cfg_scale"] = cfg_scale

            # For SD3.5 models, add model parameter
            if "sd3" in model_name.lower():
                data["model"] = model_name

            # Empty files dict required for multipart/form-data
            files = {"none": ''}

            # Make API request
            response = requests.post(
                endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=120
            )

            # Check for errors
            if response.status_code != 200:
                error_msg = f"Stability AI API error (status {response.status_code})"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data}"
                except:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)

            # Get image bytes
            image_bytes = response.content

            # Handle output - save to file or return bytes
            if output_path:
                # Save to file
                os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                with open(output_path, 'wb') as f:
                    f.write(image_bytes)
                image_data = output_path
                file_size = len(image_bytes)
                if verbose:
                    print(f"[Stability AI] Image saved to: {output_path} ({file_size} bytes)")
            else:
                image_data = image_bytes
                file_size = len(image_bytes)
                if verbose:
                    print(f"[Stability AI] Image generated in memory ({file_size} bytes)")

            # Return full response with metadata if requested
            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return ImageGenerationResponse(
                    image_data=image_data,
                    model=model_name,
                    prompt=prompt,
                    revised_prompt=None,  # Stability doesn't provide revised prompts
                    size=aspect_ratio,  # Store aspect ratio in size field
                    quality=None,  # Stability doesn't have quality parameter like DALL-E
                    style=style_preset,  # Store style_preset in style field
                    process_time=process_time,
                    provider="STABILITY_AI",
                    file_size=file_size,
                    output_path=output_path,
                    llm_provider_response=response,
                )

            return image_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"[Stability AI] Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed to generate image after {MAX_RETRIES} attempts due to: {e}"
                raise Exception(error_msg)


async def generate_image_async(
    prompt,
    model_name="stable-image-core",
    aspect_ratio="1:1",
    negative_prompt=None,
    style_preset=None,
    seed=0,
    cfg_scale=None,
    output_format="png",
    output_path=None,
    full_response=False,
    api_key=None,
    verbose=False,
):
    """
    Async version: Generate image from text prompt using Stability AI API.

    Args:
        prompt: Text description of the desired image
        model_name: Model to use (stable-image-ultra, stable-image-core,
                   sd3.5-large, sd3.5-large-turbo, sd3.5-medium, sd3.5-flash)
        aspect_ratio: Aspect ratio (1:1, 16:9, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21)
        negative_prompt: What you do NOT want to see in the output
        style_preset: Style to guide the generation (3d-model, analog-film, anime,
                     cinematic, comic-book, digital-art, enhance, fantasy-art,
                     isometric, line-art, low-poly, modeling-compound, neon-punk,
                     origami, photographic, pixel-art, tile-texture)
        seed: Randomness seed (0 for random)
        cfg_scale: Prompt adherence (1-10, higher = stricter)
        output_format: Image format (png, jpeg, webp)
        output_path: Optional file path to save image
        full_response: If True, returns ImageGenerationResponse with metadata
        api_key: Stability API key (uses env var if not provided)
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
            negative_prompt=negative_prompt,
            style_preset=style_preset,
            seed=seed,
            cfg_scale=cfg_scale,
            output_format=output_format,
            output_path=output_path,
            full_response=full_response,
            api_key=api_key,
            verbose=verbose,
        )
    )
