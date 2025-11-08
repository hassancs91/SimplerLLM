"""
Image helper utilities for vision-enabled LLM requests.

This module provides utilities for encoding, validating, and preparing
images for use with vision-capable language models.
"""

import base64
import os
from typing import Dict, Any
from urllib.parse import urlparse


def is_url(source: str) -> bool:
    """
    Determine if a source string is a URL or a file path.

    Args:
        source: String that could be either a URL or file path

    Returns:
        bool: True if source is a URL, False if it's a file path
    """
    try:
        result = urlparse(source)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string of the image

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the file
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise IOError(f"Error reading image file {image_path}: {str(e)}")


def get_image_mime_type(image_path: str) -> str:
    """
    Detect the MIME type of an image based on its extension.

    Args:
        image_path: Path to the image file

    Returns:
        str: MIME type (e.g., 'image/jpeg', 'image/png')
    """
    extension = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(extension, 'image/jpeg')  # Default to jpeg


def validate_image_source(source: str) -> bool:
    """
    Validate that an image source is accessible.

    Args:
        source: URL or file path to validate

    Returns:
        bool: True if source is valid and accessible
    """
    if is_url(source):
        # For URLs, basic validation that it's properly formatted
        return True  # We'll let the API handle URL validation
    else:
        # For file paths, check if file exists
        return os.path.exists(source)


def prepare_image_content(source: str, detail: str = "auto") -> Dict[str, Any]:
    """
    Prepare image content in OpenAI's vision API format.

    Args:
        source: URL or file path to the image
        detail: Level of detail for image processing ("low", "high", "auto")

    Returns:
        dict: Image content formatted for OpenAI API

    Raises:
        FileNotFoundError: If local file doesn't exist
        ValueError: If detail parameter is invalid
    """
    if detail not in ["low", "high", "auto"]:
        raise ValueError(f"Invalid detail level: {detail}. Must be 'low', 'high', or 'auto'")

    if is_url(source):
        # URL-based image
        return {
            "type": "image_url",
            "image_url": {
                "url": source,
                "detail": detail
            }
        }
    else:
        # Local file - encode to base64
        base64_image = encode_image_to_base64(source)
        mime_type = get_image_mime_type(source)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
                "detail": detail
            }
        }


def prepare_vision_content(text: str, images: list, detail: str = "auto") -> list:
    """
    Prepare multi-part content for vision requests combining text and images.
    (OpenAI format)

    Args:
        text: Text prompt/message
        images: List of image sources (URLs or file paths)
        detail: Level of detail for image processing

    Returns:
        list: Multi-part content array for OpenAI vision API
    """
    content = [{"type": "text", "text": text}]

    for image_source in images:
        image_content = prepare_image_content(image_source, detail)
        content.append(image_content)

    return content


# ============================================================================
# Anthropic-specific vision helpers
# ============================================================================

def validate_image_size(image_path: str) -> tuple:
    """
    Validate image file size and optionally dimensions for Anthropic API.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (is_valid, warning_message)
    """
    warnings = []

    # Check file size (5MB limit for Anthropic)
    if os.path.exists(image_path):
        file_size = os.path.getsize(image_path)
        file_size_mb = file_size / (1024 * 1024)

        if file_size_mb > 5:
            warnings.append(f"Image file size ({file_size_mb:.2f}MB) exceeds Anthropic's 5MB limit")

    # Check dimensions if Pillow is available
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
            if width > 8000 or height > 8000:
                warnings.append(f"Image dimensions ({width}×{height}) exceed Anthropic's 8000×8000 limit")
    except ImportError:
        # Pillow not available, skip dimension check
        pass
    except Exception:
        # Error opening image, skip validation
        pass

    return (len(warnings) == 0, "; ".join(warnings) if warnings else None)


def prepare_image_content_anthropic(source: str) -> Dict[str, Any]:
    """
    Prepare image content in Anthropic's vision API format.

    Args:
        source: URL or file path to the image

    Returns:
        dict: Image content formatted for Anthropic API

    Raises:
        FileNotFoundError: If local file doesn't exist
    """
    if is_url(source):
        # URL-based image
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": source
            }
        }
    else:
        # Local file - encode to base64
        # Validate size and warn if needed
        is_valid, warning = validate_image_size(source)
        if not is_valid and warning:
            import warnings
            warnings.warn(f"Image validation warning: {warning}. The API may reject this image.")

        base64_image = encode_image_to_base64(source)
        mime_type = get_image_mime_type(source)

        # Anthropic format: raw base64 data (no data URI prefix)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": base64_image
            }
        }


def prepare_vision_content_anthropic(text: str, images: list) -> list:
    """
    Prepare multi-part content for Anthropic vision requests.

    IMPORTANT: Anthropic recommends placing images BEFORE text for optimal performance.

    Args:
        text: Text prompt/message
        images: List of image sources (URLs or file paths)

    Returns:
        list: Multi-part content array for Anthropic vision API
    """
    content = []

    # Anthropic best practice: images BEFORE text
    for image_source in images:
        image_content = prepare_image_content_anthropic(image_source)
        content.append(image_content)

    # Text comes after images
    content.append({"type": "text", "text": text})

    return content
