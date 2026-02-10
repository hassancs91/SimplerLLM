"""
Image helper utilities for vision-enabled LLM requests.

This module provides utilities for encoding, validating, and preparing
images for use with vision-capable language models. Supports OpenAI,
Anthropic, and Google Gemini vision formats.

Supported Image Formats:
    - JPEG (.jpg, .jpeg) - All providers
    - PNG (.png) - All providers
    - GIF (.gif) - OpenAI, Anthropic
    - WebP (.webp) - All providers
    - BMP (.bmp) - OpenAI only
    - HEIC (.heic) - Gemini only
    - HEIF (.heif) - Gemini only

Size Limits:
    - OpenAI: 20MB per image
    - Anthropic: 5MB per image, 8000x8000 max dimensions
    - Gemini: 20MB total request size (inline), 3600 images max per request

Example:
    >>> from SimplerLLM.tools.image_helpers import prepare_vision_content
    >>>
    >>> # Prepare content for OpenAI
    >>> content = prepare_vision_content(
    ...     text="What's in this image?",
    ...     images=["photo.jpg", "https://example.com/image.png"],
    ...     detail="high"
    ... )
    >>>
    >>> # Prepare content for Gemini
    >>> from SimplerLLM.tools.image_helpers import prepare_vision_content_gemini
    >>> parts = prepare_vision_content_gemini(
    ...     text="Describe this image",
    ...     images=["photo.jpg"]
    ... )
    >>>
    >>> # Validate before sending
    >>> from SimplerLLM.tools.image_helpers import validate_image_for_openai
    >>> is_valid, warning = validate_image_for_openai("large_photo.jpg")
    >>> if not is_valid:
    ...     print(f"Warning: {warning}")
"""

import base64
import os
from typing import Dict, Any, List, Optional, Tuple
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


# OpenAI-specific constants
OPENAI_MAX_IMAGE_SIZE_MB = 20
OPENAI_SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.webp']


def validate_image_for_openai(
    source: str,
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate an image source for OpenAI's vision API requirements.

    OpenAI vision API requirements:
    - Supported formats: PNG, JPEG, GIF, WebP
    - Maximum file size: 20MB per image
    - Recommended: Images under 2000x2000 pixels for "high" detail mode

    Args:
        source: URL or file path to the image.
        verbose: If True, print validation details.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - is_valid: True if the image passes all validation checks
            - warning_message: Description of issues found, or None if valid

    Example:
        >>> is_valid, warning = validate_image_for_openai("photo.jpg")
        >>> if not is_valid:
        ...     print(f"Image issue: {warning}")
        >>> else:
        ...     print("Image is valid for OpenAI")

        >>> # Validate before preparing content
        >>> is_valid, warning = validate_image_for_openai("large_image.png")
        >>> if warning:
        ...     print(f"Warning: {warning}")
    """
    warnings = []

    # URLs are validated by the API
    if is_url(source):
        return (True, None)

    # Validate local file exists
    if not os.path.exists(source):
        return (False, f"File not found: {source}")

    # Check file extension/format
    ext = os.path.splitext(source)[1].lower()
    if ext not in OPENAI_SUPPORTED_FORMATS:
        warnings.append(
            f"Unsupported format '{ext}'. "
            f"OpenAI supports: {', '.join(OPENAI_SUPPORTED_FORMATS)}"
        )

    # Check file size (20MB limit)
    try:
        file_size = os.path.getsize(source)
        file_size_mb = file_size / (1024 * 1024)

        if file_size_mb > OPENAI_MAX_IMAGE_SIZE_MB:
            warnings.append(
                f"File size ({file_size_mb:.1f}MB) exceeds "
                f"OpenAI's {OPENAI_MAX_IMAGE_SIZE_MB}MB limit"
            )
        elif file_size_mb > 10 and verbose:
            # Warn about large files that might slow down requests
            warnings.append(
                f"Large file ({file_size_mb:.1f}MB) may slow down API requests"
            )
    except OSError as e:
        warnings.append(f"Could not read file size: {e}")

    # Check dimensions if PIL/Pillow is available
    try:
        from PIL import Image
        with Image.open(source) as img:
            width, height = img.size

            # OpenAI recommends under 2000x2000 for "high" detail
            if width > 4096 or height > 4096:
                warnings.append(
                    f"Large dimensions ({width}x{height}). "
                    f"Images over 4096px may be resized by the API"
                )
            elif (width > 2000 or height > 2000) and verbose:
                warnings.append(
                    f"Image ({width}x{height}) may use more tokens "
                    f"with 'high' detail mode"
                )
    except ImportError:
        # PIL not available, skip dimension check
        pass
    except Exception as e:
        if verbose:
            warnings.append(f"Could not read image dimensions: {e}")

    is_valid = len(warnings) == 0
    warning_message = "; ".join(warnings) if warnings else None

    return (is_valid, warning_message)


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


def prepare_vision_content(
    text: str,
    images: List[str],
    detail: str = "auto",
    validate: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepare multi-part content for vision requests combining text and images.

    This function formats text and images for OpenAI's vision API, creating
    the multi-part content structure required for vision-capable models.

    Args:
        text: Text prompt/message to accompany the images.
        images: List of image sources (URLs or local file paths).
        detail: Level of detail for image processing. Default: "auto"
            - "low": Fast processing, uses ~85 tokens per image
            - "high": Detailed analysis, uses ~765+ tokens per image
            - "auto": Let the API decide based on image size
        validate: If True, validate images before processing and log
            warnings for any issues. Default: False
        verbose: If True, print detailed validation information.
            Only used when validate=True. Default: False

    Returns:
        List[Dict[str, Any]]: Multi-part content array for OpenAI vision API,
            containing text and image_url objects.

    Example:
        >>> content = prepare_vision_content(
        ...     text="What's in these images?",
        ...     images=["photo1.jpg", "https://example.com/photo2.png"],
        ...     detail="high"
        ... )

        >>> # With validation
        >>> content = prepare_vision_content(
        ...     text="Analyze this chart",
        ...     images=["chart.png"],
        ...     validate=True,
        ...     verbose=True
        ... )

    Notes:
        - URLs are passed directly to the API without validation
        - Local files are encoded to base64 data URIs
        - Use validate=True to catch issues before API calls
    """
    import logging
    logger = logging.getLogger(__name__)

    content = [{"type": "text", "text": text}]

    for image_source in images:
        # Optionally validate each image
        if validate:
            is_valid, warning = validate_image_for_openai(image_source, verbose)
            if warning:
                if verbose:
                    logger.warning(f"[Vision] {image_source}: {warning}")
                else:
                    logger.debug(f"[Vision] {image_source}: {warning}")

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


# ============================================================================
# Google Gemini-specific vision helpers
# ============================================================================

# Gemini-specific constants
GEMINI_SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif']
GEMINI_MAX_IMAGES_PER_REQUEST = 3600
GEMINI_MAX_INLINE_SIZE_MB = 20  # Total request size limit


def get_gemini_mime_type(image_path: str) -> str:
    """
    Get MIME type for Gemini API from file extension.

    Gemini supports: image/png, image/jpeg, image/webp, image/heic, image/heif

    Args:
        image_path: Path to the image file.

    Returns:
        str: MIME type (e.g., 'image/jpeg').

    Example:
        >>> get_gemini_mime_type("photo.jpg")
        'image/jpeg'
        >>> get_gemini_mime_type("image.heic")
        'image/heic'
    """
    extension = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.heic': 'image/heic',
        '.heif': 'image/heif',
    }
    return mime_types.get(extension, 'image/jpeg')


def validate_image_for_gemini(
    source: str,
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate an image source for Google Gemini's vision API requirements.

    Gemini vision API requirements:
    - Supported formats: PNG, JPEG, WebP, HEIC, HEIF
    - Maximum total request size (with inline data): 20MB
    - Maximum images per request: 3600

    Args:
        source: URL or file path to the image.
        verbose: If True, print validation details.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - is_valid: True if the image passes all validation checks
            - warning_message: Description of issues found, or None if valid

    Example:
        >>> is_valid, warning = validate_image_for_gemini("photo.jpg")
        >>> if not is_valid:
        ...     print(f"Image issue: {warning}")
    """
    warnings = []

    # URLs are validated by the API (we download them anyway)
    if is_url(source):
        return (True, None)

    # Validate local file exists
    if not os.path.exists(source):
        return (False, f"File not found: {source}")

    # Check file extension/format
    ext = os.path.splitext(source)[1].lower()
    if ext not in GEMINI_SUPPORTED_FORMATS:
        warnings.append(
            f"Unsupported format '{ext}'. "
            f"Gemini supports: {', '.join(GEMINI_SUPPORTED_FORMATS)}"
        )

    # Check file size
    try:
        file_size = os.path.getsize(source)
        file_size_mb = file_size / (1024 * 1024)

        if file_size_mb > GEMINI_MAX_INLINE_SIZE_MB:
            warnings.append(
                f"File size ({file_size_mb:.1f}MB) may exceed "
                f"Gemini's {GEMINI_MAX_INLINE_SIZE_MB}MB request limit"
            )
        elif file_size_mb > 10 and verbose:
            warnings.append(
                f"Large file ({file_size_mb:.1f}MB) may slow down API requests"
            )
    except OSError as e:
        warnings.append(f"Could not read file size: {e}")

    is_valid = len(warnings) == 0
    warning_message = "; ".join(warnings) if warnings else None

    return (is_valid, warning_message)


def prepare_image_content_gemini(source: str) -> Dict[str, Any]:
    """
    Prepare image content in Gemini's vision API format.

    Gemini uses inline_data with mime_type and base64-encoded data.
    For URLs, the image is downloaded and encoded.

    Args:
        source: URL or file path to the image.

    Returns:
        dict: Image part formatted for Gemini API with inline_data.

    Raises:
        FileNotFoundError: If local file doesn't exist.
        requests.RequestException: If URL download fails.

    Example:
        >>> part = prepare_image_content_gemini("photo.jpg")
        >>> part["inline_data"]["mime_type"]
        'image/jpeg'

        >>> # Also works with URLs
        >>> part = prepare_image_content_gemini("https://example.com/image.png")
    """
    if is_url(source):
        # For URLs, download and encode
        import requests as req
        try:
            response = req.get(source, timeout=30)
            response.raise_for_status()

            # Detect MIME type from content-type header or URL
            content_type = response.headers.get('content-type', 'image/jpeg')
            if ';' in content_type:
                content_type = content_type.split(';')[0].strip()

            # Ensure it's an image type
            if not content_type.startswith('image/'):
                content_type = 'image/jpeg'

            base64_data = base64.b64encode(response.content).decode('utf-8')

            return {
                "inline_data": {
                    "mime_type": content_type,
                    "data": base64_data
                }
            }
        except req.RequestException as e:
            raise req.RequestException(f"Failed to download image from URL: {source}. Error: {e}")
    else:
        # Local file
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")

        base64_data = encode_image_to_base64(source)
        mime_type = get_gemini_mime_type(source)

        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64_data
            }
        }


def prepare_vision_content_gemini(
    text: str,
    images: List[str],
    validate: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepare multi-part content for Gemini vision requests.

    Creates the parts array format required by Gemini's generateContent API.
    Images are placed BEFORE text following Google's best practices.

    Args:
        text: Text prompt/message.
        images: List of image sources (URLs or file paths).
        validate: If True, validate images before processing. Default: False
        verbose: If True, print detailed validation information.

    Returns:
        list: Multi-part content array for Gemini API.

    Example:
        >>> parts = prepare_vision_content_gemini(
        ...     text="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> len(parts)
        2

        >>> # With multiple images
        >>> parts = prepare_vision_content_gemini(
        ...     text="Compare these images",
        ...     images=["img1.jpg", "img2.jpg", "https://example.com/img3.png"]
        ... )

    Notes:
        - Images are placed before text (Gemini best practice)
        - URLs are downloaded and converted to base64 inline_data
        - For very large images, consider using Gemini's File API instead
    """
    import logging
    logger = logging.getLogger(__name__)

    if len(images) > GEMINI_MAX_IMAGES_PER_REQUEST:
        raise ValueError(
            f"Too many images ({len(images)}). "
            f"Gemini supports max {GEMINI_MAX_IMAGES_PER_REQUEST} images per request."
        )

    parts = []

    # Add images first (Gemini best practice)
    for image_source in images:
        # Optionally validate each image
        if validate:
            is_valid, warning = validate_image_for_gemini(image_source, verbose)
            if not is_valid:
                logger.warning(f"[Gemini Vision] {image_source}: {warning}")
            elif warning and verbose:
                logger.debug(f"[Gemini Vision] {image_source}: {warning}")

        image_part = prepare_image_content_gemini(image_source)
        parts.append(image_part)

    # Add text after images
    parts.append({"text": text})

    return parts


# ============================================================================
# Ollama-specific vision helpers
# ============================================================================

# Ollama vision-capable model patterns
OLLAMA_VISION_PATTERNS = ["llava", "llama3.2-vision", "moondream", "bakllava", "minicpm-v"]


def is_ollama_vision_model(model_name: str) -> bool:
    """
    Check if an Ollama model supports vision based on model name patterns.

    Ollama vision-capable models include:
    - llava (and variants like llava-phi3, llava-llama3)
    - llama3.2-vision
    - moondream
    - bakllava
    - minicpm-v

    Args:
        model_name: The Ollama model name.

    Returns:
        bool: True if the model likely supports vision.

    Example:
        >>> is_ollama_vision_model("llava:latest")
        True
        >>> is_ollama_vision_model("llama3.2")
        False
        >>> is_ollama_vision_model("llama3.2-vision:11b")
        True
    """
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in OLLAMA_VISION_PATTERNS)


def prepare_image_for_ollama(source: str) -> str:
    """
    Prepare an image for Ollama's vision API format.

    Ollama expects raw base64-encoded image data (no data URI prefix).
    This function handles both local files and URLs.

    Args:
        source: URL or file path to the image.

    Returns:
        str: Base64-encoded image data.

    Raises:
        FileNotFoundError: If local file doesn't exist.
        requests.RequestException: If URL download fails.

    Example:
        >>> base64_data = prepare_image_for_ollama("photo.jpg")
        >>> len(base64_data) > 0
        True

        >>> # Also works with URLs
        >>> base64_data = prepare_image_for_ollama("https://example.com/image.png")
    """
    if is_url(source):
        # Download URL and convert to base64
        import requests as req
        try:
            response = req.get(source, timeout=30)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except req.RequestException as e:
            raise Exception(f"Failed to download image from URL: {source}. Error: {e}")
    else:
        # Local file - use existing helper
        return encode_image_to_base64(source)


def prepare_vision_message_ollama(
    text: str,
    images: List[str],
    role: str = "user"
) -> Dict[str, Any]:
    """
    Prepare a message with images for Ollama's vision API.

    Ollama's vision format places base64-encoded images in an 'images' array
    within the message, alongside the text content.

    Format:
        {
            "role": "user",
            "content": "What's in this image?",
            "images": ["base64_encoded_image_1", "base64_encoded_image_2"]
        }

    Args:
        text: Text prompt/message.
        images: List of image sources (URLs or file paths).
        role: Message role (default: "user").

    Returns:
        dict: Message formatted for Ollama vision API.

    Example:
        >>> msg = prepare_vision_message_ollama(
        ...     text="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> msg["role"]
        'user'
        >>> "images" in msg
        True

        >>> # Multiple images
        >>> msg = prepare_vision_message_ollama(
        ...     text="Compare these images",
        ...     images=["img1.jpg", "img2.jpg"]
        ... )
        >>> len(msg["images"])
        2
    """
    encoded_images = [prepare_image_for_ollama(img) for img in images]

    return {
        "role": role,
        "content": text,
        "images": encoded_images
    }


# ============================================================================
# Cohere-specific vision helpers
# ============================================================================

# Cohere vision-capable model patterns
COHERE_VISION_PATTERNS = ["vision", "aya-vision"]

# Cohere-specific constants
COHERE_SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
COHERE_MAX_IMAGE_SIZE_MB = 20


def is_cohere_vision_model(model_name: str) -> bool:
    """
    Check if a Cohere model supports vision based on model name patterns.

    Cohere vision-capable models include:
    - command-a-vision-07-2025
    - c4ai-aya-vision-8b
    - c4ai-aya-vision-32b

    Args:
        model_name: The Cohere model name.

    Returns:
        bool: True if the model likely supports vision.

    Example:
        >>> is_cohere_vision_model("command-a-vision-07-2025")
        True
        >>> is_cohere_vision_model("command-a-03-2025")
        False
    """
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in COHERE_VISION_PATTERNS)


def validate_image_for_cohere(
    source: str,
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate an image source for Cohere's vision API requirements.

    Cohere vision API requirements:
    - Supported formats: PNG, JPEG, GIF, WebP
    - Maximum file size: 20MB per image

    Args:
        source: URL or file path to the image.
        verbose: If True, print validation details.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - is_valid: True if the image passes all validation checks
            - warning_message: Description of issues found, or None if valid

    Example:
        >>> is_valid, warning = validate_image_for_cohere("photo.jpg")
        >>> if not is_valid:
        ...     print(f"Image issue: {warning}")
    """
    warnings = []

    # URLs are validated by the API
    if is_url(source):
        return (True, None)

    # Validate local file exists
    if not os.path.exists(source):
        return (False, f"File not found: {source}")

    # Check file extension/format
    ext = os.path.splitext(source)[1].lower()
    if ext not in COHERE_SUPPORTED_FORMATS:
        warnings.append(
            f"Unsupported format '{ext}'. "
            f"Cohere supports: {', '.join(COHERE_SUPPORTED_FORMATS)}"
        )

    # Check file size
    try:
        file_size = os.path.getsize(source)
        file_size_mb = file_size / (1024 * 1024)

        if file_size_mb > COHERE_MAX_IMAGE_SIZE_MB:
            warnings.append(
                f"File size ({file_size_mb:.1f}MB) exceeds "
                f"Cohere's {COHERE_MAX_IMAGE_SIZE_MB}MB limit"
            )
        elif file_size_mb > 10 and verbose:
            warnings.append(
                f"Large file ({file_size_mb:.1f}MB) may slow down API requests"
            )
    except OSError as e:
        warnings.append(f"Could not read file size: {e}")

    is_valid = len(warnings) == 0
    warning_message = "; ".join(warnings) if warnings else None

    return (is_valid, warning_message)


def prepare_image_content_cohere(source: str, detail: str = "auto") -> Dict[str, Any]:
    """
    Prepare image content in Cohere's V2 vision API format.

    Cohere V2 API uses the same format as OpenAI:
    - For URLs: {"type": "image_url", "image_url": {"url": "..."}}
    - For base64: {"type": "image_url", "image_url": {"url": "data:mime;base64,..."}}

    Args:
        source: URL or file path to the image.
        detail: Detail level ("low", "high", "auto"). Default: "auto"

    Returns:
        dict: Image content formatted for Cohere V2 API.

    Raises:
        FileNotFoundError: If local file doesn't exist.

    Example:
        >>> content = prepare_image_content_cohere("photo.jpg")
        >>> content["type"]
        'image_url'
    """
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
        # Local file - encode to base64 data URL
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")

        base64_image = encode_image_to_base64(source)
        mime_type = get_image_mime_type(source)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
                "detail": detail
            }
        }


def prepare_vision_content_cohere(
    text: str,
    images: List[str],
    detail: str = "auto",
    validate: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepare multi-part content for Cohere vision requests.

    Creates the content array format required by Cohere's V2 vision API.
    Uses the same format as OpenAI: text followed by image_url objects.

    Args:
        text: Text prompt/message.
        images: List of image sources (URLs or file paths).
        detail: Detail level for images ("low", "high", "auto"). Default: "auto"
        validate: If True, validate images before processing. Default: False
        verbose: If True, print detailed validation information.

    Returns:
        list: Multi-part content array for Cohere V2 vision API.

    Example:
        >>> content = prepare_vision_content_cohere(
        ...     text="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> len(content)
        2

        >>> # Multiple images
        >>> content = prepare_vision_content_cohere(
        ...     text="Compare these images",
        ...     images=["img1.jpg", "img2.jpg"]
        ... )
    """
    import logging
    logger = logging.getLogger(__name__)

    content = []

    # Add text first (following Cohere's example format)
    content.append({"type": "text", "text": text})

    # Add images after text
    for image_source in images:
        # Optionally validate each image
        if validate:
            valid, warning = validate_image_for_cohere(image_source, verbose)
            if not valid:
                logger.warning(f"[Cohere Vision] {image_source}: {warning}")
            elif warning and verbose:
                logger.debug(f"[Cohere Vision] {image_source}: {warning}")

        image_content = prepare_image_content_cohere(image_source, detail)
        content.append(image_content)

    return content
