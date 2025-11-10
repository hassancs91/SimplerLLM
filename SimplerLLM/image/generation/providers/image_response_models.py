from pydantic import BaseModel
from typing import Any, Optional, Union


class ImageGenerationResponse(BaseModel):
    """Full response from image generation with metadata."""

    image_data: Union[bytes, str]
    """The generated image - can be bytes, URL, or file path"""

    model: str
    """The model used (e.g., 'dall-e-3', 'dall-e-2')"""

    prompt: str
    """The original prompt provided by the user"""

    revised_prompt: Optional[str] = None
    """The revised/enhanced prompt used by the model (if provided by the model)"""

    size: str
    """The image dimensions (e.g., '1024x1024', '1792x1024')"""

    quality: Optional[str] = None
    """The quality setting (e.g., 'standard', 'hd' for DALL-E 3)"""

    style: Optional[str] = None
    """The style setting (e.g., 'vivid', 'natural' for DALL-E 3)"""

    process_time: float
    """Time taken to generate the image in seconds"""

    provider: Optional[str] = None
    """The image provider used (e.g., 'OPENAI_DALL_E')"""

    file_size: Optional[int] = None
    """Size of the image data in bytes"""

    output_path: Optional[str] = None
    """File path if image was saved to disk"""

    llm_provider_response: Optional[Any] = None
    """Raw response from the provider API"""

    class Config:
        json_schema_extra = {
            "example": {
                "image_data": "https://oaidalleapiprodscus.blob.core.windows.net/...",
                "model": "dall-e-3",
                "prompt": "A serene landscape with mountains",
                "revised_prompt": "A peaceful mountain landscape at sunset with snow-capped peaks",
                "size": "1024x1024",
                "quality": "standard",
                "style": "vivid",
                "process_time": 8.45,
                "provider": "OPENAI_DALL_E",
                "file_size": 245760,
                "output_path": "output/image.png",
            }
        }
