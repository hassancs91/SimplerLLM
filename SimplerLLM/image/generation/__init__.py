"""
Image generation module.
Provides unified interface for generating images from text prompts
across multiple providers (OpenAI DALL-E, Stability AI, Google Gemini, etc.).
"""

from .base import ImageGenerator, ImageProvider, ImageSize
from .wrappers.openai_wrapper import OpenAIImageGenerator
from .wrappers.stability_wrapper import StabilityImageGenerator
from .wrappers.google_wrapper import GoogleImageGenerator
from .providers.image_response_models import ImageGenerationResponse

__all__ = [
    'ImageGenerator',
    'ImageProvider',
    'ImageSize',
    'OpenAIImageGenerator',
    'StabilityImageGenerator',
    'GoogleImageGenerator',
    'ImageGenerationResponse',
]
