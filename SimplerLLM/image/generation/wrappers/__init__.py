"""
Image generation wrapper classes.
Provides unified interfaces for different image generation providers.
"""

from .openai_wrapper import OpenAIImageGenerator
from .stability_wrapper import StabilityImageGenerator
from .google_wrapper import GoogleImageGenerator

__all__ = [
    'OpenAIImageGenerator',
    'StabilityImageGenerator',
    'GoogleImageGenerator',
]
