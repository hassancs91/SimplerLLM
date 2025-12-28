"""
Image Service - Handles image generation using SimplerLLM
"""
import os
from typing import Dict, Optional
from SimplerLLM.image.generation import ImageGenerator, ImageProvider


class ImageService:
    """Service for generating images using AI models via SimplerLLM."""

    # Provider configuration
    PROVIDERS = {
        'google': {
            'id': 'google',
            'name': 'Google AI',
            'env_key': 'GEMINI_API_KEY',
            'models': [
                {
                    'id': 'gemini-2.5-flash-image-preview',
                    'name': 'Gemini 2.5 Flash (Image Preview)',
                    'description': 'Fast image generation with Gemini 2.5 Flash'
                }
            ]
        }
    }

    def __init__(self, settings_service):
        self._settings_service = settings_service

    def _get_api_key(self, provider: str = 'google') -> Optional[str]:
        """Get API key for a provider."""
        # First check settings
        api_key = self._settings_service.get_api_key(provider)
        if api_key:
            return api_key

        # Fallback to environment variable
        provider_config = self.PROVIDERS.get(provider)
        if provider_config:
            return os.environ.get(provider_config['env_key'])
        return None

    def generate_image(self, prompt: str, model: str = 'gemini-2.5-flash-image-preview', provider: str = 'google', aspect_ratio: str = '1:1') -> Dict:
        """
        Generate an image from a text prompt using SimplerLLM.

        Args:
            prompt: Text description of the image to generate
            model: Model ID to use for generation
            provider: Provider ID (currently only 'google' is supported)
            aspect_ratio: Aspect ratio for the image (e.g., '1:1', '16:9', '9:16')

        Returns:
            Dict with 'success', 'image_bytes', and optionally 'error'
        """
        api_key = self._get_api_key(provider)
        if not api_key:
            return {
                'success': False,
                'error': f'API key not configured for {provider}. Please add your API key in settings.'
            }

        try:
            # Create SimplerLLM image generator
            img_gen = ImageGenerator.create(
                provider=ImageProvider.GOOGLE_GEMINI,
                model_name=model,
                api_key=api_key
            )

            # Generate the image (returns bytes by default)
            image_bytes = img_gen.generate_image(prompt=prompt, size=aspect_ratio)

            if image_bytes:
                return {
                    'success': True,
                    'image_bytes': image_bytes
                }
            else:
                return {
                    'success': False,
                    'error': 'No image was generated'
                }

        except Exception as e:
            error_message = str(e)
            # Handle specific errors
            if 'API_KEY' in error_message.upper() or 'authentication' in error_message.lower():
                error_message = 'Invalid API key. Please check your API key in settings.'
            elif 'quota' in error_message.lower():
                error_message = 'API quota exceeded. Please try again later.'
            elif 'safety' in error_message.lower():
                error_message = 'The prompt was blocked due to safety filters. Please try a different prompt.'

            return {
                'success': False,
                'error': error_message
            }

    def validate_api_key(self, api_key: str, provider: str = 'google') -> Dict:
        """
        Validate an API key by making a simple API call.

        Returns:
            Dict with 'valid' boolean and optionally 'error'
        """
        try:
            if provider == 'google':
                # Use google.generativeai to validate the key
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                # Try to list models to validate the key
                list(genai.list_models())
                return {'valid': True}
            else:
                return {'valid': False, 'error': f'Unknown provider: {provider}'}
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def get_providers(self) -> Dict:
        """Get list of available providers and their status."""
        providers = []
        for provider_id, config in self.PROVIDERS.items():
            has_key = self._settings_service.has_api_key(provider_id)
            providers.append({
                'id': config['id'],
                'name': config['name'],
                'configured': has_key,
                'models': config['models']
            })
        return {'providers': providers}
