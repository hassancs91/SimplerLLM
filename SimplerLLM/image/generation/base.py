from enum import Enum
from SimplerLLM.utils.custom_verbose import verbose_print


class ImageProvider(Enum):
    """Enumeration of supported image generation providers."""
    OPENAI_DALL_E = 1
    STABILITY_AI = 2
    GOOGLE_GEMINI = 3
    SEEDREAM = 4
    # Future providers can be added here:
    # MIDJOURNEY = 5
    # REPLICATE = 6


class ImageSize(Enum):
    """User-friendly image size options that map to provider-specific dimensions."""
    # Standard sizes
    SQUARE = "square"           # 1:1
    HORIZONTAL = "horizontal"   # 16:9
    VERTICAL = "vertical"       # 9:16

    # Portrait ratios
    PORTRAIT_3_4 = "portrait_3_4"   # 3:4 aspect ratio, closest to A4 paper
    PORTRAIT_2_3 = "portrait_2_3"   # 2:3 aspect ratio
    PORTRAIT_4_5 = "portrait_4_5"   # 4:5 aspect ratio (Instagram portrait)

    # Landscape ratios
    LANDSCAPE_3_2 = "landscape_3_2" # 3:2 aspect ratio (classic photo)
    LANDSCAPE_4_3 = "landscape_4_3" # 4:3 aspect ratio (standard monitor)
    LANDSCAPE_5_4 = "landscape_5_4" # 5:4 aspect ratio

    # Ultrawide
    ULTRAWIDE = "ultrawide"         # 21:9 aspect ratio (cinematic)


class ImageGenerator:
    """
    Base class for Image Generation functionality.
    Provides a unified interface across different image generation providers.
    """

    def __init__(
        self,
        provider=ImageProvider.OPENAI_DALL_E,
        model_name="dall-e-3",
        api_key=None,
        verbose=False,
    ):
        """
        Initialize ImageGenerator instance.

        Args:
            provider: Image generation provider to use (ImageProvider enum)
            model_name: Model to use (e.g., "dall-e-3", "dall-e-2")
            api_key: API key for the provider (uses env var if not provided)
            verbose: Enable verbose logging
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.verbose = verbose

        if self.verbose:
            verbose_print(
                f"Initializing {provider.name} Image Generator with model: {model_name}",
                "info"
            )

    @staticmethod
    def create(
        provider=None,
        model_name=None,
        api_key=None,
        verbose=False,
    ):
        """
        Factory method to create ImageGenerator instances for different providers.

        Args:
            provider: Image generation provider (ImageProvider enum)
            model_name: Model to use (provider-specific)
            api_key: API key for the provider
            verbose: Enable verbose logging

        Returns:
            Provider-specific ImageGenerator instance (e.g., OpenAIImageGenerator)
        """
        if provider == ImageProvider.OPENAI_DALL_E:
            from .wrappers.openai_wrapper import OpenAIImageGenerator
            return OpenAIImageGenerator(
                provider=provider,
                model_name=model_name or "dall-e-3",
                api_key=api_key,
                verbose=verbose,
            )
        elif provider == ImageProvider.STABILITY_AI:
            from .wrappers.stability_wrapper import StabilityImageGenerator
            return StabilityImageGenerator(
                provider=provider,
                model_name=model_name or "stable-image-core",
                api_key=api_key,
                verbose=verbose,
            )
        elif provider == ImageProvider.GOOGLE_GEMINI:
            from .wrappers.google_wrapper import GoogleImageGenerator
            return GoogleImageGenerator(
                provider=provider,
                model_name=model_name or "gemini-2.5-flash-image",
                api_key=api_key,
                verbose=verbose,
            )
        elif provider == ImageProvider.SEEDREAM:
            from .wrappers.seedream_wrapper import SeedreamImageGenerator
            return SeedreamImageGenerator(
                provider=provider,
                model_name=model_name or "seedream-4-5-251128",
                api_key=api_key,
                verbose=verbose,
            )
        # Future providers can be added here
        # elif provider == ImageProvider.MIDJOURNEY:
        #     from .wrappers.midjourney_wrapper import MidjourneyImageGenerator
        #     return MidjourneyImageGenerator(...)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def prepare_params(self, size=None, model=None, **kwargs):
        """
        Prepare parameters for image generation, using instance defaults
        if parameters are not provided.

        Args:
            size: Size to use (None = use default)
            model: Model to use (None = use instance default)
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary of parameters
        """
        params = {
            "model_name": model if model is not None else self.model_name,
            "size": size if size is not None else ImageSize.SQUARE,
        }
        # Add any additional kwargs for provider-specific parameters
        params.update(kwargs)
        return params

    def _map_size_to_dimensions(self, size, provider=None):
        """
        Map user-friendly size options to provider-specific dimensions.

        Args:
            size: ImageSize enum or string dimension (e.g., "1024x1024")
            provider: Optional provider override (uses self.provider if not provided)

        Returns:
            String dimension appropriate for the provider (e.g., "1024x1024")
        """
        # If size is already a string (custom dimensions), return as-is
        if isinstance(size, str):
            return size

        # If size is not an ImageSize enum, try to convert it
        if not isinstance(size, ImageSize):
            try:
                size = ImageSize(size)
            except ValueError:
                # If conversion fails, return the original value
                return str(size)

        provider = provider or self.provider

        # Map sizes based on provider
        if provider == ImageProvider.OPENAI_DALL_E:
            # OpenAI has limited size options, map to closest equivalent
            size_map = {
                ImageSize.SQUARE: "1024x1024",
                ImageSize.HORIZONTAL: "1792x1024",
                ImageSize.VERTICAL: "1024x1792",
                ImageSize.PORTRAIT_3_4: "1024x1792",      # Map to vertical
                ImageSize.PORTRAIT_2_3: "1024x1792",      # Map to vertical
                ImageSize.PORTRAIT_4_5: "1024x1792",      # Map to vertical
                ImageSize.LANDSCAPE_3_2: "1792x1024",     # Map to horizontal
                ImageSize.LANDSCAPE_4_3: "1792x1024",     # Map to horizontal
                ImageSize.LANDSCAPE_5_4: "1024x1024",     # Map to square (closest)
                ImageSize.ULTRAWIDE: "1792x1024",         # Map to horizontal
            }
            return size_map.get(size, "1024x1024")

        elif provider == ImageProvider.STABILITY_AI:
            # Stability AI uses aspect ratios
            size_map = {
                ImageSize.SQUARE: "1:1",
                ImageSize.HORIZONTAL: "16:9",
                ImageSize.VERTICAL: "9:16",
                ImageSize.PORTRAIT_3_4: "3:4",
                ImageSize.PORTRAIT_2_3: "2:3",
                ImageSize.PORTRAIT_4_5: "4:5",
                ImageSize.LANDSCAPE_3_2: "3:2",
                ImageSize.LANDSCAPE_4_3: "4:3",
                ImageSize.LANDSCAPE_5_4: "5:4",
                ImageSize.ULTRAWIDE: "21:9",
            }
            return size_map.get(size, "1:1")

        elif provider == ImageProvider.GOOGLE_GEMINI:
            # Google Gemini supports: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
            size_map = {
                ImageSize.SQUARE: "1:1",
                ImageSize.HORIZONTAL: "16:9",
                ImageSize.VERTICAL: "9:16",
                ImageSize.PORTRAIT_3_4: "3:4",
                ImageSize.PORTRAIT_2_3: "2:3",
                ImageSize.PORTRAIT_4_5: "4:5",
                ImageSize.LANDSCAPE_3_2: "3:2",
                ImageSize.LANDSCAPE_4_3: "4:3",
                ImageSize.LANDSCAPE_5_4: "5:4",
                ImageSize.ULTRAWIDE: "21:9",
            }
            return size_map.get(size, "1:1")

        elif provider == ImageProvider.SEEDREAM:
            # Seedream uses resolution presets (2K, 4K) rather than aspect ratios
            size_map = {
                ImageSize.SQUARE: "2K",
                ImageSize.HORIZONTAL: "2K",
                ImageSize.VERTICAL: "2K",
                ImageSize.PORTRAIT_3_4: "2K",
                ImageSize.PORTRAIT_2_3: "2K",
                ImageSize.PORTRAIT_4_5: "2K",
                ImageSize.LANDSCAPE_3_2: "2K",
                ImageSize.LANDSCAPE_4_3: "2K",
                ImageSize.LANDSCAPE_5_4: "2K",
                ImageSize.ULTRAWIDE: "2K",
            }
            return size_map.get(size, "2K")

        # Default fallback
        return "1024x1024"

    def set_provider(self, provider):
        """
        Set the image generation provider.

        Args:
            provider: ImageProvider enum value
        """
        if not isinstance(provider, ImageProvider):
            raise ValueError("Provider must be an instance of ImageProvider Enum")
        self.provider = provider
