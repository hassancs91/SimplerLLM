import SimplerLLM.image.generation.providers.stability_image as stability_image
import os
from ..base import ImageGenerator, ImageSize
from SimplerLLM.utils.custom_verbose import verbose_print


class StabilityImageGenerator(ImageGenerator):
    """
    Stability AI Image Generation wrapper.
    Provides a unified interface for Stability AI image generation models
    (Ultra, Core, and Stable Diffusion 3.5 models).
    """

    def __init__(self, provider, model_name, api_key, verbose=False):
        """
        Initialize Stability AI Image Generator instance.

        Args:
            provider: ImageProvider.STABILITY_AI
            model_name: Model to use (stable-image-ultra, stable-image-core,
                       sd3.5-large, sd3.5-large-turbo, sd3.5-medium, sd3.5-flash)
            api_key: Stability API key (uses STABILITY_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        super().__init__(provider, model_name, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("STABILITY_API_KEY", "")

    def generate_image(
        self,
        prompt: str,
        size=ImageSize.SQUARE,
        negative_prompt: str = None,
        style_preset: str = None,
        seed: int = 0,
        cfg_scale: float = None,
        model: str = None,
        output_format: str = "bytes",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Generate image from text prompt using Stability AI.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or aspect ratio string (e.g., "16:9", "1:1", "21:9")
                  Default: ImageSize.SQUARE
            negative_prompt: Keywords of what you do NOT want in the image
                           Helps refine the output by specifying unwanted elements
            style_preset: Style to guide the generation
                         Options: 3d-model, analog-film, anime, cinematic, comic-book,
                                 digital-art, enhance, fantasy-art, isometric, line-art,
                                 low-poly, modeling-compound, neon-punk, origami,
                                 photographic, pixel-art, tile-texture
            seed: Randomness seed (0 for random, 1-4294967294 for reproducible results)
            cfg_scale: How strictly to follow the prompt (1-10, higher = stricter)
                      Default varies by model (4 for Large/Medium, 1 for Turbo/Flash)
            model: Model to use (None = use instance default)
                   Options: stable-image-ultra, stable-image-core,
                           sd3.5-large, sd3.5-large-turbo, sd3.5-medium, sd3.5-flash
            output_format: How to return the image (default: "bytes")
                          Options: "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
                          Note: Stability doesn't provide URLs, so "url" will save
                                to temp file and return path
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.STABILITY_AI)
            >>> # Get bytes
            >>> image_bytes = img_gen.generate_image("A serene landscape")
            >>> # Save to file
            >>> path = img_gen.generate_image("A city", output_format="file", output_path="city.png")
            >>> # Use Stability-specific features
            >>> img_gen.generate_image(
            ...     "A dragon",
            ...     negative_prompt="blurry, distorted",
            ...     style_preset="fantasy-art",
            ...     seed=12345
            ... )
        """
        # Validate input
        if not prompt:
            if self.verbose:
                verbose_print("Error: Prompt parameter is required", "error")
            raise ValueError("Prompt parameter is required for image generation")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name

        # Map size to aspect ratio
        aspect_ratio = self._map_size_to_aspect_ratio(size)

        if self.verbose:
            verbose_print(
                f"Generating image - Model: {model_to_use}, Aspect Ratio: {aspect_ratio}",
                "info"
            )
            if style_preset:
                verbose_print(f"Style Preset: {style_preset}", "debug")
            if negative_prompt:
                verbose_print(f"Negative Prompt: {negative_prompt[:50]}...", "debug")
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine actual output format for API
        # Stability returns bytes by default, so we handle "url" specially
        if output_format == "url":
            # Stability doesn't provide URLs, so we'll save to a temp file
            if self.verbose:
                verbose_print("Note: Stability AI doesn't provide URLs. Saving to file instead.", "warning")
            api_output_path = output_path or "output/stability_image.png"
            output_format = "file"
        else:
            api_output_path = output_path if output_format == "file" else None

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "aspect_ratio": aspect_ratio,
            "negative_prompt": negative_prompt,
            "style_preset": style_preset,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "output_format": "png",  # File format (png, jpeg, webp)
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = stability_image.generate_image(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image generated successfully in {response.process_time:.2f}s", "info")
                else:
                    verbose_print("Image generated successfully", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating image: {str(e)}", "error")
            raise

    async def generate_image_async(
        self,
        prompt: str,
        size=ImageSize.SQUARE,
        negative_prompt: str = None,
        style_preset: str = None,
        seed: int = 0,
        cfg_scale: float = None,
        model: str = None,
        output_format: str = "bytes",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Asynchronously generate image from text prompt using Stability AI.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or aspect ratio string (e.g., "16:9", "1:1", "21:9")
                  Default: ImageSize.SQUARE
            negative_prompt: Keywords of what you do NOT want in the image
            style_preset: Style to guide the generation (see generate_image for options)
            seed: Randomness seed (0 for random)
            cfg_scale: How strictly to follow the prompt (1-10, higher = stricter)
            model: Model to use (None = use instance default)
            output_format: How to return the image (default: "bytes")
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.STABILITY_AI)
            >>> image_bytes = await img_gen.generate_image_async("A serene landscape")
        """
        # Validate input
        if not prompt:
            if self.verbose:
                verbose_print("Error: Prompt parameter is required", "error")
            raise ValueError("Prompt parameter is required for image generation")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name

        # Map size to aspect ratio
        aspect_ratio = self._map_size_to_aspect_ratio(size)

        if self.verbose:
            verbose_print(
                f"Generating image (async) - Model: {model_to_use}, Aspect Ratio: {aspect_ratio}",
                "info"
            )
            if style_preset:
                verbose_print(f"Style Preset: {style_preset}", "debug")
            if negative_prompt:
                verbose_print(f"Negative Prompt: {negative_prompt[:50]}...", "debug")
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine actual output format for API
        if output_format == "url":
            if self.verbose:
                verbose_print("Note: Stability AI doesn't provide URLs. Saving to file instead.", "warning")
            api_output_path = output_path or "output/stability_image.png"
            output_format = "file"
        else:
            api_output_path = output_path if output_format == "file" else None

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "aspect_ratio": aspect_ratio,
            "negative_prompt": negative_prompt,
            "style_preset": style_preset,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "output_format": "png",
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await stability_image.generate_image_async(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image generated successfully in {response.process_time:.2f}s", "info")
                else:
                    verbose_print("Image generated successfully", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating image: {str(e)}", "error")
            raise

    def _map_size_to_aspect_ratio(self, size):
        """
        Map ImageSize enum or custom size to Stability AI aspect ratio format.

        Args:
            size: ImageSize enum or string (aspect ratio or dimensions)

        Returns:
            str: Aspect ratio string (e.g., "1:1", "16:9")
        """
        # If size is already a string, check if it's an aspect ratio format
        if isinstance(size, str):
            # If it contains ":", assume it's already an aspect ratio
            if ":" in size:
                return size
            # If it's dimensions like "1024x1024", try to convert to aspect ratio
            # For Stability, we'll just return the closest standard aspect ratio
            if "x" in size.lower():
                # Could implement dimension-to-aspect-ratio conversion here
                # For now, default to 1:1
                return "1:1"
            return size

        # If size is not an ImageSize enum, try to convert it
        if not isinstance(size, ImageSize):
            try:
                size = ImageSize(size)
            except ValueError:
                # If conversion fails, return default
                return "1:1"

        # Map ImageSize enum to Stability's aspect ratios
        size_map = {
            ImageSize.SQUARE: "1:1",
            ImageSize.HORIZONTAL: "16:9",
            ImageSize.VERTICAL: "9:16",
            ImageSize.PORTRAIT_3_4: "3:4",
        }

        return size_map.get(size, "1:1")
