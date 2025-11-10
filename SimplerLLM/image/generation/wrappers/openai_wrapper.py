import SimplerLLM.image.generation.providers.openai_image as openai_image
import os
from ..base import ImageGenerator, ImageSize
from SimplerLLM.utils.custom_verbose import verbose_print


class OpenAIImageGenerator(ImageGenerator):
    """
    OpenAI DALL-E Image Generation wrapper.
    Provides a unified interface for OpenAI image generation models.
    """

    def __init__(self, provider, model_name, api_key, verbose=False):
        """
        Initialize OpenAI Image Generator instance.

        Args:
            provider: ImageProvider.OPENAI_DALL_E
            model_name: Model to use ("dall-e-3" or "dall-e-2")
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        super().__init__(provider, model_name, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def generate_image(
        self,
        prompt: str,
        size=ImageSize.SQUARE,
        quality: str = "standard",
        style: str = "vivid",
        model: str = None,
        output_format: str = "url",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Generate image from text prompt using OpenAI DALL-E.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or string dimension (e.g., "1024x1024")
                  Default: ImageSize.SQUARE
            quality: Quality setting for DALL-E 3 (default: "standard")
                    Options: "standard", "hd"
            style: Style setting for DALL-E 3 (default: "vivid")
                   Options: "vivid", "natural"
            model: Model to use (None = use instance default)
                   Options: "dall-e-3", "dall-e-2"
            output_format: How to return the image (default: "url")
                          Options: "url" (returns URL string),
                                   "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": URL string
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.OPENAI_DALL_E)
            >>> # Get URL
            >>> url = img_gen.generate_image("A serene landscape")
            >>> # Save to file
            >>> path = img_gen.generate_image("A city", output_format="file", output_path="city.png")
            >>> # Get full response
            >>> response = img_gen.generate_image("Abstract art", full_response=True)
        """
        # Validate input
        if not prompt:
            if self.verbose:
                verbose_print("Error: Prompt parameter is required", "error")
            raise ValueError("Prompt parameter is required for image generation")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name

        # Map size to actual dimensions
        size_dimension = self._map_size_to_dimensions(size)

        if self.verbose:
            verbose_print(
                f"Generating image - Model: {model_to_use}, Size: {size_dimension}, Quality: {quality}, Style: {style}",
                "info"
            )
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine response_format for API call
        # If output_format is "bytes", we need b64_json from API
        api_response_format = "b64_json" if output_format == "bytes" else "url"

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "size": size_dimension,
            "quality": quality,
            "style": style,
            "n": 1,  # Always generate 1 image (DALL-E 3 only supports 1)
            "response_format": api_response_format,
            "output_path": output_path if output_format == "file" else None,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = openai_image.generate_image(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image generated successfully in {response.process_time:.2f}s", "info")
                    if response.revised_prompt:
                        verbose_print(f"Revised prompt: {response.revised_prompt[:100]}...", "debug")
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
        quality: str = "standard",
        style: str = "vivid",
        model: str = None,
        output_format: str = "url",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Asynchronously generate image from text prompt using OpenAI DALL-E.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or string dimension (e.g., "1024x1024")
                  Default: ImageSize.SQUARE
            quality: Quality setting for DALL-E 3 (default: "standard")
                    Options: "standard", "hd"
            style: Style setting for DALL-E 3 (default: "vivid")
                   Options: "vivid", "natural"
            model: Model to use (None = use instance default)
                   Options: "dall-e-3", "dall-e-2"
            output_format: How to return the image (default: "url")
                          Options: "url" (returns URL string),
                                   "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": URL string
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.OPENAI_DALL_E)
            >>> url = await img_gen.generate_image_async("A serene landscape")
        """
        # Validate input
        if not prompt:
            if self.verbose:
                verbose_print("Error: Prompt parameter is required", "error")
            raise ValueError("Prompt parameter is required for image generation")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name

        # Map size to actual dimensions
        size_dimension = self._map_size_to_dimensions(size)

        if self.verbose:
            verbose_print(
                f"Generating image (async) - Model: {model_to_use}, Size: {size_dimension}, Quality: {quality}, Style: {style}",
                "info"
            )
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine response_format for API call
        # If output_format is "bytes", we need b64_json from API
        api_response_format = "b64_json" if output_format == "bytes" else "url"

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "size": size_dimension,
            "quality": quality,
            "style": style,
            "n": 1,  # Always generate 1 image (DALL-E 3 only supports 1)
            "response_format": api_response_format,
            "output_path": output_path if output_format == "file" else None,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await openai_image.generate_image_async(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image generated successfully in {response.process_time:.2f}s", "info")
                    if response.revised_prompt:
                        verbose_print(f"Revised prompt: {response.revised_prompt[:100]}...", "debug")
                else:
                    verbose_print("Image generated successfully", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating image: {str(e)}", "error")
            raise
