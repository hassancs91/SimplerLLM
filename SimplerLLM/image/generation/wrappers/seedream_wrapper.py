import SimplerLLM.image.generation.providers.seedream_image as seedream_image
import os
from ..base import ImageGenerator, ImageSize
from SimplerLLM.utils.custom_verbose import verbose_print


class SeedreamImageGenerator(ImageGenerator):
    """
    BytePlus Seedream Image Generation wrapper.
    Provides a unified interface for Seedream image generation models
    with support for text-to-image and image-to-image editing.
    """

    def __init__(self, provider, model_name, api_key, verbose=False):
        """
        Initialize Seedream Image Generator instance.

        Args:
            provider: ImageProvider.SEEDREAM
            model_name: Model to use (default: seedream-4-5-251128)
            api_key: BytePlus ARK API key (uses ARK_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        super().__init__(provider, model_name, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("ARK_API_KEY", "")

    def generate_image(
        self,
        prompt: str,
        size=ImageSize.SQUARE,
        watermark: bool = False,
        model: str = None,
        output_format: str = "url",
        output_path: str = None,
        full_response: bool = False,
        image: str = None,
        **kwargs,
    ):
        """
        Generate image from text prompt using BytePlus Seedream.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or size string ("2K", "4K", or custom like "2048x2048")
                  Default: ImageSize.SQUARE (maps to "2K")
            watermark: If True, adds "AI generated" watermark to bottom-right corner
                      Default: False
            model: Model to use (None = use instance default)
                   Default: seedream-4-5-251128
            output_format: How to return the image (default: "url")
                          Options: "url" (returns image URL),
                                   "bytes" (downloads and returns bytes),
                                   "file" (saves to file, requires output_path)
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            image: Optional URL of reference image for image-to-image generation
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": image URL string
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.SEEDREAM)
            >>> # Get image URL
            >>> image_url = img_gen.generate_image("A serene landscape")
            >>> # Save to file
            >>> path = img_gen.generate_image("A city", output_format="file", output_path="city.png")
            >>> # Image-to-image generation
            >>> edited_url = img_gen.generate_image(
            ...     "Transform this into a watercolor painting",
            ...     image="https://example.com/photo.jpg"
            ... )
        """
        # Validate input
        if not prompt:
            if self.verbose:
                verbose_print("Error: Prompt parameter is required", "error")
            raise ValueError("Prompt parameter is required for image generation")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name

        # Map size to Seedream size format
        seedream_size = self._map_size_to_seedream_size(size)

        if self.verbose:
            verbose_print(
                f"Generating image - Model: {model_to_use}, Size: {seedream_size}",
                "info"
            )
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if image:
                verbose_print(f"Reference image: {image}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine response format for API
        if output_format == "bytes":
            api_response_format = "b64_json"
            api_output_path = None
        elif output_format == "file":
            api_response_format = "url"
            api_output_path = output_path
        else:
            api_response_format = "url"
            api_output_path = output_path if output_path else None

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "size": seedream_size,
            "watermark": watermark,
            "response_format": api_response_format,
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "image": image,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = seedream_image.generate_image(**params)

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
        watermark: bool = False,
        model: str = None,
        output_format: str = "url",
        output_path: str = None,
        full_response: bool = False,
        image: str = None,
        **kwargs,
    ):
        """
        Asynchronously generate image from text prompt using BytePlus Seedream.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or size string ("2K", "4K", or custom like "2048x2048")
                  Default: ImageSize.SQUARE (maps to "2K")
            watermark: If True, adds "AI generated" watermark to bottom-right corner
            model: Model to use (None = use instance default)
            output_format: How to return the image (default: "url")
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            image: Optional URL of reference image for image-to-image generation
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": image URL string
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.SEEDREAM)
            >>> image_url = await img_gen.generate_image_async("A serene landscape")
        """
        # Validate input
        if not prompt:
            if self.verbose:
                verbose_print("Error: Prompt parameter is required", "error")
            raise ValueError("Prompt parameter is required for image generation")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name

        # Map size to Seedream size format
        seedream_size = self._map_size_to_seedream_size(size)

        if self.verbose:
            verbose_print(
                f"Generating image (async) - Model: {model_to_use}, Size: {seedream_size}",
                "info"
            )
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if image:
                verbose_print(f"Reference image: {image}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine response format for API
        if output_format == "bytes":
            api_response_format = "b64_json"
            api_output_path = None
        elif output_format == "file":
            api_response_format = "url"
            api_output_path = output_path
        else:
            api_response_format = "url"
            api_output_path = output_path if output_path else None

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "size": seedream_size,
            "watermark": watermark,
            "response_format": api_response_format,
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "image": image,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await seedream_image.generate_image_async(**params)

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

    def edit_image(
        self,
        image_source: str,
        edit_prompt: str,
        size=ImageSize.SQUARE,
        watermark: bool = False,
        model: str = None,
        output_format: str = "url",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Edit an existing image using text instructions with BytePlus Seedream.

        Args:
            image_source: URL of the source image to edit (required)
            edit_prompt: Text instructions for how to edit the image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or size string ("2K", "4K", or custom like "2048x2048")
                  Default: ImageSize.SQUARE (maps to "2K")
            watermark: If True, adds "AI generated" watermark to bottom-right corner
            model: Model to use (None = use instance default)
            output_format: How to return the image (default: "url")
                          Options: "url" (returns image URL),
                                   "bytes" (downloads and returns bytes),
                                   "file" (saves to file, requires output_path)
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": image URL string
            If output_format="bytes": edited image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.SEEDREAM)
            >>> # Edit image and get URL
            >>> edited_url = img_gen.edit_image(
            ...     image_source="https://example.com/original.jpg",
            ...     edit_prompt="Change the background to a sunset sky"
            ... )
            >>> # Edit and save to file
            >>> path = img_gen.edit_image(
            ...     image_source="https://example.com/photo.jpg",
            ...     edit_prompt="Make it black and white",
            ...     output_format="file",
            ...     output_path="output/edited.png"
            ... )
        """
        # Validate input
        if not image_source:
            if self.verbose:
                verbose_print("Error: image_source parameter is required", "error")
            raise ValueError("image_source parameter is required for image editing")

        if not edit_prompt:
            if self.verbose:
                verbose_print("Error: edit_prompt parameter is required", "error")
            raise ValueError("edit_prompt parameter is required for image editing")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name

        # Map size to Seedream size format
        seedream_size = self._map_size_to_seedream_size(size)

        if self.verbose:
            verbose_print(
                f"Editing image - Model: {model_to_use}, Size: {seedream_size}",
                "info"
            )
            verbose_print(f"Edit prompt: {edit_prompt[:100]}...", "debug") if len(edit_prompt) > 100 else verbose_print(f"Edit prompt: {edit_prompt}", "debug")
            verbose_print(f"Source image: {image_source}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine response format for API
        if output_format == "bytes":
            api_response_format = "b64_json"
            api_output_path = None
        elif output_format == "file":
            api_response_format = "url"
            api_output_path = output_path
        else:
            api_response_format = "url"
            api_output_path = output_path if output_path else None

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "model_name": model_to_use,
            "size": seedream_size,
            "watermark": watermark,
            "response_format": api_response_format,
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = seedream_image.edit_image(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image edited successfully in {response.process_time:.2f}s", "info")
                else:
                    verbose_print("Image edited successfully", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error editing image: {str(e)}", "error")
            raise

    async def edit_image_async(
        self,
        image_source: str,
        edit_prompt: str,
        size=ImageSize.SQUARE,
        watermark: bool = False,
        model: str = None,
        output_format: str = "url",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Asynchronously edit an existing image using text instructions with BytePlus Seedream.

        Args:
            image_source: URL of the source image to edit (required)
            edit_prompt: Text instructions for how to edit the image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or size string ("2K", "4K", or custom like "2048x2048")
                  Default: ImageSize.SQUARE (maps to "2K")
            watermark: If True, adds "AI generated" watermark to bottom-right corner
            model: Model to use (None = use instance default)
            output_format: How to return the image (default: "url")
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": image URL string
            If output_format="bytes": edited image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.SEEDREAM)
            >>> edited_url = await img_gen.edit_image_async(
            ...     image_source="https://example.com/original.jpg",
            ...     edit_prompt="Add a vintage filter"
            ... )
        """
        # Validate input
        if not image_source:
            if self.verbose:
                verbose_print("Error: image_source parameter is required", "error")
            raise ValueError("image_source parameter is required for image editing")

        if not edit_prompt:
            if self.verbose:
                verbose_print("Error: edit_prompt parameter is required", "error")
            raise ValueError("edit_prompt parameter is required for image editing")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name

        # Map size to Seedream size format
        seedream_size = self._map_size_to_seedream_size(size)

        if self.verbose:
            verbose_print(
                f"Editing image (async) - Model: {model_to_use}, Size: {seedream_size}",
                "info"
            )
            verbose_print(f"Edit prompt: {edit_prompt[:100]}...", "debug") if len(edit_prompt) > 100 else verbose_print(f"Edit prompt: {edit_prompt}", "debug")
            verbose_print(f"Source image: {image_source}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine response format for API
        if output_format == "bytes":
            api_response_format = "b64_json"
            api_output_path = None
        elif output_format == "file":
            api_response_format = "url"
            api_output_path = output_path
        else:
            api_response_format = "url"
            api_output_path = output_path if output_path else None

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "model_name": model_to_use,
            "size": seedream_size,
            "watermark": watermark,
            "response_format": api_response_format,
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await seedream_image.edit_image_async(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image edited successfully in {response.process_time:.2f}s", "info")
                else:
                    verbose_print("Image edited successfully", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error editing image: {str(e)}", "error")
            raise

    def _map_size_to_seedream_size(self, size):
        """
        Map ImageSize enum or custom size to Seedream size format.

        Args:
            size: ImageSize enum or string (size preset or dimensions)

        Returns:
            str: Size string (e.g., "2K", "4K", or custom dimensions)
        """
        # If size is already a string, return as-is
        if isinstance(size, str):
            return size

        # If size is not an ImageSize enum, try to convert it
        if not isinstance(size, ImageSize):
            try:
                size = ImageSize(size)
            except ValueError:
                # If conversion fails, return default
                return "2K"

        # Map ImageSize enum to Seedream sizes
        # Seedream uses resolution presets rather than aspect ratios
        # All ImageSize values map to 2K by default (user can override with string)
        size_map = {
            ImageSize.SQUARE: "2K",
            ImageSize.HORIZONTAL: "2K",
            ImageSize.VERTICAL: "2K",
            ImageSize.PORTRAIT_3_4: "2K",
        }

        return size_map.get(size, "2K")
