import SimplerLLM.image.generation.providers.google_image as google_image
import os
from ..base import ImageGenerator, ImageSize
from SimplerLLM.utils.custom_verbose import verbose_print


class GoogleImageGenerator(ImageGenerator):
    """
    Google Gemini Image Generation wrapper.
    Provides a unified interface for Google Gemini image generation models.
    """

    def __init__(self, provider, model_name, api_key, verbose=False):
        """
        Initialize Google Gemini Image Generator instance.

        Args:
            provider: ImageProvider.GOOGLE_GEMINI
            model_name: Model to use (default: gemini-2.5-flash-image-preview)
            api_key: Google API key (uses GEMINI_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        super().__init__(provider, model_name, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def generate_image(
        self,
        prompt: str,
        size=ImageSize.SQUARE,
        resolution: str = "1K",
        model: str = None,
        output_format: str = "bytes",
        output_path: str = None,
        full_response: bool = False,
        reference_images=None,
        **kwargs,
    ):
        """
        Generate image from text prompt using Google Gemini.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or aspect ratio string (e.g., "16:9", "1:1", "21:9")
                  Default: ImageSize.SQUARE
            resolution: Image resolution - "1K", "2K", or "4K" (default: "1K")
            model: Model to use (None = use instance default)
                   Default: gemini-2.5-flash-image
            output_format: How to return the image (default: "bytes")
                          Options: "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
                          Note: Gemini doesn't provide URLs, so "url" will save
                                to temp file and return path
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
                          Includes text description in revised_prompt field
            reference_images: Optional list of reference images for character consistency.
                             Each item can be:
                             - str: File path to image
                             - bytes: Raw image data
                             - dict: {'data': bytes, 'mime_type': str}
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.GOOGLE_GEMINI)
            >>> # Get bytes
            >>> image_bytes = img_gen.generate_image("A serene landscape")
            >>> # Save to file
            >>> path = img_gen.generate_image("A city", output_format="file", output_path="city.png")
            >>> # Get full response with text description
            >>> response = img_gen.generate_image("Abstract art", full_response=True)
            >>> print(response.revised_prompt)  # Text description from Gemini
            >>> # Generate with reference images for character consistency
            >>> image_bytes = img_gen.generate_image(
            ...     "The same character in a different pose",
            ...     reference_images=["character_ref.jpg"]
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
                f"Generating image - Model: {model_to_use}, Aspect Ratio: {aspect_ratio}, Resolution: {resolution}",
                "info"
            )
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine actual output format for API
        # Gemini returns base64, so we handle "url" specially
        if output_format == "url":
            # Gemini doesn't provide URLs, so we'll save to a temp file
            if self.verbose:
                verbose_print("Note: Google Gemini doesn't provide URLs. Saving to file instead.", "warning")
            api_output_path = output_path or "output/gemini_image.png"
            output_format = "file"
        else:
            api_output_path = output_path if output_format == "file" else None

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": "png",  # File format (metadata only)
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "reference_images": reference_images,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = google_image.generate_image(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image generated successfully in {response.process_time:.2f}s", "info")
                    if response.revised_prompt:
                        desc_preview = response.revised_prompt[:100] + "..." if len(response.revised_prompt) > 100 else response.revised_prompt
                        verbose_print(f"Text description: {desc_preview}", "debug")
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
        resolution: str = "1K",
        model: str = None,
        output_format: str = "bytes",
        output_path: str = None,
        full_response: bool = False,
        reference_images=None,
        **kwargs,
    ):
        """
        Asynchronously generate image from text prompt using Google Gemini.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or aspect ratio string (e.g., "16:9", "1:1", "21:9")
                  Default: ImageSize.SQUARE
            resolution: Image resolution - "1K", "2K", or "4K" (default: "1K")
            model: Model to use (None = use instance default)
            output_format: How to return the image (default: "bytes")
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            reference_images: Optional list of reference images for character consistency.
                             Each item can be:
                             - str: File path to image
                             - bytes: Raw image data
                             - dict: {'data': bytes, 'mime_type': str}
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.GOOGLE_GEMINI)
            >>> image_bytes = await img_gen.generate_image_async("A serene landscape")
            >>> # With reference images
            >>> image_bytes = await img_gen.generate_image_async(
            ...     "Character in new scene",
            ...     reference_images=["ref.jpg"]
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
                f"Generating image (async) - Model: {model_to_use}, Aspect Ratio: {aspect_ratio}, Resolution: {resolution}",
                "info"
            )
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine actual output format for API
        if output_format == "url":
            if self.verbose:
                verbose_print("Note: Google Gemini doesn't provide URLs. Saving to file instead.", "warning")
            api_output_path = output_path or "output/gemini_image.png"
            output_format = "file"
        else:
            api_output_path = output_path if output_format == "file" else None

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": "png",
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "reference_images": reference_images,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await google_image.generate_image_async(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image generated successfully in {response.process_time:.2f}s", "info")
                    if response.revised_prompt:
                        desc_preview = response.revised_prompt[:100] + "..." if len(response.revised_prompt) > 100 else response.revised_prompt
                        verbose_print(f"Text description: {desc_preview}", "debug")
                else:
                    verbose_print("Image generated successfully", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating image: {str(e)}", "error")
            raise

    def edit_image(
        self,
        image_source,
        edit_prompt: str,
        size=ImageSize.SQUARE,
        resolution: str = "1K",
        model: str = None,
        output_format: str = "bytes",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Edit an existing image using text instructions with Google Gemini.

        Args:
            image_source: Source image to edit. Can be:
                         - str: File path to image
                         - bytes: Raw image data
                         - dict: {'data': bytes, 'mime_type': str}
            edit_prompt: Text instructions for how to edit the image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or aspect ratio string (e.g., "16:9", "1:1", "21:9")
                  Default: ImageSize.SQUARE
            resolution: Image resolution - "1K", "2K", or "4K" (default: "1K")
            model: Model to use (None = use instance default)
                   Default: gemini-2.5-flash-image
            output_format: How to return the image (default: "bytes")
                          Options: "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="bytes": edited image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.GOOGLE_GEMINI)
            >>> # Edit image and get bytes
            >>> edited_bytes = img_gen.edit_image(
            ...     image_source="original.jpg",
            ...     edit_prompt="Change the background to a sunset sky"
            ... )
            >>> # Edit and save to file
            >>> path = img_gen.edit_image(
            ...     image_source="photo.jpg",
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

        # Map size to aspect ratio
        aspect_ratio = self._map_size_to_aspect_ratio(size)

        if self.verbose:
            verbose_print(
                f"Editing image - Model: {model_to_use}, Aspect Ratio: {aspect_ratio}, Resolution: {resolution}",
                "info"
            )
            verbose_print(f"Edit prompt: {edit_prompt[:100]}...", "debug") if len(edit_prompt) > 100 else verbose_print(f"Edit prompt: {edit_prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine actual output format for API
        if output_format == "url":
            if self.verbose:
                verbose_print("Note: Google Gemini doesn't provide URLs. Saving to file instead.", "warning")
            api_output_path = output_path or "output/gemini_edited_image.png"
            output_format = "file"
        else:
            api_output_path = output_path if output_format == "file" else None

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "model_name": model_to_use,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": "png",  # File format (metadata only)
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = google_image.edit_image(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image edited successfully in {response.process_time:.2f}s", "info")
                    if response.revised_prompt:
                        desc_preview = response.revised_prompt[:100] + "..." if len(response.revised_prompt) > 100 else response.revised_prompt
                        verbose_print(f"Text description: {desc_preview}", "debug")
                else:
                    verbose_print("Image edited successfully", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error editing image: {str(e)}", "error")
            raise

    async def edit_image_async(
        self,
        image_source,
        edit_prompt: str,
        size=ImageSize.SQUARE,
        resolution: str = "1K",
        model: str = None,
        output_format: str = "bytes",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Asynchronously edit an existing image using text instructions with Google Gemini.

        Args:
            image_source: Source image to edit. Can be:
                         - str: File path to image
                         - bytes: Raw image data
                         - dict: {'data': bytes, 'mime_type': str}
            edit_prompt: Text instructions for how to edit the image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or aspect ratio string (e.g., "16:9", "1:1", "21:9")
                  Default: ImageSize.SQUARE
            resolution: Image resolution - "1K", "2K", or "4K" (default: "1K")
            model: Model to use (None = use instance default)
            output_format: How to return the image (default: "bytes")
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="bytes": edited image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.GOOGLE_GEMINI)
            >>> edited_bytes = await img_gen.edit_image_async(
            ...     image_source="original.jpg",
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

        # Map size to aspect ratio
        aspect_ratio = self._map_size_to_aspect_ratio(size)

        if self.verbose:
            verbose_print(
                f"Editing image (async) - Model: {model_to_use}, Aspect Ratio: {aspect_ratio}, Resolution: {resolution}",
                "info"
            )
            verbose_print(f"Edit prompt: {edit_prompt[:100]}...", "debug") if len(edit_prompt) > 100 else verbose_print(f"Edit prompt: {edit_prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine actual output format for API
        if output_format == "url":
            if self.verbose:
                verbose_print("Note: Google Gemini doesn't provide URLs. Saving to file instead.", "warning")
            api_output_path = output_path or "output/gemini_edited_image.png"
            output_format = "file"
        else:
            api_output_path = output_path if output_format == "file" else None

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "model_name": model_to_use,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": "png",
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await google_image.edit_image_async(**params)

            if self.verbose:
                if full_response:
                    verbose_print(f"Image edited successfully in {response.process_time:.2f}s", "info")
                    if response.revised_prompt:
                        desc_preview = response.revised_prompt[:100] + "..." if len(response.revised_prompt) > 100 else response.revised_prompt
                        verbose_print(f"Text description: {desc_preview}", "debug")
                else:
                    verbose_print("Image edited successfully", "info")

            return response

        except Exception as e:
            if self.verbose:
                verbose_print(f"Error editing image: {str(e)}", "error")
            raise

    def _map_size_to_aspect_ratio(self, size):
        """
        Map ImageSize enum or custom size to Google Gemini aspect ratio format.

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
            # If it's dimensions like "1024x1024", convert to closest aspect ratio
            if "x" in size.lower():
                # For Gemini, default to 1:1 for custom dimensions
                return "1:1"
            return size

        # If size is not an ImageSize enum, try to convert it
        if not isinstance(size, ImageSize):
            try:
                size = ImageSize(size)
            except ValueError:
                # If conversion fails, return default
                return "1:1"

        # Map ImageSize enum to Gemini's aspect ratios
        size_map = {
            ImageSize.SQUARE: "1:1",
            ImageSize.HORIZONTAL: "16:9",
            ImageSize.VERTICAL: "9:16",
            ImageSize.PORTRAIT_3_4: "3:4",
        }

        return size_map.get(size, "1:1")
