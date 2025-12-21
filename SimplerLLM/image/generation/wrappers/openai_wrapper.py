import SimplerLLM.image.generation.providers.openai_image as openai_image
import os
from ..base import ImageGenerator, ImageSize
from SimplerLLM.utils.custom_verbose import verbose_print


class OpenAIImageGenerator(ImageGenerator):
    """
    OpenAI Image Generation wrapper.
    Provides a unified interface for OpenAI image generation models (DALL-E and GPT Image).
    """

    def __init__(self, provider, model_name, api_key, verbose=False):
        """
        Initialize OpenAI Image Generator instance.

        Args:
            provider: ImageProvider.OPENAI_DALL_E
            model_name: Model to use ("dall-e-3", "dall-e-2", "gpt-image-1", "gpt-image-1.5")
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        super().__init__(provider, model_name, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def _is_gpt_image_model(self, model_name=None):
        """Check if the model is a GPT image model (not DALL-E)."""
        model = model_name or self.model_name
        return model.startswith("gpt-image")

    def generate_image(
        self,
        prompt: str,
        size=ImageSize.SQUARE,
        quality: str = "standard",
        style: str = "vivid",
        model: str = None,
        output_format: str = "url",
        image_output_format: str = "png",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Generate image from text prompt using OpenAI Image API.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or string dimension (e.g., "1024x1024")
                  Default: ImageSize.SQUARE
            quality: Quality setting (default: "standard")
                    - DALL-E 3: "standard", "hd"
                    - GPT Image: "low", "medium", "high", "auto"
            style: Style setting for DALL-E 3 only (default: "vivid")
                   Options: "vivid", "natural"
            model: Model to use (None = use instance default)
                   Options: "dall-e-3", "dall-e-2", "gpt-image-1", "gpt-image-1.5"
            output_format: How to return the image (default: "url")
                          Options: "url" (DALL-E only - returns URL string),
                                   "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
                          Note: GPT Image models always return bytes (no URL support)
            image_output_format: For GPT Image models only - image file format
                                Options: "png", "jpeg", "webp" (default: "png")
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": URL string (DALL-E only)
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.OPENAI_DALL_E)
            >>> # Get URL (DALL-E only)
            >>> url = img_gen.generate_image("A serene landscape")
            >>> # Save to file
            >>> path = img_gen.generate_image("A city", output_format="file", output_path="city.png")
            >>> # GPT Image model
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.OPENAI_DALL_E, model_name="gpt-image-1")
            >>> response = img_gen.generate_image("Abstract art", quality="high", full_response=True)
        """
        # Validate input
        if not prompt:
            if self.verbose:
                verbose_print("Error: Prompt parameter is required", "error")
            raise ValueError("Prompt parameter is required for image generation")

        # Get model (use parameter or instance default)
        model_to_use = model if model is not None else self.model_name
        is_gpt_image = self._is_gpt_image_model(model_to_use)

        # GPT Image models don't support URL output - force bytes
        if is_gpt_image and output_format == "url":
            if self.verbose:
                verbose_print("Note: GPT Image models don't support URL output, returning bytes instead", "info")
            output_format = "bytes"

        # Map size to actual dimensions
        size_dimension = self._map_size_to_dimensions(size)

        if self.verbose:
            if is_gpt_image:
                verbose_print(
                    f"Generating image - Model: {model_to_use}, Size: {size_dimension}, Quality: {quality}",
                    "info"
                )
            else:
                verbose_print(
                    f"Generating image - Model: {model_to_use}, Size: {size_dimension}, Quality: {quality}, Style: {style}",
                    "info"
                )
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "size": size_dimension,
            "quality": quality,
            "style": style,
            "n": 1,  # Always generate 1 image
            "output_path": output_path if output_format == "file" else None,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add model-specific parameters
        if is_gpt_image:
            params["output_format"] = image_output_format
        else:
            # Determine response_format for DALL-E API call
            api_response_format = "b64_json" if output_format == "bytes" else "url"
            params["response_format"] = api_response_format

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
        image_output_format: str = "png",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Asynchronously generate image from text prompt using OpenAI Image API.

        Args:
            prompt: Text description of the desired image (required)
            size: Image size - can be ImageSize enum (SQUARE, HORIZONTAL, VERTICAL)
                  or string dimension (e.g., "1024x1024")
                  Default: ImageSize.SQUARE
            quality: Quality setting (default: "standard")
                    - DALL-E 3: "standard", "hd"
                    - GPT Image: "low", "medium", "high", "auto"
            style: Style setting for DALL-E 3 only (default: "vivid")
                   Options: "vivid", "natural"
            model: Model to use (None = use instance default)
                   Options: "dall-e-3", "dall-e-2", "gpt-image-1", "gpt-image-1.5"
            output_format: How to return the image (default: "url")
                          Options: "url" (DALL-E only - returns URL string),
                                   "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
                          Note: GPT Image models always return bytes (no URL support)
            image_output_format: For GPT Image models only - image file format
                                Options: "png", "jpeg", "webp" (default: "png")
            output_path: File path to save image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": URL string (DALL-E only)
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
        is_gpt_image = self._is_gpt_image_model(model_to_use)

        # GPT Image models don't support URL output - force bytes
        if is_gpt_image and output_format == "url":
            if self.verbose:
                verbose_print("Note: GPT Image models don't support URL output, returning bytes instead", "info")
            output_format = "bytes"

        # Map size to actual dimensions
        size_dimension = self._map_size_to_dimensions(size)

        if self.verbose:
            if is_gpt_image:
                verbose_print(
                    f"Generating image (async) - Model: {model_to_use}, Size: {size_dimension}, Quality: {quality}",
                    "info"
                )
            else:
                verbose_print(
                    f"Generating image (async) - Model: {model_to_use}, Size: {size_dimension}, Quality: {quality}, Style: {style}",
                    "info"
                )
            verbose_print(f"Prompt: {prompt[:100]}...", "debug") if len(prompt) > 100 else verbose_print(f"Prompt: {prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Build parameters for provider call
        params = {
            "prompt": prompt,
            "model_name": model_to_use,
            "size": size_dimension,
            "quality": quality,
            "style": style,
            "n": 1,  # Always generate 1 image
            "output_path": output_path if output_format == "file" else None,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add model-specific parameters
        if is_gpt_image:
            params["output_format"] = image_output_format
        else:
            # Determine response_format for DALL-E API call
            api_response_format = "b64_json" if output_format == "bytes" else "url"
            params["response_format"] = api_response_format

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

    def edit_image(
        self,
        image_source,
        edit_prompt: str,
        mask_source=None,
        size=ImageSize.SQUARE,
        model: str = None,
        output_format: str = "url",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Edit an existing image using OpenAI's image edit API.

        Note: Only DALL-E 2 supports image editing. DALL-E 3 and GPT Image models do NOT support editing.

        Args:
            image_source: Source image to edit. Can be:
                         - str: File path to PNG image (must be square, < 4MB)
                         - bytes: Raw PNG image data
            edit_prompt: Text description of the desired edit (required)
            mask_source: Optional mask image. Transparent areas indicate where to edit.
                        Can be str (file path) or bytes. Must be same size as image.
            size: Output size - ImageSize.SQUARE maps to "1024x1024"
                  Options: "256x256", "512x512", "1024x1024"
            model: Model to use (only "dall-e-2" supported for editing)
            output_format: How to return the image (default: "url")
                          Options: "url" (returns URL string),
                                   "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": URL string
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Raises:
            ValueError: If model doesn't support editing (only dall-e-2 supported)

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.OPENAI_DALL_E, model_name="dall-e-2")
            >>> edited = img_gen.edit_image(
            ...     image_source="original.png",
            ...     edit_prompt="Add a red hat to the person",
            ...     mask_source="mask.png",  # transparent where to edit
            ...     output_format="file",
            ...     output_path="edited.png"
            ... )
        """
        # Validate input
        if not edit_prompt:
            if self.verbose:
                verbose_print("Error: edit_prompt parameter is required", "error")
            raise ValueError("edit_prompt parameter is required for image editing")

        # Get model - default to dall-e-2 for editing
        model_to_use = model if model is not None else "dall-e-2"

        # Validate model supports editing
        if model_to_use != "dall-e-2":
            raise ValueError(f"Image editing is only supported by dall-e-2. Model '{model_to_use}' does not support editing.")

        # Map size to actual dimensions
        size_dimension = self._map_size_to_dimensions(size)

        if self.verbose:
            verbose_print(f"Editing image - Model: {model_to_use}, Size: {size_dimension}", "info")
            verbose_print(f"Edit prompt: {edit_prompt[:100]}...", "debug") if len(edit_prompt) > 100 else verbose_print(f"Edit prompt: {edit_prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine response_format for API call
        api_response_format = "b64_json" if output_format == "bytes" else "url"

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "mask_source": mask_source,
            "model_name": model_to_use,
            "size": size_dimension,
            "n": 1,
            "response_format": api_response_format,
            "output_path": output_path if output_format == "file" else None,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = openai_image.edit_image(**params)

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
        image_source,
        edit_prompt: str,
        mask_source=None,
        size=ImageSize.SQUARE,
        model: str = None,
        output_format: str = "url",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Asynchronously edit an existing image using OpenAI's image edit API.

        Note: Only DALL-E 2 supports image editing. DALL-E 3 and GPT Image models do NOT support editing.

        Args:
            image_source: Source image to edit. Can be:
                         - str: File path to PNG image (must be square, < 4MB)
                         - bytes: Raw PNG image data
            edit_prompt: Text description of the desired edit (required)
            mask_source: Optional mask image. Transparent areas indicate where to edit.
                        Can be str (file path) or bytes. Must be same size as image.
            size: Output size - ImageSize.SQUARE maps to "1024x1024"
                  Options: "256x256", "512x512", "1024x1024"
            model: Model to use (only "dall-e-2" supported for editing)
            output_format: How to return the image (default: "url")
                          Options: "url" (returns URL string),
                                   "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": URL string
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Raises:
            ValueError: If model doesn't support editing (only dall-e-2 supported)
        """
        # Validate input
        if not edit_prompt:
            if self.verbose:
                verbose_print("Error: edit_prompt parameter is required", "error")
            raise ValueError("edit_prompt parameter is required for image editing")

        # Get model - default to dall-e-2 for editing
        model_to_use = model if model is not None else "dall-e-2"

        # Validate model supports editing
        if model_to_use != "dall-e-2":
            raise ValueError(f"Image editing is only supported by dall-e-2. Model '{model_to_use}' does not support editing.")

        # Map size to actual dimensions
        size_dimension = self._map_size_to_dimensions(size)

        if self.verbose:
            verbose_print(f"Editing image (async) - Model: {model_to_use}, Size: {size_dimension}", "info")
            verbose_print(f"Edit prompt: {edit_prompt[:100]}...", "debug") if len(edit_prompt) > 100 else verbose_print(f"Edit prompt: {edit_prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine response_format for API call
        api_response_format = "b64_json" if output_format == "bytes" else "url"

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "mask_source": mask_source,
            "model_name": model_to_use,
            "size": size_dimension,
            "n": 1,
            "response_format": api_response_format,
            "output_path": output_path if output_format == "file" else None,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await openai_image.edit_image_async(**params)

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
