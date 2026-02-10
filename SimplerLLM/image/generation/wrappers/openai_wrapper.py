import SimplerLLM.image.generation.providers.openai_image as openai_image
import os
from ..base import ImageGenerator, ImageSize
from SimplerLLM.utils.custom_verbose import verbose_print


class OpenAIImageGenerator(ImageGenerator):
    """
    OpenAI Image Generation wrapper.
    Provides a unified interface for OpenAI image generation models.

    Supported Models:
        - dall-e-3 (default): High-quality, detailed images, 1 image per request
        - dall-e-2: Faster generation, image editing with mask, 1-10 images per request
        - gpt-image-1: GPT Image model with quality levels, editing support
        - gpt-image-1.5: Latest GPT Image model, first 5 reference images preserved with higher fidelity

    Supported Sizes (DALL-E 3):
        1024x1024 (square), 1792x1024 (horizontal), 1024x1792 (vertical)

    Supported Sizes (DALL-E 2):
        256x256, 512x512, 1024x1024

    Supported Sizes (GPT Image):
        1024x1024, 1536x1024, 1024x1536, auto

    Quality Options:
        - DALL-E 3: "standard", "hd"
        - GPT Image: "low", "medium", "high", "auto"

    Style Options (DALL-E 3 only):
        "vivid" (default), "natural"

    Image Editing:
        - dall-e-2: Supports editing with optional mask
        - gpt-image-1, gpt-image-1.5: Supports editing with input_fidelity control

    Reference Images (GPT Image only):
        Up to 16 reference images for compositing and style reference.
        - gpt-image-1: First image preserved with higher fidelity
        - gpt-image-1.5: First 5 images preserved with higher fidelity
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
        reference_images=None,
        input_fidelity: str = "low",
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
            reference_images: Optional list of reference images for GPT Image models.
                             Up to 16 images for compositing and style reference.
                             Each item can be str (file path) or bytes.
                             Note: gpt-image-1 preserves first image with higher fidelity,
                             gpt-image-1.5 preserves first 5 images with higher fidelity.
            input_fidelity: For GPT Image models - "low" (default) or "high".
                           Controls how precisely the model preserves details from input images.
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
            >>> # GPT Image model with reference images
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.OPENAI_DALL_E, model_name="gpt-image-1")
            >>> response = img_gen.generate_image(
            ...     "Create a gift basket with these items",
            ...     reference_images=["item1.jpg", "item2.jpg"],
            ...     input_fidelity="high",
            ...     quality="high",
            ...     full_response=True
            ... )
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
                if reference_images:
                    verbose_print(f"Using {len(reference_images)} reference image(s), input_fidelity={input_fidelity}", "info")
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
            params["reference_images"] = reference_images
            params["input_fidelity"] = input_fidelity
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
        reference_images=None,
        input_fidelity: str = "low",
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
            reference_images: Optional list of reference images for GPT Image models.
                             Up to 16 images for compositing and style reference.
            input_fidelity: For GPT Image models - "low" (default) or "high".
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
                if reference_images:
                    verbose_print(f"Using {len(reference_images)} reference image(s), input_fidelity={input_fidelity}", "info")
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
            params["reference_images"] = reference_images
            params["input_fidelity"] = input_fidelity
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
        image_output_format: str = "png",
        quality: str = "medium",
        output_path: str = None,
        full_response: bool = False,
        input_fidelity: str = "low",
        **kwargs,
    ):
        """
        Edit an existing image using OpenAI's image edit API.

        Supports both DALL-E 2 and GPT Image models for editing.

        Args:
            image_source: Source image to edit. Can be:
                         - str: File path to image
                         - bytes: Raw image data
            edit_prompt: Text description of the desired edit (required)
            mask_source: Optional mask image (DALL-E 2 only). Transparent areas indicate where to edit.
                        Can be str (file path) or bytes. Must be same size as image.
            size: Output size - ImageSize enum or string dimension
                  - DALL-E 2: "256x256", "512x512", "1024x1024"
                  - GPT Image: "1024x1024", "1536x1024", "1024x1536", "auto"
            model: Model to use - "dall-e-2", "gpt-image-1", or "gpt-image-1.5"
                   Default: "dall-e-2"
            output_format: How to return the image (default: "url")
                          Options: "url" (DALL-E 2 only - returns URL string),
                                   "bytes" (returns image bytes),
                                   "file" (saves to file, requires output_path)
                          Note: GPT Image models always return bytes (no URL support)
            image_output_format: For GPT Image models only - "png", "jpeg", or "webp"
            quality: For GPT Image models only - "low", "medium", "high", or "auto"
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            input_fidelity: For GPT Image models - "low" (default) or "high".
                           Controls how precisely the model preserves details from input images.
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": URL string (DALL-E 2 only)
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Raises:
            ValueError: If model doesn't support editing

        Example:
            >>> # DALL-E 2 editing with mask
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.OPENAI_DALL_E, model_name="dall-e-2")
            >>> edited = img_gen.edit_image(
            ...     image_source="original.png",
            ...     edit_prompt="Add a red hat to the person",
            ...     mask_source="mask.png",
            ...     output_format="file",
            ...     output_path="edited.png"
            ... )
            >>> # GPT Image editing
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.OPENAI_DALL_E, model_name="gpt-image-1")
            >>> edited = img_gen.edit_image(
            ...     image_source="photo.jpg",
            ...     edit_prompt="Add a sunset sky in the background",
            ...     quality="high",
            ...     input_fidelity="high",
            ...     full_response=True
            ... )
        """
        # Validate input
        if not edit_prompt:
            if self.verbose:
                verbose_print("Error: edit_prompt parameter is required", "error")
            raise ValueError("edit_prompt parameter is required for image editing")

        # Get model - default to dall-e-2 for editing
        model_to_use = model if model is not None else "dall-e-2"
        is_gpt_image = self._is_gpt_image_model(model_to_use)

        # Validate model supports editing
        if not is_gpt_image and model_to_use != "dall-e-2":
            raise ValueError(f"Image editing is only supported by dall-e-2 and gpt-image models. Model '{model_to_use}' does not support editing.")

        # GPT Image models don't support URL output - force bytes
        if is_gpt_image and output_format == "url":
            if self.verbose:
                verbose_print("Note: GPT Image models don't support URL output, returning bytes instead", "info")
            output_format = "bytes"

        # Map size to actual dimensions
        size_dimension = self._map_size_to_dimensions(size)

        if self.verbose:
            if is_gpt_image:
                verbose_print(f"Editing image - Model: {model_to_use}, Size: {size_dimension}, Quality: {quality}, Input Fidelity: {input_fidelity}", "info")
            else:
                verbose_print(f"Editing image - Model: {model_to_use}, Size: {size_dimension}", "info")
            verbose_print(f"Edit prompt: {edit_prompt[:100]}...", "debug") if len(edit_prompt) > 100 else verbose_print(f"Edit prompt: {edit_prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "model_name": model_to_use,
            "size": size_dimension,
            "n": 1,
            "output_path": output_path if output_format == "file" else None,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add model-specific parameters
        if is_gpt_image:
            params["output_format"] = image_output_format
            params["quality"] = quality
            params["input_fidelity"] = input_fidelity
        else:
            params["mask_source"] = mask_source
            api_response_format = "b64_json" if output_format == "bytes" else "url"
            params["response_format"] = api_response_format

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
        image_output_format: str = "png",
        quality: str = "medium",
        output_path: str = None,
        full_response: bool = False,
        input_fidelity: str = "low",
        **kwargs,
    ):
        """
        Asynchronously edit an existing image using OpenAI's image edit API.

        Supports both DALL-E 2 and GPT Image models for editing.

        Args:
            image_source: Source image to edit. Can be:
                         - str: File path to image
                         - bytes: Raw image data
            edit_prompt: Text description of the desired edit (required)
            mask_source: Optional mask image (DALL-E 2 only). Transparent areas indicate where to edit.
                        Can be str (file path) or bytes. Must be same size as image.
            size: Output size - ImageSize enum or string dimension
                  - DALL-E 2: "256x256", "512x512", "1024x1024"
                  - GPT Image: "1024x1024", "1536x1024", "1024x1536", "auto"
            model: Model to use - "dall-e-2", "gpt-image-1", or "gpt-image-1.5"
            output_format: How to return the image (default: "url")
                          Note: GPT Image models always return bytes (no URL support)
            image_output_format: For GPT Image models only - "png", "jpeg", or "webp"
            quality: For GPT Image models only - "low", "medium", "high", or "auto"
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            input_fidelity: For GPT Image models - "low" (default) or "high".
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="url": URL string (DALL-E 2 only)
            If output_format="bytes": image bytes
            If output_format="file": file path string

        Raises:
            ValueError: If model doesn't support editing
        """
        # Validate input
        if not edit_prompt:
            if self.verbose:
                verbose_print("Error: edit_prompt parameter is required", "error")
            raise ValueError("edit_prompt parameter is required for image editing")

        # Get model - default to dall-e-2 for editing
        model_to_use = model if model is not None else "dall-e-2"
        is_gpt_image = self._is_gpt_image_model(model_to_use)

        # Validate model supports editing
        if not is_gpt_image and model_to_use != "dall-e-2":
            raise ValueError(f"Image editing is only supported by dall-e-2 and gpt-image models. Model '{model_to_use}' does not support editing.")

        # GPT Image models don't support URL output - force bytes
        if is_gpt_image and output_format == "url":
            if self.verbose:
                verbose_print("Note: GPT Image models don't support URL output, returning bytes instead", "info")
            output_format = "bytes"

        # Map size to actual dimensions
        size_dimension = self._map_size_to_dimensions(size)

        if self.verbose:
            if is_gpt_image:
                verbose_print(f"Editing image (async) - Model: {model_to_use}, Size: {size_dimension}, Quality: {quality}, Input Fidelity: {input_fidelity}", "info")
            else:
                verbose_print(f"Editing image (async) - Model: {model_to_use}, Size: {size_dimension}", "info")
            verbose_print(f"Edit prompt: {edit_prompt[:100]}...", "debug") if len(edit_prompt) > 100 else verbose_print(f"Edit prompt: {edit_prompt}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "model_name": model_to_use,
            "size": size_dimension,
            "n": 1,
            "output_path": output_path if output_format == "file" else None,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add model-specific parameters
        if is_gpt_image:
            params["output_format"] = image_output_format
            params["quality"] = quality
            params["input_fidelity"] = input_fidelity
        else:
            params["mask_source"] = mask_source
            api_response_format = "b64_json" if output_format == "bytes" else "url"
            params["response_format"] = api_response_format

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
