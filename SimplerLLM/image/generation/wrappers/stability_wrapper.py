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
            ImageSize.PORTRAIT_2_3: "2:3",
            ImageSize.PORTRAIT_4_5: "4:5",
            ImageSize.LANDSCAPE_3_2: "3:2",
            ImageSize.LANDSCAPE_4_3: "4:3",
            ImageSize.LANDSCAPE_5_4: "5:4",
            ImageSize.ULTRAWIDE: "21:9",
        }

        return size_map.get(size, "1:1")

    def edit_image(
        self,
        image_source,
        edit_prompt: str,
        search_prompt: str = None,
        mask=None,
        style_preset: str = None,
        negative_prompt: str = None,
        seed: int = 0,
        grow_mask: int = 5,
        output_format: str = "bytes",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Edit an existing image using text instructions with Stability AI.

        Uses Search and Replace endpoint by default (no mask needed, works with JPEG).
        When a mask is provided, uses Inpaint endpoint instead.

        Args:
            image_source: Source image to edit. Can be:
                         - str: File path to image
                         - bytes: Raw image data
                         - str: Base64 encoded image
            edit_prompt: What to replace with (e.g., "a watercolor painting",
                        "a sunset sky", "a vintage photograph")
            search_prompt: What to find and replace in the image
                          (e.g., "the subject", "the background", "the sky").
                          Defaults to "the subject" if not provided.
            mask: Optional mask image for Inpaint mode (black=keep, white=edit).
                  When provided, uses Inpaint instead of Search and Replace.
                  Can be: str (path), bytes, or None.
            style_preset: Style to guide the generation
                         Options: 3d-model, analog-film, anime, cinematic, comic-book,
                                 digital-art, enhance, fantasy-art, isometric, line-art,
                                 low-poly, modeling-compound, neon-punk, origami,
                                 photographic, pixel-art, tile-texture
            negative_prompt: Keywords of what you do NOT want in the result
            seed: Randomness seed (0 for random, 1-4294967294 for reproducible)
            grow_mask: Expands mask edges (0-20, default 5). Only used with Inpaint.
            output_format: How to return the image (default: "bytes")
                          Options: "bytes", "file"
            output_path: File path to save edited image (required if output_format="file")
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="bytes": edited image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.STABILITY_AI)
            >>> # Transform subject into watercolor (works with JPEG!)
            >>> edited = img_gen.edit_image(
            ...     image_source="photo.jpg",
            ...     edit_prompt="a watercolor painting",
            ...     search_prompt="the person"
            ... )
            >>> # Replace background with sunset
            >>> edited = img_gen.edit_image(
            ...     image_source="portrait.jpg",
            ...     edit_prompt="a beautiful sunset sky",
            ...     search_prompt="the background"
            ... )
            >>> # Use mask for precise control (Inpaint mode)
            >>> edited = img_gen.edit_image(
            ...     image_source="photo.png",
            ...     edit_prompt="a red sports car",
            ...     mask="car_mask.png"
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

        if self.verbose:
            edit_mode = "Inpaint (with mask)" if mask else "Search and Replace"
            verbose_print(f"Editing image with Stability AI {edit_mode}", "info")
            if not mask:
                sp = search_prompt or "the subject"
                verbose_print(f"Search: '{sp}' -> Replace with: '{edit_prompt[:50]}...'", "debug")
            if style_preset:
                verbose_print(f"Style Preset: {style_preset}", "debug")
            if negative_prompt:
                verbose_print(f"Negative Prompt: {negative_prompt[:50]}...", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine actual output format for API
        if output_format == "url":
            if self.verbose:
                verbose_print("Note: Stability AI doesn't provide URLs. Saving to file instead.", "warning")
            api_output_path = output_path or "output/stability_edited_image.png"
            output_format = "file"
        else:
            api_output_path = output_path if output_format == "file" else None

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "search_prompt": search_prompt,
            "mask": mask,
            "style_preset": style_preset,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "grow_mask": grow_mask,
            "output_format": "png",  # File format
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = stability_image.edit_image(**params)

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
        search_prompt: str = None,
        mask=None,
        style_preset: str = None,
        negative_prompt: str = None,
        seed: int = 0,
        grow_mask: int = 5,
        output_format: str = "bytes",
        output_path: str = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Asynchronously edit an existing image using text instructions with Stability AI.

        Uses Search and Replace endpoint by default (no mask needed).
        When a mask is provided, uses Inpaint endpoint instead.

        Args:
            image_source: Source image to edit. Can be:
                         - str: File path to image
                         - bytes: Raw image data
                         - str: Base64 encoded image
            edit_prompt: What to replace with
            search_prompt: What to find and replace in the image
            mask: Optional mask image for Inpaint mode
            style_preset: Style to guide the generation
            negative_prompt: Keywords of what you do NOT want in the result
            seed: Randomness seed (0 for random)
            grow_mask: Expands mask edges (0-20). Only used with Inpaint.
            output_format: How to return the image (default: "bytes")
            output_path: File path to save edited image
            full_response: If True, returns ImageGenerationResponse with metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            If full_response=True: ImageGenerationResponse object with metadata
            If output_format="bytes": edited image bytes
            If output_format="file": file path string

        Example:
            >>> img_gen = ImageGenerator.create(provider=ImageProvider.STABILITY_AI)
            >>> edited = await img_gen.edit_image_async(
            ...     image_source="photo.jpg",
            ...     edit_prompt="a sunset sky",
            ...     search_prompt="the background"
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

        if self.verbose:
            edit_mode = "Inpaint (with mask)" if mask else "Search and Replace"
            verbose_print(f"Editing image (async) with Stability AI {edit_mode}", "info")
            if not mask:
                sp = search_prompt or "the subject"
                verbose_print(f"Search: '{sp}' -> Replace with: '{edit_prompt[:50]}...'", "debug")
            if style_preset:
                verbose_print(f"Style Preset: {style_preset}", "debug")
            if output_path:
                verbose_print(f"Output will be saved to: {output_path}", "debug")

        # Determine actual output format for API
        if output_format == "url":
            if self.verbose:
                verbose_print("Note: Stability AI doesn't provide URLs. Saving to file instead.", "warning")
            api_output_path = output_path or "output/stability_edited_image.png"
            output_format = "file"
        else:
            api_output_path = output_path if output_format == "file" else None

        # Build parameters for provider call
        params = {
            "image_source": image_source,
            "edit_prompt": edit_prompt,
            "search_prompt": search_prompt,
            "mask": mask,
            "style_preset": style_preset,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "grow_mask": grow_mask,
            "output_format": "png",
            "output_path": api_output_path,
            "full_response": full_response,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }

        # Add any additional kwargs
        params.update(kwargs)

        try:
            response = await stability_image.edit_image_async(**params)

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
