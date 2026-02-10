"""
Hugging Face Transformers local model wrapper for SimplerLLM.

This wrapper provides a high-level interface for running local Hugging Face models.
Requires optional dependencies: pip install simplerllm[transformers]
"""

import SimplerLLM.language.llm_providers.hf_local_llm as hf_local_llm
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print


class HuggingFaceLocalLLM(LLM):
    """
    LLM wrapper for local Hugging Face Transformers models.

    This provider runs models locally using the transformers library.
    No API key required - models are downloaded from HuggingFace Hub
    or loaded from local paths.

    Environment Variables:
        HF_DEVICE: Device to use ('cuda', 'cpu', 'mps', 'auto'). Default: 'auto'
        HF_TORCH_DTYPE: Data type ('float16', 'bfloat16', 'float32', 'auto'). Default: 'auto'
        HF_LOAD_IN_4BIT: Enable 4-bit quantization. Default: 'false'
        HF_LOAD_IN_8BIT: Enable 8-bit quantization. Default: 'false'
        HF_TRUST_REMOTE_CODE: Trust remote code for custom models. Default: 'false'

    Example:
        >>> from SimplerLLM.language import LLM, LLMProvider
        >>> llm = LLM.create(
        ...     provider=LLMProvider.HUGGING_FACE_LOCAL,
        ...     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ... )
        >>> response = llm.generate_response(prompt="Hello, how are you?")
    """

    def __init__(self, provider, model_name, temperature, top_p, verbose=False):
        # No API key for local models
        super().__init__(provider, model_name, temperature, top_p, None, verbose=verbose)

    def append_messages(self, system_prompt: str, messages: list):
        """Prepend system prompt to messages list."""
        model_messages = [{"role": "system", "content": system_prompt}]
        if messages:
            model_messages.extend(messages)
        return model_messages

    def unload_model(self):
        """
        Unload the current model from memory to free GPU/RAM.

        Use this when you're done with a model and want to free resources
        before loading a different model.
        """
        hf_local_llm.unload_model(self.model_name)
        if self.verbose:
            verbose_print(f"Model '{self.model_name}' unloaded from memory", "info")

    def generate_response(
        self,
        model_name: str = None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        json_mode: bool = False,
        images: list = None,
        detail: str = "auto",
        web_search: bool = False,
        **kwargs,
    ):
        """
        Generate a response using a local Hugging Face model.

        Args:
            model_name: Override the default model. Can be a HuggingFace model ID
                       (e.g., "mistralai/Mistral-7B-Instruct-v0.3") or a local path.
            prompt: A single prompt string to generate a response for.
            messages: A list of message dictionaries for chat-based interactions.
            system_prompt: The system prompt to set context. Default: "You are a helpful AI Assistant"
            temperature: Controls randomness (0.0-2.0). Default: 0.7
            max_tokens: Maximum tokens to generate. Default: 300
            top_p: Nucleus sampling parameter. Default: 1.0
            full_response: If True, returns LLMFullResponse with metadata.
            json_mode: If True, instructs model to output JSON (not strictly enforced).
            images: Not supported in v1. Will warn and ignore.
            detail: Not supported. Parameter ignored.
            web_search: Not supported. Parameter ignored.
            **kwargs: Additional parameters (ignored for compatibility).

        Returns:
            str or LLMFullResponse: Generated text, or full response object if full_response=True.

        Raises:
            ValueError: If both prompt and messages provided, or neither provided.
            ImportError: If transformers/torch not installed.
        """
        params = self.prepare_params(model_name, temperature, top_p)
        effective_model = params.get("model_name") or self.model_name

        # Validate inputs
        if prompt and messages:
            if self.verbose:
                verbose_print("Error: Both prompt and messages provided", "error")
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            if self.verbose:
                verbose_print("Error: Neither prompt nor messages provided", "error")
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Warn about unsupported features
        if web_search:
            if self.verbose:
                verbose_print(
                    "Warning: web_search is not supported by HuggingFace Local. "
                    "Parameter will be ignored. Consider using OpenAI or Anthropic for web search.",
                    "warning"
                )

        if images:
            if self.verbose:
                verbose_print(
                    "Warning: Vision/images are not supported by HuggingFace Local in v1. "
                    "Parameter will be ignored. Vision support planned for v2.",
                    "warning"
                )

        # Prepare messages based on input type
        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
                verbose_print(f"System prompt: {system_prompt}", "debug")
                verbose_print(f"User prompt: {prompt[:100]}...", "debug")

            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = self.append_messages(system_prompt, messages)

        params.update({
            "messages": model_messages,
            "max_tokens": max_tokens,
            "full_response": full_response,
            "json_mode": json_mode,
        })

        if self.verbose:
            verbose_print(f"Generating response with HuggingFace Local ({effective_model})...", "info")

        try:
            response = hf_local_llm.generate_response(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except ImportError as e:
            if self.verbose:
                verbose_print(f"Import error: {str(e)}", "error")
            raise
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise

    async def generate_response_async(
        self,
        model_name: str = None,
        prompt: str = None,
        messages: list = None,
        system_prompt: str = "You are a helpful AI Assistant",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0,
        full_response: bool = False,
        json_mode: bool = False,
        images: list = None,
        detail: str = "auto",
        web_search: bool = False,
        **kwargs,
    ):
        """
        Asynchronously generate a response using a local Hugging Face model.

        This wraps the synchronous generation in a thread to avoid blocking
        the event loop during model inference.

        Args:
            model_name: Override the default model.
            prompt: A single prompt string to generate a response for.
            messages: A list of message dictionaries for chat-based interactions.
            system_prompt: The system prompt to set context.
            temperature: Controls randomness (0.0-2.0). Default: 0.7
            max_tokens: Maximum tokens to generate. Default: 300
            top_p: Nucleus sampling parameter. Default: 1.0
            full_response: If True, returns LLMFullResponse with metadata.
            json_mode: If True, instructs model to output JSON.
            images: Not supported in v1.
            detail: Not supported.
            web_search: Not supported.
            **kwargs: Additional parameters (ignored for compatibility).

        Returns:
            str or LLMFullResponse: Generated text, or full response object.

        Raises:
            ValueError: If both prompt and messages provided, or neither provided.
            ImportError: If transformers/torch not installed.
        """
        params = self.prepare_params(model_name, temperature, top_p)
        effective_model = params.get("model_name") or self.model_name

        # Validate inputs
        if prompt and messages:
            if self.verbose:
                verbose_print("Error: Both prompt and messages provided", "error")
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            if self.verbose:
                verbose_print("Error: Neither prompt nor messages provided", "error")
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Warn about unsupported features
        if web_search:
            if self.verbose:
                verbose_print(
                    "Warning: web_search is not supported by HuggingFace Local. "
                    "Parameter will be ignored.",
                    "warning"
                )

        if images:
            if self.verbose:
                verbose_print(
                    "Warning: Vision/images are not supported by HuggingFace Local in v1. "
                    "Parameter will be ignored.",
                    "warning"
                )

        # Prepare messages based on input type
        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = self.append_messages(system_prompt, messages)

        params.update({
            "messages": model_messages,
            "max_tokens": max_tokens,
            "full_response": full_response,
            "json_mode": json_mode,
        })

        if self.verbose:
            verbose_print(f"Generating response with HuggingFace Local (async) ({effective_model})...", "info")

        try:
            response = await hf_local_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except ImportError as e:
            if self.verbose:
                verbose_print(f"Import error: {str(e)}", "error")
            raise
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
