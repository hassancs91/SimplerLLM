import SimplerLLM.language.llm_providers.anthropic_llm as anthropic_llm
import os
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print
from SimplerLLM.tools.image_helpers import prepare_vision_content_anthropic

class AnthropicLLM(LLM):
    def __init__(self, provider, model_name, temperature, top_p, api_key, verbose=False):
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    
    def append_messages(self, messages: list):
        model_messages = []
        if messages:
            model_messages.extend(messages)
        return model_messages

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
        prompt_caching: bool = False,
        cached_input: str = "",
        json_mode=False,
        images: list = None,
    ):
        """
        Generate a response using the Anthropic LLM.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to the instance's model_name.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt to set the context. Defaults to "You are a helpful AI Assistant".
            temperature (float, optional): Controls randomness in output. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 300.
            top_p (float, optional): Controls diversity of output. Defaults to 1.0.
            full_response (bool, optional): If True, returns the full API response. If False, returns only the generated text. Defaults to False.
            prompt_caching (bool, optional): Whether to use prompt caching. Defaults to False.
            cached_input (str, optional): The cached input to use if prompt_caching is True. Defaults to "".
            json_mode (bool, optional): If True, enables JSON mode. Defaults to False.
            images (list, optional): List of image sources (URLs or file paths) for vision-capable models. Defaults to None.

        Returns:
            The generated response from the Anthropic LLM.

        Raises:
            ValueError: If both prompt and messages are provided, or if neither is provided.
        """
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            if self.verbose:
                verbose_print("Error: Both prompt and messages provided", "error")
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            if self.verbose:
                verbose_print("Error: Neither prompt nor messages provided", "error")
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
                verbose_print(f"User prompt: {prompt}", "debug")
                if images:
                    verbose_print(f"Images provided: {len(images)} image(s)", "debug")

            # Handle vision content if images are provided
            if images:
                user_content = prepare_vision_content_anthropic(prompt, images)
            else:
                user_content = prompt

            model_messages = [
                {"role": "user", "content": user_content},
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
                if images:
                    verbose_print("Warning: Images parameter is ignored when using messages format", "warning")
            model_messages = self.append_messages(messages)

        params.update(
            {
                "api_key": self.api_key,
                "system_prompt": system_prompt,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "prompt_caching": prompt_caching,
                "cached_input": cached_input,
                "json_mode" : json_mode
            }
        )

        if self.verbose:
            verbose_print("Generating response with Anthropic...", "info")
            
        try:
            response = anthropic_llm.generate_response(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise

    async def generate_response_async(
        self,
        model_name=None,
        prompt=None,
        messages: list = None,
        system_prompt="You are a helpful AI Assistant",
        temperature=0.7,
        max_tokens=300,
        top_p=1.0,
        full_response=False,
        prompt_caching: bool = False,
        cached_input: str = "",
        json_mode=False,
        images: list = None,
    ):
        """
        Asynchronously generate a response from the Anthropic LLM.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to the instance's model_name.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt to set the context. Defaults to "You are a helpful AI Assistant".
            temperature (float, optional): Controls randomness in output. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 300.
            top_p (float, optional): Controls diversity of output. Defaults to 1.0.
            full_response (bool, optional): If True, returns the full API response. If False, returns only the generated text. Defaults to False.
            prompt_caching (bool, optional): Whether to use prompt caching. Defaults to False.
            cached_input (str, optional): The cached input to use. Defaults to "".
            json_mode (bool, optional): If True, enables JSON mode. Defaults to False.
            images (list, optional): List of image sources (URLs or file paths) for vision-capable models. Defaults to None.

        Returns:
            The asynchronously generated response from the Anthropic LLM.

        Raises:
            ValueError: If both prompt and messages are provided, or if neither is provided.
        """
        params = self.prepare_params(model_name, temperature, top_p)

        # Validate inputs
        if prompt and messages:
            if self.verbose:
                verbose_print("Error: Both prompt and messages provided", "error")
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        if not prompt and not messages:
            if self.verbose:
                verbose_print("Error: Neither prompt nor messages provided", "error")
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Prepare messages based on input type
        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
                verbose_print(f"User prompt: {prompt}", "debug")
                if images:
                    verbose_print(f"Images provided: {len(images)} image(s)", "debug")

            # Handle vision content if images are provided
            if images:
                user_content = prepare_vision_content_anthropic(prompt, images)
            else:
                user_content = prompt

            model_messages = [
                {"role": "user", "content": user_content},
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
                if images:
                    verbose_print("Warning: Images parameter is ignored when using messages format", "warning")
            model_messages = self.append_messages(messages)

        params.update(
            {
                "api_key": self.api_key,
                "system_prompt": system_prompt,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "prompt_caching": prompt_caching,
                "cached_input": cached_input,
                "json_mode" : json_mode
            }
        )

        if self.verbose:
            verbose_print("Generating response with Anthropic (async)...", "info")
            
        try:
            response = await anthropic_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
