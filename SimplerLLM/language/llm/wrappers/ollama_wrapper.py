import SimplerLLM.language.llm_providers.ollama_llm as ollama_llm
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print

class OllamaLLM(LLM):
    def __init__(self, provider, model_name, temperature, top_p, verbose=False):
        super().__init__(provider, model_name, temperature, top_p, None, verbose=verbose)

    def append_messages(self, system_prompt: str, messages: list):
        model_messages = [{"role": "system", "content": system_prompt}]
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
        json_mode=False,
    ):
        """
        Generate a response using the Ollama LLM.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to the instance's model_name.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt to set the context. Defaults to "You are a helpful AI Assistant".
            temperature (float, optional): Controls randomness in output. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 300.
            top_p (float, optional): Controls diversity of output. Defaults to 1.0.
            full_response (bool, optional): If True, returns the full API response. If False, returns only the generated text. Defaults to False.

        Returns:
            str or dict: The generated response as a string, or the full response object if full_response is True.

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
                verbose_print(f"System prompt: {system_prompt}", "debug")
                verbose_print(f"User prompt: {prompt}", "debug")
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = self.append_messages(system_prompt, messages)

        params.update(
            {
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "json_mode" : json_mode     
            }
        )
        
        if self.verbose:
            verbose_print("Generating response with Ollama...", "info")
            
        try:
            response = ollama_llm.generate_response(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
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
        json_mode=False,
    ):
        """
        Asynchronously generate a response using the Ollama LLM.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to the instance's model_name.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt to set the context. Defaults to "You are a helpful AI Assistant".
            temperature (float, optional): Controls randomness in output. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 300.
            top_p (float, optional): Controls diversity of output. Defaults to 1.0.
            full_response (bool, optional): If True, returns the full API response. If False, returns only the generated text. Defaults to False.

        Returns:
            str or dict: The generated response as a string, or the full response object if full_response is True.

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
                verbose_print(f"System prompt: {system_prompt}", "debug")
                verbose_print(f"User prompt: {prompt}", "debug")
            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = self.append_messages(system_prompt, messages)

        params.update(
            {
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "json_mode" : json_mode     
            }
        )
        
        if self.verbose:
            verbose_print("Generating response with Ollama (async)...", "info")
            
        try:
            response = await ollama_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
