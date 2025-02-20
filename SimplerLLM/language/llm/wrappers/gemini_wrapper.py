import SimplerLLM.language.llm_providers.gemini_llm as gemini_llm
import os
import base64
import requests
import json
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print

class GeminiLLM(LLM):
    def __init__(self, provider, model_name, temperature, top_p, api_key, verbose=False):
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def convert_messages_template(self, messages):
        return [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages]

    def append_messages(self, system_prompt, messages):
        model_messages = [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "model", "parts": [{"text": "ok, confirmed."}]},
        ]
        messages = self.convert_messages_template(messages)
        model_messages.extend(messages)
        return model_messages

    def create_cache(
        self,
        cached_input: str,
        ttl: int = 600,
    ) -> str:
        encoded_content = base64.b64encode(cached_input.encode()).decode()
        cache_url = f"https://generativelanguage.googleapis.com/v1beta/cachedContents?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        cache_payload = {
            "model": f"models/gemini-1.5-flash-001",
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "text/plain",
                                "data": encoded_content
                            }
                        }
                    ],
                    "role": "user"
                }
            ],
            "ttl": f"{ttl}s"
        }
        
        response = requests.post(cache_url, headers=headers, data=json.dumps(cache_payload))
        response.raise_for_status()

        return response.json()["name"]

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
        cache_id: str = None,
        json_mode=False,
    ):
        """
        Generate a response using the Gemini model.

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
            cache_id (str, optional): The cache ID to use if prompt_caching is True.

        Returns:
            str or dict: The generated response text, or the full API response if full_response is True.

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

        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
                verbose_print(f"System prompt: {system_prompt}", "debug")
                verbose_print(f"User prompt: {prompt}", "debug")
            model_messages = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "ok, confirmed."}]},
                {"role": "user", "parts": [{"text": prompt}]}
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = self.append_messages(system_prompt, messages)

        params.update(
            {
                "api_key": self.api_key,
                "system_prompt": system_prompt,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "prompt_caching": prompt_caching,
                "cache_id": cache_id,
                "json_mode" : json_mode     
            }
        )

        if self.verbose:
            verbose_print("Generating response with Gemini...", "info")
            if prompt_caching:
                verbose_print(f"Using cache ID: {cache_id}", "debug")
            
        try:
            response = gemini_llm.generate_response(**params)
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
        prompt_caching: bool = False,
        cache_id: str = None,
        json_mode=False,
    ):
        """
        Asynchronously generate a response from the Gemini model.

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
            cache_id (str, optional): The cache ID to use if prompt_caching is True.

        Returns:
            str or dict: The generated response text, or the full API response if full_response is True.

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

        if prompt:
            if self.verbose:
                verbose_print("Preparing single prompt message", "debug")
                verbose_print(f"System prompt: {system_prompt}", "debug")
                verbose_print(f"User prompt: {prompt}", "debug")
            model_messages = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "ok, confirmed."}]},
                {"role": "user", "parts": [{"text": prompt}]}
            ]

        if messages:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            model_messages = self.append_messages(system_prompt, messages)

        params.update(
            {
                "api_key": self.api_key,
                "system_prompt": system_prompt,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "prompt_caching": prompt_caching,
                "cache_id": cache_id,
                "json_mode" : json_mode     
            }
        )

        if self.verbose:
            verbose_print("Generating response with Gemini (async)...", "info")
            if prompt_caching:
                verbose_print(f"Using cache ID: {cache_id}", "debug")
            
        try:
            response = await gemini_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
