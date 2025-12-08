import SimplerLLM.language.llm_providers.perplexity_llm as perplexity_llm
import os
from ..base import LLM
from SimplerLLM.utils.custom_verbose import verbose_print


class PerplexityLLM(LLM):
    def __init__(self, provider, model_name, temperature, top_p, api_key, verbose=False):
        super().__init__(provider, model_name, temperature, top_p, api_key, verbose=verbose)
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY", "")

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
        json_mode: bool = False,
        search_domain_filter: list = None,
        search_recency_filter: str = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        images: list = None,
        detail: str = "auto",
        web_search: bool = False,
    ):
        """
        Generate a response using the Perplexity LLM.

        Perplexity has built-in web search by default - every request performs a web search.
        Citations are returned in web_sources when full_response=True.

        Args:
            model_name (str, optional): The model to use (e.g., 'sonar', 'sonar-pro'). Defaults to instance's model_name.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt. Defaults to "You are a helpful AI Assistant".
            temperature (float, optional): Controls randomness (0-2). Defaults to 0.7.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 300.
            top_p (float, optional): Nucleus sampling threshold. Defaults to 1.0.
            full_response (bool, optional): If True, returns LLMFullResponse with web_sources. Defaults to False.
            json_mode (bool, optional): If True, enables JSON mode. Defaults to False.
            search_domain_filter (list, optional): Domains to include/exclude (prefix with "-" to exclude).
            search_recency_filter (str, optional): Filter by time ("day", "week", "month").
            return_images (bool, optional): Include images in results. Defaults to False.
            return_related_questions (bool, optional): Return related queries. Defaults to False.

        Returns:
            str or LLMFullResponse: Generated text or full response with web sources.

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

            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            # Prepend system message if not already present
            model_messages = self.append_messages(messages)
            if not any(msg.get("role") == "system" for msg in model_messages):
                model_messages.insert(0, {"role": "system", "content": system_prompt})

        params.update(
            {
                "api_key": self.api_key,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "json_mode": json_mode,
                "search_domain_filter": search_domain_filter,
                "search_recency_filter": search_recency_filter,
                "return_images": return_images,
                "return_related_questions": return_related_questions,
            }
        )

        if self.verbose:
            verbose_print(f"Generating response with Perplexity using {params['model_name']}...", "info")

        try:
            response = perplexity_llm.generate_response(**params)
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
        json_mode: bool = False,
        search_domain_filter: list = None,
        search_recency_filter: str = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        images: list = None,
        detail: str = "auto",
        web_search: bool = False,
    ):
        """
        Asynchronously generate a response using the Perplexity LLM.

        Perplexity has built-in web search by default - every request performs a web search.
        Citations are returned in web_sources when full_response=True.

        Args:
            model_name (str, optional): The model to use (e.g., 'sonar', 'sonar-pro'). Defaults to instance's model_name.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt. Defaults to "You are a helpful AI Assistant".
            temperature (float, optional): Controls randomness (0-2). Defaults to 0.7.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 300.
            top_p (float, optional): Nucleus sampling threshold. Defaults to 1.0.
            full_response (bool, optional): If True, returns LLMFullResponse with web_sources. Defaults to False.
            json_mode (bool, optional): If True, enables JSON mode. Defaults to False.
            search_domain_filter (list, optional): Domains to include/exclude (prefix with "-" to exclude).
            search_recency_filter (str, optional): Filter by time ("day", "week", "month").
            return_images (bool, optional): Include images in results. Defaults to False.
            return_related_questions (bool, optional): Return related queries. Defaults to False.

        Returns:
            str or LLMFullResponse: Generated text or full response with web sources.

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

            model_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            if self.verbose:
                verbose_print("Preparing chat messages", "debug")
            # Prepend system message if not already present
            model_messages = self.append_messages(messages)
            if not any(msg.get("role") == "system" for msg in model_messages):
                model_messages.insert(0, {"role": "system", "content": system_prompt})

        params.update(
            {
                "api_key": self.api_key,
                "messages": model_messages,
                "max_tokens": max_tokens,
                "full_response": full_response,
                "json_mode": json_mode,
                "search_domain_filter": search_domain_filter,
                "search_recency_filter": search_recency_filter,
                "return_images": return_images,
                "return_related_questions": return_related_questions,
            }
        )

        if self.verbose:
            verbose_print(f"Generating response with Perplexity (async) using {params['model_name']}...", "info")

        try:
            response = await perplexity_llm.generate_response_async(**params)
            if self.verbose:
                verbose_print("Response received successfully", "info")
            return response
        except Exception as e:
            if self.verbose:
                verbose_print(f"Error generating response: {str(e)}", "error")
            raise
