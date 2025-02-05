from typing import Type
from pydantic import BaseModel
from .base import LLM

class ReliableLLM:
    def __init__(self, primary_llm: LLM, secondary_llm: LLM):
        """
        Initialize ReliableLLM with primary and secondary LLM providers.
        
        Args:
            primary_llm (LLM): The primary LLM provider to use first
            secondary_llm (LLM): The secondary LLM provider to use as fallback
        """
        self.primary_llm = primary_llm
        self.secondary_llm = secondary_llm
        self._validate_providers()

    def _validate_providers(self):
        """
        Validate both providers during initialization.
        Sets internal flags for which providers are valid.
        """
        self.primary_valid = True
        self.secondary_valid = True

        # Test primary provider
        try:
            response = self.primary_llm.generate_response(
                prompt="test",
                max_tokens=1
            )
            if response is None:
                self.primary_valid = False
                print("Primary provider returned None response")
        except Exception as e:
            self.primary_valid = False
            print(f"Primary provider validation failed: {str(e)}")

        # Test secondary provider
        try:
            response = self.secondary_llm.generate_response(
                prompt="test",
                max_tokens=1
            )
            if response is None:
                self.secondary_valid = False
                print("Secondary provider returned None response")
        except Exception as e:
            self.secondary_valid = False
            print(f"Secondary provider validation failed: {str(e)}")

        if not self.primary_valid and not self.secondary_valid:
            raise ValueError("Both providers have invalid configurations")

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
        return_provider: bool = False,
        json_mode=False,
    ):
        """
        Generate a response using the primary LLM, falling back to secondary if primary fails.
        
        Args:
            model_name (str, optional): The name of the model to use.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt to set the context.
            temperature (float, optional): Controls randomness in output.
            max_tokens (int, optional): The maximum number of tokens to generate.
            top_p (float, optional): Controls diversity of output.
            full_response (bool, optional): If True, returns the full API response.
            return_provider (bool, optional): If True, returns a tuple of (response, provider) where provider is the LLMProvider that generated the response.
            
        Returns:
            Union[str, dict, Tuple[Union[str, dict], LLMProvider]]: 
                - If return_provider is False: The generated response from either primary or secondary LLM
                - If return_provider is True: A tuple of (response, provider) where provider is the LLMProvider that was used
            
        Raises:
            Exception: If both primary and secondary LLMs fail
        """
        if self.primary_valid:
            try:
                response = self.primary_llm.generate_response(
                    model_name=model_name,
                    prompt=prompt,
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    full_response=full_response,
                    json_mode=json_mode,
                )
                return (response, self.primary_llm.provider) if return_provider else response
            except Exception as e:
                if self.secondary_valid:
                    response = self.secondary_llm.generate_response(
                        model_name=model_name,
                        prompt=prompt,
                        messages=messages,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        full_response=full_response,
                        json_mode=json_mode,
                    )
                    return (response, self.secondary_llm.provider) if return_provider else response
                raise ValueError("Both providers failed")
        elif self.secondary_valid:
            response = self.secondary_llm.generate_response(
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                full_response=full_response,
                json_mode=json_mode,
            )
            return (response, self.secondary_llm.provider) if return_provider else response
        raise ValueError("No valid providers available")

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
        return_provider: bool = False,
        json_mode: bool = False,
    ):
        """
        Asynchronously generate a response using the primary LLM, falling back to secondary if primary fails.
        
        Args:
            model_name (str, optional): The name of the model to use.
            prompt (str, optional): A single prompt string to generate a response for.
            messages (list, optional): A list of message dictionaries for chat-based interactions.
            system_prompt (str, optional): The system prompt to set the context.
            temperature (float, optional): Controls randomness in output.
            max_tokens (int, optional): The maximum number of tokens to generate.
            top_p (float, optional): Controls diversity of output.
            full_response (bool, optional): If True, returns the full API response.
            return_provider (bool, optional): If True, returns a tuple of (response, provider) where provider is the LLMProvider that generated the response.
            
        Returns:
            Union[str, dict, Tuple[Union[str, dict], LLMProvider]]: 
                - If return_provider is False: The generated response from either primary or secondary LLM
                - If return_provider is True: A tuple of (response, provider) where provider is the LLMProvider that was used
            
        Raises:
            Exception: If both primary and secondary LLMs fail
        """
        if self.primary_valid:
            try:
                response = await self.primary_llm.generate_response_async(
                    model_name=model_name,
                    prompt=prompt,
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    full_response=full_response,
                    json_mode=json_mode,
                )
                return (response, self.primary_llm.provider) if return_provider else response
            except Exception as e:
                if self.secondary_valid:
                    response = await self.secondary_llm.generate_response_async(
                        model_name=model_name,
                        prompt=prompt,
                        messages=messages,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        full_response=full_response,
                        json_mode=json_mode,
                    )
                    return (response, self.secondary_llm.provider) if return_provider else response
                raise ValueError("Both providers failed")
        elif self.secondary_valid:
            response = await self.secondary_llm.generate_response_async(
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                full_response=full_response,
                json_mode=json_mode,
            )
            return (response, self.secondary_llm.provider) if return_provider else response
        raise ValueError("No valid providers available")
