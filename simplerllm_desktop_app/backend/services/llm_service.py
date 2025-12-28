"""
LLM Service - Wrapper for SimplerLLM
"""
import os
from typing import Dict, List, Optional, Any
from services.settings_service import SettingsService

# Import SimplerLLM
try:
    from SimplerLLM.language.llm import LLM, LLMProvider
except ImportError:
    # Fallback for when installed as a package
    from simplerllm.language.llm import LLM, LLMProvider


class LLMService:
    """Service for interacting with LLMs through SimplerLLM."""

    # Map provider strings to LLMProvider enum
    PROVIDER_MAP = {
        'openai': LLMProvider.OPENAI,
        'anthropic': LLMProvider.ANTHROPIC,
        'gemini': LLMProvider.GEMINI,
        'cohere': LLMProvider.COHERE,
        'deepseek': LLMProvider.DEEPSEEK,
        'openrouter': LLMProvider.OPENROUTER,
        'ollama': LLMProvider.OLLAMA,
        'perplexity': LLMProvider.PERPLEXITY,
    }

    # Map providers to their API key environment variable names
    API_KEY_ENV_MAP = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'cohere': 'COHERE_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY',
        'perplexity': 'PERPLEXITY_API_KEY',
    }

    def __init__(self):
        self.settings_service = SettingsService()
        self._llm_instances: Dict[str, LLM] = {}

    def _set_api_key_env(self, provider: str) -> bool:
        """Set the API key in environment for the provider."""
        if provider == 'ollama':
            return True  # Ollama doesn't need an API key

        env_var = self.API_KEY_ENV_MAP.get(provider)
        if not env_var:
            return False

        api_key = self.settings_service.get_api_key(provider)
        if not api_key:
            return False

        os.environ[env_var] = api_key
        return True

    def get_llm(self, provider: str, model: str) -> Optional[LLM]:
        """Get or create an LLM instance."""
        cache_key = f"{provider}:{model}"

        if cache_key not in self._llm_instances:
            # Set API key in environment
            if not self._set_api_key_env(provider):
                return None

            llm_provider = self.PROVIDER_MAP.get(provider)
            if not llm_provider:
                return None

            try:
                llm = LLM.create(provider=llm_provider, model_name=model)
                self._llm_instances[cache_key] = llm
            except Exception as e:
                print(f"Error creating LLM: {e}")
                return None

        return self._llm_instances.get(cache_key)

    def generate_response(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            provider: The LLM provider (e.g., 'openai', 'anthropic')
            model: The model name (e.g., 'gpt-4o')
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt for the conversation
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with 'success', 'response', 'usage', and optionally 'error'
        """
        try:
            # Set API key
            if not self._set_api_key_env(provider):
                return {
                    'success': False,
                    'error': f'API key not configured for {provider}'
                }

            llm_provider = self.PROVIDER_MAP.get(provider)
            if not llm_provider:
                return {
                    'success': False,
                    'error': f'Unknown provider: {provider}'
                }

            # Create LLM instance
            llm = LLM.create(provider=llm_provider, model_name=model)

            # Build the prompt from messages
            # SimplerLLM can use either 'prompt' for simple queries
            # or 'messages' for conversation context
            # Use full_response=True to get token usage data
            if len(messages) == 1:
                # Single message, use simple prompt
                response = llm.generate_response(
                    prompt=messages[0]['content'],
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    full_response=True
                )
            else:
                # Multiple messages, use messages parameter
                response = llm.generate_response(
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    full_response=True
                )

            # Extract token usage from full response
            usage = {}
            if hasattr(response, 'input_token_count') and response.input_token_count is not None:
                usage['input_tokens'] = response.input_token_count
            if hasattr(response, 'output_token_count') and response.output_token_count is not None:
                usage['output_tokens'] = response.output_token_count

            # Get the generated text from full response
            generated_text = response.generated_text if hasattr(response, 'generated_text') else str(response)

            return {
                'success': True,
                'response': generated_text,
                'usage': usage
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def validate_api_key(self, provider: str, api_key: str) -> bool:
        """Validate an API key by attempting a minimal request."""
        if provider == 'ollama':
            # Ollama doesn't need validation
            return True

        try:
            # Temporarily set the API key
            env_var = self.API_KEY_ENV_MAP.get(provider)
            if env_var:
                original_key = os.environ.get(env_var)
                os.environ[env_var] = api_key

            llm_provider = self.PROVIDER_MAP.get(provider)
            if not llm_provider:
                return False

            # Get a default model for the provider
            default_models = {
                'openai': 'gpt-3.5-turbo',
                'anthropic': 'claude-3-haiku-20240307',
                'gemini': 'gemini-1.5-flash',
                'cohere': 'command-r',
                'deepseek': 'deepseek-chat',
                'openrouter': 'openai/gpt-3.5-turbo',
                'perplexity': 'llama-3.1-sonar-small-128k-online',
            }

            model = default_models.get(provider, 'gpt-3.5-turbo')

            # Try to create an LLM instance and make a simple request
            llm = LLM.create(provider=llm_provider, model_name=model)
            response = llm.generate_response(
                prompt="Say 'OK' in one word.",
                max_tokens=5
            )

            # Restore original key if it existed
            if env_var and original_key:
                os.environ[env_var] = original_key
            elif env_var:
                os.environ.pop(env_var, None)

            return bool(response)

        except Exception as e:
            print(f"API key validation failed: {e}")
            return False
