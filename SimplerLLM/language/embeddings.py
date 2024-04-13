# import SimplerLLM.language.llm_providers.openai_llm as openai_llm
# import SimplerLLM.language.llm_providers.gemini_llm as gemini_llm
# import SimplerLLM.language.llm_providers.anthropic_llm as anthropic_llm
# from enum import Enum


# class EmbeddingsProvider(Enum):
#     OPENAI = 1
#     GEMINI = 2
#     ANTHROPIC = 3


# class LLM:
#     def __init__(
#         self, provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small"
#     ):
#         self.provider = provider
#         self.model_name = model_name

#     @staticmethod
#     def create(
#         provider=None,
#         model_name=None,
#     ):
#         if provider == EmbeddingsProvider.OPENAI:
#             return OpenAILLM(provider, model_name)
#         if provider == EmbeddingsProvider.GEMINI:
#             return GeminiLLM(provider, model_name)
#         if provider == EmbeddingsProvider.ANTHROPIC:
#             return AnthropicLLM(provider, model_name)
#         else:
#             return None

#     def set_model(self, provider):
#         if not isinstance(provider, EmbeddingsProvider):
#             raise ValueError("Provider must be an instance of EmbeddingsProvider Enum")
#         self.provider = provider


# class OpenAILLM(LLM):
#     def __init__(self, model, model_name):
#         super().__init__(model, model_name)

#     def generate_embeddings(
#         self,
#         user_input,
#         model_name=None,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return openai_llm.generate_text(
#             user_input=user_input,
#             model_name=model_name,
#         )


# class GeminiLLM(LLM):
#     def __init__(self, model, model_name, temperature, top_p):
#         super().__init__(model, model_name, temperature, top_p)

#     def generate_text(
#         self,
#         user_prompt,
#         system_prompt=None,
#         model_name=None,
#         temperature=None,
#         top_p=None,
#         max_tokens=2024,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return gemini_llm.generate_text(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             model_name=model_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#         )

#     def generate_full_response(
#         self,
#         user_prompt,
#         system_prompt,
#         model_name=None,
#         temperature=None,
#         top_p=None,
#         max_tokens=2024,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return gemini_llm.generate_full_response(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             model_name=model_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#         )

#     async def generate_text_async(
#         self,
#         user_prompt,
#         system_prompt=None,
#         model_name=None,
#         temperature=None,
#         top_p=None,
#         max_tokens=2024,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return await gemini_llm.generate_text_async(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             model_name=model_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#         )

#     async def generate_full_response_async(
#         self,
#         user_prompt,
#         system_prompt,
#         model_name=None,
#         temperature=None,
#         top_p=None,
#         max_tokens=2024,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return await gemini_llm.generate_full_response_async(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             model_name=model_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#         )


# class AnthropicLLM(LLM):
#     def __init__(self, model, model_name, temperature, top_p):
#         super().__init__(model, model_name, temperature, top_p)

#     def generate_text(
#         self,
#         user_prompt,
#         system_prompt=None,
#         model_name=None,
#         temperature=None,
#         top_p=None,
#         max_tokens=2024,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return anthropic_llm.generate_text(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             model_name=model_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#         )

#     def generate_full_response(
#         self,
#         user_prompt,
#         system_prompt,
#         model_name=None,
#         temperature=None,
#         top_p=None,
#         max_tokens=2024,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return anthropic_llm.generate_full_response(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             model_name=model_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#         )

#     async def generate_text_async(
#         self,
#         user_prompt,
#         system_prompt=None,
#         model_name=None,
#         temperature=None,
#         top_p=None,
#         max_tokens=2024,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return await anthropic_llm.generate_text_async(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             model_name=model_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#         )

#     async def generate_full_response_async(
#         self,
#         user_prompt,
#         system_prompt,
#         model_name=None,
#         temperature=None,
#         top_p=None,
#         max_tokens=2024,
#     ):
#         # Use instance values as defaults if not provided
#         model_name = model_name if model_name is not None else self.model_name
#         temperature = temperature if temperature is not None else self.temperature
#         top_p = top_p if top_p is not None else self.top_p

#         return await anthropic_llm.generate_full_response_async(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#             model_name=model_name,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#         )
