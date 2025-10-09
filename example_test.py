from SimplerLLM.language.llm import LLMProvider, LLM

llm_instance = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")

response = llm_instance.generate_response(
    prompt="Give me a sentence of 5 words", max_tokens=500
)

print(response)