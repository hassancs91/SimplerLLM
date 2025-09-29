from SimplerLLM.language.llm import LLMProvider, LLM

llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-5-mini")

response = llm_instance.generate_response(
    prompt="Give me a sentence of 5 words",
)
print("hi")
print(response)