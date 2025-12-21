from SimplerLLM.language.llm import LLM, LLMProvider

# THIS IS THE ONLY LINE YOU CHANGE TO SWITCH PROVIDERS
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o-mini")

# Your app code stays exactly the same
response = llm.generate_response(prompt="What is AI?")
print(response)
