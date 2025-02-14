from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create( provider=LLMProvider.OPENAI, model_name="gpt-4o", verbose=True)

response = llm.generate_response(
    prompt="Hello!",
    system_prompt="You are a helpful assistant",
)
