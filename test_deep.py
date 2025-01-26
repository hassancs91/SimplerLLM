from SimplerLLM.language.llm import LLM, LLMProvider

llm = LLM.create(
    provider=LLMProvider.DEEPSEEK,
    model_name="deepseek-chat",
    temperature=0.7,
    #api_key="sk-97c443da8177407d8e2c2466906429da"
)

response = llm.generate_response(
    prompt="Hello!",
    system_prompt="You are a helpful assistant"
)


print(response)