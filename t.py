from SimplerLLM.langauge.llm import LLM, LLMProvider

llm_instance = LLM.create(model=LLMProvider.COHERE)

result = llm_instance.gene("hi")
print(result)