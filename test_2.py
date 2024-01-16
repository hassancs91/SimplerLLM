
from SimplerLLM.llms.llm import LLM, LLMModel


llm_instance = LLM(model=LLMModel.OPENAI)
print(llm_instance.generate_text("generate a sentence from 5 words"))