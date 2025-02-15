from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM

#llm = LLM.create( provider=LLMProvider.OPENAI, model_name="gpt-4o", verbose=True)
#llm = LLM.create( provider=LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022", verbose=True)
#llm = LLM.create( provider=LLMProvider.DEEPSEEK, model_name="deepseek-chat", verbose=True)
llm_1 = LLM.create( provider=LLMProvider.GEMINI, model_name="gemini-1.5-pro", verbose=True)
llm_2 = LLM.create( provider=LLMProvider.OLLAMA, model_name="smollm", verbose=True)
reliable_llm = ReliableLLM(primary_llm=llm_1, secondary_llm=llm_2)

response = reliable_llm.generate_response(
    prompt="Hello!",
    system_prompt="You are a helpful assistant",
    full_response=True
)

print(response.llm_provider_response)


input_tokens = response.input_token_count
output_tokens = response.output_token_count

print(input_tokens)
print(output_tokens)
