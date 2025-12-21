from SimplerLLM.language.llm import LLM, LLMProvider, ReliableLLM

# Set up two providers
primary = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o-mini")
backup = LLM.create(provider=LLMProvider.GEMINI, model_name="gemini-2.0-flash")

# Create a reliable LLM that auto-switches if primary fails
llm = ReliableLLM(primary_llm=primary, secondary_llm=backup)

# Use it normally - failover happens automatically!
response = llm.generate_response(prompt="What is AI?")
print(response)
