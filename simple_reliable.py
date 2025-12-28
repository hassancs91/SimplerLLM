from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM


primary_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-5"
    )

secondary_llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-4-5-sonnet"
    )


reliable_llm = ReliableLLM(primary_llm,
                           secondary_llm)


response = reliable_llm.generate_response(prompt="")

