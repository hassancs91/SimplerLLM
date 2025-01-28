from SimplerLLM.language.llm import LLM, LLMProvider, ReliableLLM

def test_reliable_llm():
    # Create primary and secondary LLMs
    primary_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o"
    )
    
    secondary_llm = LLM.create(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat"
    )
    
    # Create reliable LLM with fallback
    reliable_llm = ReliableLLM(primary_llm, secondary_llm)
    
    # Test the fallback mechanism
    try:
        response, provider = reliable_llm.generate_response(
            prompt="What is the meaning of life?",
            max_tokens=100,return_provider=True
        )
        print("Response:", response)
        print("provider:", provider.name)
    except Exception as e:
        print("Both providers failed:", e)

if __name__ == "__main__":
    test_reliable_llm()
