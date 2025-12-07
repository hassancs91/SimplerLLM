"""
Test script to verify the ReliableLLM validation fix for GPT-5
"""
from pydantic import BaseModel, Field
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_reliable


class SimpleResponse(BaseModel):
    answer: str = Field(description="A short answer")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


def test_reliable_with_gpt5():
    print("=" * 50)
    print("Testing ReliableLLM with GPT-5")
    print("=" * 50)

    # Create LLM instances
    primary_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-5"
    )
    secondary_llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-haiku-20241022"
    )

    # Create ReliableLLM - this should now work with the fix
    print("\nCreating ReliableLLM...")
    try:
        reliable_llm = ReliableLLM(primary_llm, secondary_llm, verbose=True,validation_max_tokens=10)
        print("ReliableLLM created successfully!")
    except Exception as e:
        print(f"Failed to create ReliableLLM: {e}")
        return

    # Test with pydantic model
    print("\nTesting generate_pydantic_json_model_reliable...")
    result = generate_pydantic_json_model_reliable(
        model_class=SimpleResponse,
        prompt="What is 2 + 2?",
        reliable_llm=reliable_llm,
        max_retries=2
    )

    if isinstance(result, tuple):
        model, provider, model_name = result
        print(f"\nSUCCESS!")
        print(f"Provider: {provider.name}")
        print(f"Model: {model_name}")
        print(f"Answer: {model.answer}")
        print(f"Confidence: {model.confidence}")
    else:
        print(f"\nFailed: {result}")


if __name__ == "__main__":
    test_reliable_with_gpt5()
