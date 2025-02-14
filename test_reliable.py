from pydantic import BaseModel
from typing import List
from SimplerLLM.language.llm import LLM, LLMProvider, ReliableLLM

class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    instructions: List[str]

def test_reliable_llm():
    # Create primary and secondary LLMs with verbose logging
    primary_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",  # Intentionally wrong model name to test fallback
        verbose=True
    )
    
    secondary_llm = LLM.create(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        verbose=True
    )
    
    # Create reliable LLM with fallback and verbose logging
    reliable_llm = ReliableLLM(primary_llm, secondary_llm, verbose=True)


    response, provider = reliable_llm.generate_response(
            prompt="What is the meaning of life?",
            max_tokens=100,return_provider=True
        )
    
    print("Response:", response)
    print("provider:", provider.name)
    
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

def test_reliable_pydantic_model():
    # Create primary and secondary LLMs with verbose logging
    primary_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",  # Intentionally wrong model name to test fallback
        verbose=True
    )
    
    secondary_llm = LLM.create(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        verbose=True
    )
    
    # Create reliable LLM with fallback and verbose logging
    reliable_llm = ReliableLLM(primary_llm, secondary_llm, verbose=True)
    
    # Test the pydantic model generation with fallback
    try:
        recipe = reliable_llm.generate_pydantic_model(
            model_class=Recipe,
            prompt="Generate a recipe for chocolate chip cookies",
            max_tokens=1000
        )
        print("\nGenerated Recipe:")
        print(f"Name: {recipe.name}")
        print("\nIngredients:")
        for ingredient in recipe.ingredients:
            print(f"- {ingredient}")
        print("\nInstructions:")
        for i, step in enumerate(recipe.instructions, 1):
            print(f"{i}. {step}")
    except Exception as e:
        print("Both providers failed:", e)

if __name__ == "__main__":
    print("\nTesting without verbose logging:")
    test_reliable_llm()
    print("\nTesting Pydantic Model Generation without verbose logging:")
    test_reliable_pydantic_model()
    
    
