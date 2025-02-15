from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import (
    generate_pydantic_json_model,
    generate_pydantic_json_model_reliable
)

# Define a simple Pydantic model
class Person(BaseModel):
    name: str
    age: int
    city: str

def test_regular_llm():
    print("\nTesting Regular LLM:")
    print("-------------------")
    
    # Initialize LLM (using OpenAI as example)
    llm_instance = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",verbose=True
    )
    
    prompt = "Generate information about a person who lives in New York"
    
    # Test without token counts
    print("\nWithout token counts:")
    person = generate_pydantic_json_model(Person, prompt, llm_instance)
    print(f"Person: {person}")
    
    # Test with token counts
    print("\nWith token counts:")
    person, response = generate_pydantic_json_model(
        Person, prompt, llm_instance, full_response=True
    )
    print(f"Person: {person}")
    print(f"Input tokens: {response.input_token_count}")
    print(f"Output tokens: {response.output_token_count}")



def test_reliable_llm():
    print("\nTesting Reliable LLM:")
    print("--------------------")
    
    # Initialize primary and secondary LLMs
    primary_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o"
    )
    secondary_llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022"
    )
    
    # Create ReliableLLM instance
    reliable_llm = ReliableLLM(primary_llm, secondary_llm,verbose=True)
    
    prompt = "Generate information about a person who lives in New York"
    
    # Test without token counts
    print("\nWithout token counts:")
    person, provider = generate_pydantic_json_model_reliable(
        Person, prompt, reliable_llm
    )
    print(f"Person: {person}")
    print(f"Provider used: {provider.name}")
    
    # Test with token counts
    print("\nWith token counts:")
    person, response, provider = generate_pydantic_json_model_reliable(
        Person, prompt, reliable_llm, full_response=True
    )
    print(f"Person: {person}")
    print(f"Provider used: {provider.name}")
    print(f"Input tokens: {response.input_token_count}")
    print(f"Output tokens: {response.output_token_count}")

if __name__ == "__main__":
    # Test regular LLM
    test_regular_llm()
    
    # Test reliable LLM
    #test_reliable_llm()
