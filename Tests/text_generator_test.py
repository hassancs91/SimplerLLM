# test_text_generator.py

import pytest
from SimplerLLM.llms.openai_llm import basic_generation

@pytest.mark.asyncio
async def test_generate_response():
    # Setup: Define the inputs for your function
    user_prompt = "what is a dolphin?"
    model = "gpt-3.5-turbo"
    
    # Call the function you're testing
    response = await basic_generation(user_prompt, model)

    # Assertions: Check the output is as expected
    assert isinstance(response, str), "The response should be a string."
    assert len(response) > 0, "The response should not be empty."
    # Add more assertions as necessary to test different aspects of your function
