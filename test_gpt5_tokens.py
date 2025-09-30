"""
Test script to demonstrate GPT-5 token handling with different settings
"""

from SimplerLLM.language.llm import LLMProvider, LLM

print("=" * 60)
print("GPT-5 TOKEN HANDLING TEST")
print("=" * 60)

# Test 1: Using default (should use 4000 tokens automatically)
print("\n1. Testing with DEFAULT tokens (will use 4000 for GPT-5):")
print("-" * 50)
llm_default = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-5")
response = llm_default.generate_response(
    prompt="Give me a sentence of 5 words"
)
print(f"Response: {response}")

# Test 2: Explicitly setting to 100 tokens (to test reasoning message)
print("\n2. Testing with EXPLICIT 100 tokens (should show reasoning notice):")
print("-" * 50)
llm_100 = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-5")
response = llm_100.generate_response(
    prompt="Give me a sentence of 5 words",
    max_tokens=100  # Explicitly set to 100
)
print(f"Response: {response}")

# Test 3: Setting to 500 tokens (should work fine)
print("\n3. Testing with 500 tokens (should work normally):")
print("-" * 50)
llm_500 = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-5")
response = llm_500.generate_response(
    prompt="Give me a sentence of 5 words",
    max_tokens=500
)
print(f"Response: {response}")

# Test 4: Complex prompt with default tokens
print("\n4. Testing complex prompt with DEFAULT tokens:")
print("-" * 50)
complex_prompt = """
Explain the concept of recursion in programming in 3 sentences.
Then provide a simple Python example.
"""
response = llm_default.generate_response(prompt=complex_prompt)
print(f"Response: {response}")

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)