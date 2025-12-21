from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import calculate_text_generation_costs

llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o-mini")

prompt = "What is AI?"
response = llm.generate_response(prompt=prompt)

# Calculate the cost of this call
cost = calculate_text_generation_costs(
    input=prompt,
    response=response,
    cost_per_million_input_tokens=0.15,   # GPT-4o-mini pricing
    cost_per_million_output_tokens=0.60
)

print(f"Tokens used: {cost['input_tokens']} in, {cost['output_tokens']} out")
print(f"Cost: ${cost['total_cost']:.6f}")
