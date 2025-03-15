from SimplerLLM.language.llm_addons import generate_pydantic_json_model_reliable
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from pydantic import BaseModel
from typing import List, Optional

# Define a simple Pydantic model for product recommendations
class ProductFeature(BaseModel):
    name: str
    description: str
    importance: int  # 1-10 scale

class ProductRecommendation(BaseModel):
    product_name: str
    price_range: str
    target_audience: str
    key_features: List[ProductFeature]
    pros: List[str]
    cons: List[str]
    overall_rating: int  # 1-10 scale

# Create the LLM instances and ReliableLLM
llm_1 = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o-mini", verbose=True)
llm_2 = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o", verbose=True)
reliable_llm = ReliableLLM(primary_llm=llm_1, secondary_llm=llm_2)

# Create a prompt for product recommendation
product_recommendation_prompt = """
As a product expert, please provide a detailed recommendation for a {product_type}.
Include key features, pros and cons, and an overall rating.
"""

# Format the prompt with a specific product type
formatted_prompt = product_recommendation_prompt.format(product_type="smartphone")

# Generate the Pydantic model using the reliable LLM
response = generate_pydantic_json_model_reliable(
    model_class=ProductRecommendation,
    prompt=formatted_prompt,
    reliable_llm=reliable_llm,
    max_retries=2,
    temperature=0.2,
    top_p=0.9,
    system_prompt="You are a helpful product recommendation assistant",
    full_response=True,
)

# Access token counts and model object directly
input_tokens = response.input_token_count
output_tokens = response.output_token_count
model_object = response.model_object
provider_used = response.provider
model_used = response.model_name

# Print the results
print(f"Provider used: {provider_used}")
print(f"Model used: {model_used}")
print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Product recommendation: {model_object.product_name}")
print(f"Overall rating: {model_object.overall_rating}/10")
print(f"Key features:")
for feature in model_object.key_features:
    print(f"  - {feature.name}: {feature.description} (Importance: {feature.importance}/10)")
print(f"Pros: {', '.join(model_object.pros)}")
print(f"Cons: {', '.join(model_object.cons)}")