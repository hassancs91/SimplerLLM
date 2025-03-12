from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from SimplerLLM.language.llm import LLM, LLMProvider

# Model with Field descriptions (similar to the one causing issues)
class ToolCallModel(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(..., description="Parameters to pass to the tool")
    optional_field: Optional[str] = Field(None, description="An optional field")

# Create a simple LLM instance for testing
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Test prompt
prompt = "Generate a tool call to search for information about Python programming."

print("Testing generate_pydantic_json_model with a model that has Field descriptions:")
try:
    # Set full_response=True to get the full response object
    result = generate_pydantic_json_model(
        model_class=ToolCallModel,
        prompt=prompt,
        llm_instance=llm,
        max_retries=1,
        full_response=True
    )
    
    # Check if the result is successful
    if hasattr(result, 'model_object'):
        print("Success! Generated model object:")
        print(f"Tool name: {result.model_object.tool_name}")
        print(f"Parameters: {result.model_object.parameters}")
        print(f"Optional field: {result.model_object.optional_field}")
    else:
        print(f"Error: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
