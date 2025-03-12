from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, get_type_hints
from SimplerLLM.tools.json_helpers import generate_json_example_from_pydantic, example_value_for_type

# Model with Field descriptions (similar to the one causing issues)
class ModelWithFieldDesc(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(..., description="Parameters to pass to the tool")
    optional_field: Optional[str] = Field(None, description="An optional field")

# Similar model without Field descriptions
class ModelWithoutFieldDesc(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    optional_field: Optional[str] = None

# Debug type hints and example values
print("Type hints for ModelWithFieldDesc:")
for field_name, field_type in get_type_hints(ModelWithFieldDesc).items():
    print(f"  {field_name}: {field_type}")
    try:
        example = example_value_for_type(field_type)
        print(f"    Example value: {example}")
    except Exception as e:
        print(f"    Error generating example: {type(e).__name__}: {e}")

print("\nType hints for ModelWithoutFieldDesc:")
for field_name, field_type in get_type_hints(ModelWithoutFieldDesc).items():
    print(f"  {field_name}: {field_type}")
    try:
        example = example_value_for_type(field_type)
        print(f"    Example value: {example}")
    except Exception as e:
        print(f"    Error generating example: {type(e).__name__}: {e}")

# Create a modified version of generate_json_example_from_pydantic to debug the issue
def debug_generate_json_example(model_class):
    example_data = {}
    for field_name, field_type in get_type_hints(model_class).items():
        example_data[field_name] = example_value_for_type(field_type)
    
    print("Raw example_data dictionary:")
    import json
    print(json.dumps(example_data, indent=2))
    
    print("Parameters dictionary:")
    print(json.dumps(example_data['parameters'], indent=2))
    
    model_instance = model_class(**example_data)
    model_dict = model_instance.model_dump()
    print("Model dict after model_dump:")
    print(json.dumps(model_dict, indent=2))
    
    json_str = json.dumps(model_dict)
    print("JSON after json.dumps:")
    print(json_str)
    
    # Write to file to verify
    with open(f"test_{model_class.__name__}.json", "w") as f:
        f.write(json_str)
    print(f"Wrote JSON to test_{model_class.__name__}.json")
    
    # Read back from file
    with open(f"test_{model_class.__name__}.json", "r") as f:
        file_content = f.read()
    print(f"Read from file: {file_content}")
    
    return json_str

# Test both models
print("\nTesting model with Field descriptions:")
try:
    json_with_desc = debug_generate_json_example(ModelWithFieldDesc)
    
    # Parse the JSON string to verify it's valid JSON
    import json
    parsed_json = json.loads(json_with_desc)
    print(f"Parsed JSON: {json.dumps(parsed_json, indent=2)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting model without Field descriptions:")
try:
    json_without_desc = debug_generate_json_example(ModelWithoutFieldDesc)
    
    # Parse the JSON string to verify it's valid JSON
    import json
    parsed_json = json.loads(json_without_desc)
    print(f"Parsed JSON: {json.dumps(parsed_json, indent=2)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
