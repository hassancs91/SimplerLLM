from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from SimplerLLM.agents.models import ToolCall, AgentThought, AgentAction, AgentResponse
from SimplerLLM.tools.json_helpers import generate_json_example_from_pydantic, example_value_for_type
import json

def print_section(title):
    """Helper function to print a section title with formatting"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

print_section("TEST 1: EXAMINING AGENT ACTION MODEL STRUCTURE")
print("\nAgentAction model fields and types:")
for field_name, field in AgentAction.model_fields.items():
    print(f"  {field_name}: {field.annotation} (required: {field.is_required()})")
    if hasattr(field, 'description') and field.description:
        print(f"    Description: {field.description}")

print_section("TEST 2: GENERATING EXAMPLE VALUES FOR EACH FIELD TYPE")
print("\nGenerating example value for 'thought' field (str):")
thought_example = example_value_for_type(str)
print(f"  Example: {thought_example}")

print("\nGenerating example value for 'action' field (Optional[ToolCall]):")
try:
    action_example = example_value_for_type(Optional[ToolCall])
    print(f"  Example: {action_example}")
    print(f"  Type: {type(action_example)}")
    if action_example:
        print(f"  Tool name: {action_example.tool_name}")
        print(f"  Parameters: {action_example.parameters}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("\nGenerating example value for 'response' field (Optional[str]):")
try:
    response_example = example_value_for_type(Optional[str])
    print(f"  Example: {response_example}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print_section("TEST 3: MANUALLY CREATING AND SERIALIZING AGENT ACTION MODELS")
print("\nTest 3.1: Creating AgentAction with tool call:")
try:
    tool_call = ToolCall(
        tool_name="example_tool",
        parameters={"param1": "value1", "param2": "value2"}
    )
    
    agent_action = AgentAction(
        thought="This is a test thought",
        action=tool_call,
        response=None
    )
    
    print("  Successfully created AgentAction with tool call")
    print(f"  Thought: {agent_action.thought}")
    print(f"  Action tool name: {agent_action.action.tool_name}")
    print(f"  Action parameters: {agent_action.action.parameters}")
    print(f"  Response: {agent_action.response}")
    
    # Convert to dict
    agent_action_dict = agent_action.model_dump()
    print("\n  Model dict:")
    print(f"  {agent_action_dict}")
    
    # Convert to JSON
    agent_action_json = json.dumps(agent_action_dict)
    print("\n  JSON string:")
    print(f"  {agent_action_json}")
    
    # Parse JSON back to dict
    parsed_dict = json.loads(agent_action_json)
    print("\n  Parsed back to dict:")
    print(f"  {parsed_dict}")
    
    # Create model from parsed dict
    recreated_model = AgentAction(**parsed_dict)
    print("\n  Recreated model:")
    print(f"  Thought: {recreated_model.thought}")
    print(f"  Action tool name: {recreated_model.action.tool_name}")
    print(f"  Action parameters: {recreated_model.action.parameters}")
    print(f"  Response: {recreated_model.response}")
    
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("\nTest 3.2: Creating AgentAction with response (no tool call):")
try:
    agent_action = AgentAction(
        thought="This is a test thought with response",
        action=None,
        response="This is a test response"
    )
    
    print("  Successfully created AgentAction with response")
    print(f"  Thought: {agent_action.thought}")
    print(f"  Action: {agent_action.action}")
    print(f"  Response: {agent_action.response}")
    
    # Convert to dict
    agent_action_dict = agent_action.model_dump()
    print("\n  Model dict:")
    print(f"  {agent_action_dict}")
    
    # Convert to JSON
    agent_action_json = json.dumps(agent_action_dict)
    print("\n  JSON string:")
    print(f"  {agent_action_json}")
    
    # Parse JSON back to dict
    parsed_dict = json.loads(agent_action_json)
    print("\n  Parsed back to dict:")
    print(f"  {parsed_dict}")
    
    # Create model from parsed dict
    recreated_model = AgentAction(**parsed_dict)
    print("\n  Recreated model:")
    print(f"  Thought: {recreated_model.thought}")
    print(f"  Action: {recreated_model.action}")
    print(f"  Response: {recreated_model.response}")
    
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print_section("TEST 4: USING GENERATE_JSON_EXAMPLE_FROM_PYDANTIC WITH AGENT ACTION MODEL")
print("\nGenerating JSON example for AgentAction model:")
try:
    # Get the JSON example
    json_example = generate_json_example_from_pydantic(AgentAction)
    print(f"  JSON example string: {json_example}")
    print(f"  Length: {len(json_example)}")
    
    # Parse the JSON
    parsed_json = json.loads(json_example)
    print("\n  Parsed JSON:")
    print(json.dumps(parsed_json, indent=2))
    
    # Create a model from the parsed JSON
    model_from_json = AgentAction(**parsed_json)
    print("\n  Created model from JSON:")
    print(f"  Thought: {model_from_json.thought}")
    if model_from_json.action:
        print(f"  Action tool name: {model_from_json.action.tool_name}")
        print(f"  Action parameters: {model_from_json.action.parameters}")
    else:
        print("  Action: None")
    print(f"  Response: {model_from_json.response}")
    
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print_section("TEST 5: DEBUGGING THE EXAMPLE_VALUE_FOR_TYPE FUNCTION WITH NESTED MODELS")
print("\nDebugging example_value_for_type with ToolCall:")
try:
    # Get the type hints for ToolCall
    print("  ToolCall fields:")
    for field_name, field in ToolCall.model_fields.items():
        print(f"    {field_name}: {field.annotation} (required: {field.is_required()})")
    
    # Generate example value for ToolCall
    tool_call_example = example_value_for_type(ToolCall)
    print("\n  Generated example value for ToolCall:")
    print(f"    Type: {type(tool_call_example)}")
    print(f"    Tool name: {tool_call_example.tool_name}")
    print(f"    Parameters: {tool_call_example.parameters}")
    
    # Convert to dict and JSON
    tool_call_dict = tool_call_example.model_dump()
    print("\n  ToolCall as dict:")
    print(f"    {tool_call_dict}")
    
    tool_call_json = json.dumps(tool_call_dict)
    print("\n  ToolCall as JSON:")
    print(f"    {tool_call_json}")
    
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nDebugging example_value_for_type with Optional[ToolCall]:")
try:
    # Generate example value for Optional[ToolCall]
    optional_tool_call_example = example_value_for_type(Optional[ToolCall])
    print("\n  Generated example value for Optional[ToolCall]:")
    print(f"    Type: {type(optional_tool_call_example)}")
    if optional_tool_call_example:
        print(f"    Tool name: {optional_tool_call_example.tool_name}")
        print(f"    Parameters: {optional_tool_call_example.parameters}")
    else:
        print("    Value is None")
    
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
