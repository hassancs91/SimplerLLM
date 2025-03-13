from pydantic import BaseModel
from typing import Dict, Any, Optional
from SimplerLLM.agents.models import ToolCall, AgentThought, AgentAction, AgentResponse
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.tools.json_helpers import generate_json_example_from_pydantic
import json

# Create a simple LLM instance for testing
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

def print_section(title):
    """Helper function to print a section title with formatting"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

# Test 1: Generate JSON examples for each model type
print_section("GENERATING JSON EXAMPLES FOR EACH MODEL TYPE")





print("\n3. AgentAction Model:")
agent_action_json = generate_json_example_from_pydantic(AgentAction)
print(f"JSON example: {agent_action_json}")
# Parse and pretty print
parsed = json.loads(agent_action_json)
print("Parsed and formatted:")
print(json.dumps(parsed, indent=2))

print("\n4. AgentResponse Model:")
agent_response_json = generate_json_example_from_pydantic(AgentResponse)
print(f"JSON example: {agent_response_json}")
# Parse and pretty print
parsed = json.loads(agent_response_json)
print("Parsed and formatted:")
print(json.dumps(parsed, indent=2))

# Test 2: Use generate_pydantic_json_model with each model type
print_section("USING GENERATE_PYDANTIC_JSON_MODEL WITH EACH MODEL TYPE")



print("\n3. AgentAction Model with Tool Call:")
agent_action_tool_prompt = """
Generate an agent action that includes a thought process and a tool call.
The agent should think about finding information on climate change and decide to use a search tool.
"""
try:
    result = generate_pydantic_json_model(
        model_class=AgentAction,
        prompt=agent_action_tool_prompt,
        llm_instance=llm,
        max_retries=3,  # Increase max_retries to give the LLM more chances
        temperature=0.5,  # Lower temperature for more deterministic output
        full_response=True
    )
    
    if hasattr(result, 'model_object'):
        print("Success! Generated model object:")
        print(f"Thought: {result.model_object.thought}")
        if result.model_object.action:
            print(f"Action - Tool name: {result.model_object.action.tool_name}")
            print(f"Action - Parameters: {result.model_object.action.parameters}")
        print(f"Response: {result.model_object.response}")
        # Convert to JSON for display
        print("As JSON:")
        print(json.dumps(result.model_object.model_dump(), indent=2))
    else:
        print(f"Error: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\n4. AgentAction Model with Response:")
agent_action_response_prompt = """
Generate an agent action that includes a thought process and a final response to the user.
The agent should think about a question on the capital of France and provide a direct answer without using tools.
"""
try:
    result = generate_pydantic_json_model(
        model_class=AgentAction,
        prompt=agent_action_response_prompt,
        llm_instance=llm,
        max_retries=3,  # Increase max_retries to give the LLM more chances
        temperature=0.5,  # Lower temperature for more deterministic output
        full_response=True
    )
    
    if hasattr(result, 'model_object'):
        print("Success! Generated model object:")
        print(f"Thought: {result.model_object.thought}")
        if result.model_object.action:
            print(f"Action - Tool name: {result.model_object.action.tool_name}")
            print(f"Action - Parameters: {result.model_object.action.parameters}")
        print(f"Response: {result.model_object.response}")
        # Convert to JSON for display
        print("As JSON:")
        print(json.dumps(result.model_object.model_dump(), indent=2))
    else:
        print(f"Error: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

