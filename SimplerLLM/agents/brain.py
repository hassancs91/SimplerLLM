"""
Agent brain module.

This module implements the core decision-making logic for the agent,
using an iterative refinement process to determine the next best action.
"""

from typing import Dict, Any, Optional, List
from SimplerLLM.language.llm import LLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
from .memory_interface import BaseMemory
from .tool_registry import ToolRegistry
from .models import NextStepDecision, ToolCall # Added NextStepDecision, ToolCall

class AgentBrain:
    """
    Core decision-making component for the agent.
    Implements an iterative loop to process queries, deciding whether to
    use a tool, ask for clarification, or provide a final answer.
    """
    
    def __init__(
        self,
        llm: LLM,
        memory: BaseMemory,
        tool_registry: ToolRegistry,
        verbose: bool = False
    ):
        """
        Initialize the agent brain.
        
        Args:
            llm: LLM instance for generating responses and making decisions.
            memory: Memory system for storing conversation history and observations.
            tool_registry: Registry of available tools.
            verbose: Whether to print verbose output.
        """
        self.llm = llm
        self.memory = memory
        self.tool_registry = tool_registry
        self.verbose = verbose
        
    def _determine_next_step(self, original_query: str) -> NextStepDecision:
        """
        Determines the next step for the agent based on the current memory state.
        This can be: answer directly, use a tool, or ask for clarification.
        """
        history = self.memory.get_chat_history() # Gets full history including observations
        tool_descriptions = self.tool_registry.get_tool_descriptions()

        # Constructing the prompt for the LLM to decide the next step.
        # This prompt asks the LLM to act as a reasoning agent and fill a Pydantic model.
        prompt = f"""
You are an AI agent processing a user query.
Original User Query: {original_query}

Current Conversation History (including your previous thoughts and tool observations):
{history}

Available Tools:
{tool_descriptions}

Based on the original query and the full conversation history (including any tool observations you've made):
1.  Analyze if you have sufficient information to provide a complete and final answer to the original user query.
2.  If yes, formulate the final answer.
3.  If no, decide if using one of the available tools can help you get the remaining information.
4.  If no tool can help, or if the query is too ambiguous even after tool use, formulate a question to the user for clarification.

Respond with a JSON object conforming to the 'NextStepDecision' Pydantic model.
Make sure 'thought' explains your reasoning.
- If you can answer directly: set 'can_answer_directly' to true, provide the 'final_response'.
- If you need to use a tool: set 'can_answer_directly' to false, populate 'next_action' (with 'tool_name' and 'parameters'), and leave 'final_response' and 'clarification_request' as null.
- If you need to ask the user for clarification: set 'can_answer_directly' to false, provide the 'clarification_request', and leave 'next_action' and 'final_response' as null.

Pydantic Model for NextStepDecision:
class ToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class NextStepDecision(BaseModel):
    thought: str
    can_answer_directly: bool
    next_action: Optional[ToolCall] = None
    clarification_request: Optional[str] = None
    final_response: Optional[str] = None

JSON Response:
"""
        # Use generate_pydantic_json_model to get a structured decision
        decision = generate_pydantic_json_model(
            model_class=NextStepDecision,
            prompt=prompt,
            llm_instance=self.llm,
            temperature=0.7, 
            max_tokens=1500, 
            system_prompt="You are an expert at deciding the next step in a complex task."
        )
        if self.verbose:
            # Using .model_dump_json for Pydantic v2+
            print(f"Next Step Decision: {decision.model_dump_json(indent=2)}")
        return decision

    def process_query(self, query: str, max_iterations: int = 10) -> str:
        """
        Process a user query through an iterative decision-making loop.
        
        Args:
            query: The user's original query.
            max_iterations: Maximum number of iterations to prevent infinite loops.
            
        Returns:
            The final response or a clarification request to the user.
        """
        self.memory.add_user_message(query) # Add initial user query to memory

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            if self.verbose:
                print(f"\nIteration {iteration} - Original query: {query}")

            # Core decision making for the next step
            next_step_decision = self._determine_next_step(original_query=query)

            # Record the agent's thought process from the decision
            if next_step_decision.thought: # Check if thought is not None
                 self.memory.add_assistant_message(f"Thought: {next_step_decision.thought}")

            if next_step_decision.can_answer_directly:
                if next_step_decision.final_response:
                    self.memory.add_assistant_message(next_step_decision.final_response)
                    return next_step_decision.final_response
                else:
                    # This case should ideally be prevented by good prompting and Pydantic validation
                    err_msg = "Agent decided it can answer but provided no response. Please check LLM output or prompt."
                    if self.verbose: print(err_msg)
                    self.memory.add_assistant_message(err_msg)
                    return err_msg

            elif next_step_decision.next_action:
                tool_name = next_step_decision.next_action.tool_name
                parameters = next_step_decision.next_action.parameters
                
                if self.verbose:
                    print(f"Executing tool: {tool_name} with parameters: {parameters}")
                
                if tool_name in self.tool_registry.list_tools():
                    try:
                        tool_result = self.tool_registry.execute_tool(tool_name, **parameters)
                        observation = f"Tool {tool_name} returned: {str(tool_result)}" # Ensure tool_result is string
                        if self.verbose: print(f"Tool observation: {observation}")
                        self.memory.add_assistant_message(f"Observation: {observation}")
                        # Loop continues, _determine_next_step will be called again with new observation in memory
                        continue 
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        self.memory.add_assistant_message(f"Error: {error_msg}")
                        if self.verbose: print(error_msg)
                        return error_msg # Exit on tool error, returning the error message
                else:
                    error_msg = f"Tool {tool_name} not found. Available tools: {', '.join(self.tool_registry.list_tools())}"
                    self.memory.add_assistant_message(f"Error: {error_msg}")
                    if self.verbose: print(error_msg)
                    return error_msg # Exit if tool not found

            elif next_step_decision.clarification_request:
                # If the agent needs to ask for clarification
                self.memory.add_assistant_message(next_step_decision.clarification_request)
                return next_step_decision.clarification_request

            else:
                # Fallback if NextStepDecision is not logically sound 
                # (e.g., can_answer_directly is false, but no action and no clarification)
                # This should be rare if the Pydantic model and LLM prompt are well-defined.
                fallback_msg = "I'm not sure how to proceed with the information I have. Can you please rephrase your request or provide more details?"
                if self.verbose: print("Fallback: No clear next step from LLM decision. The decision object might be malformed or illogical.")
                self.memory.add_assistant_message(fallback_msg)
                return fallback_msg
        
        # Max iterations reached
        max_iter_msg = f"Reached maximum number of iterations ({max_iterations}). I may not have a complete answer based on the current information flow."
        self.memory.add_assistant_message(max_iter_msg)
        if self.verbose: print(max_iter_msg)
        # Consider attempting one last generation or returning a summary of thoughts.
        # For now, just return the max_iter_msg.
        return max_iter_msg

# Old methods to be removed:
# _setup_router
# _make_initial_decision
# _check_memory
# _select_tool
# _create_tool_selection_prompt
# _can_answer_now
# _generate_direct_response
# _generate_response_with_memory
# _generate_response_with_tool_result
