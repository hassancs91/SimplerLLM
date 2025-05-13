from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union

class RouterDecision(BaseModel):
    """
    Represents a routing decision made by the LLMRouter.
    """
    decision_type: str = Field(..., description="Type of decision (direct_answer, use_tools, check_memory)")
    confidence: float = Field(..., description="Confidence in this decision", ge=0, le=1)
    reasoning: str = Field(..., description="Reasoning behind this decision")

class AgentRole(BaseModel):
    """
    Represents a role definition for an agent.
    """
    name: str = Field(..., description="Name of the role")
    description: str = Field(..., description="Description of the role")
    system_prompt: str = Field(..., description="System prompt for the role")
    responsibilities: Optional[List[str]] = Field(None, description="List of responsibilities")
    constraints: Optional[List[str]] = Field(None, description="List of constraints")
    allowed_tools: Optional[List[str]] = Field(None, description="List of allowed tools")
    priority_level: Optional[int] = Field(None, description="Priority level for orchestration")
    fallback_behavior: Optional[str] = Field(None, description="Fallback behavior when uncertain")

class ToolCall(BaseModel):
    """
    Represents a tool call that the agent wants to make.
    """
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(..., description="Parameters to pass to the tool")

class AgentThought(BaseModel):
    """
    Represents the agent's reasoning process.
    """
    thought: str = Field(..., description="The agent's reasoning process")
    
class AgentAction(BaseModel):
    """
    Represents an action the agent wants to take, which could be either:
    1. Calling a tool (action field is populated)
    2. Providing a final response (response field is populated)
    """
    thought: str = Field(..., description="The agent's reasoning process")
    action: Optional[ToolCall] = Field(None, description="Tool to call, if any")
    response: Optional[str] = Field(None, description="Final response to user if no tool call is needed")

class AgentResponse(BaseModel):
    """
    Represents a complete agent response cycle including:
    - The agent's thought process
    - An optional tool call
    - An optional final response
    - An optional observation from a tool execution
    """
    thought: str = Field(..., description="The agent's reasoning process")

class NextStepDecision(BaseModel):
    """
    Represents the agent's decision on how to proceed after an analysis step.
    It can decide to answer directly, use a tool, or ask for clarification.
    """
    thought: str = Field(..., description="The agent's reasoning about the current state and what to do next.")
    can_answer_directly: bool = Field(..., description="True if the agent believes it can answer the original query with the current information.")
    next_action: Optional[ToolCall] = Field(None, description="Tool to call if further action is needed. Null if answering directly or asking for clarification.")
    clarification_request: Optional[str] = Field(None, description="A question to the user if more information is needed from them and no tool can help.")
    final_response: Optional[str] = Field(None, description="The final answer to the user if can_answer_directly is true.")

    # Potential validator example (can be added if strict logic is needed at Pydantic level):
    # from pydantic import root_validator
    # @root_validator
    # def check_logical_consistency(cls, values):
    #     can_answer = values.get('can_answer_directly')
    #     action = values.get('next_action')
    #     clarification = values.get('clarification_request')
    #     response = values.get('final_response')

    #     if can_answer and not response:
    #         raise ValueError("If can_answer_directly is true, final_response must be provided.")
    #     if can_answer and (action or clarification):
    #         raise ValueError("If can_answer_directly is true, next_action and clarification_request must be null.")
    #     if not can_answer and not action and not clarification:
    #         raise ValueError("If not answering directly, either next_action or clarification_request must be provided.")
    #     if action and clarification:
    #         raise ValueError("Cannot have both next_action and clarification_request.")
    #     return values
