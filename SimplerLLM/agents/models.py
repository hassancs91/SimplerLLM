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
