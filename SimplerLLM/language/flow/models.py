from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class StepResult(BaseModel):
    """Result from a single step execution in a flow."""
    step_number: int = Field(description="The step number in the flow (1-indexed)")
    step_type: str = Field(description="Type of step: 'llm' or 'tool'")
    input_data: Any = Field(description="Input data for this step")
    output_data: Any = Field(description="Output data from this step (can be str, dict, or Pydantic model)")
    duration_seconds: float = Field(description="Time taken to execute this step in seconds")
    tool_used: Optional[str] = Field(default=None, description="Name of the tool used (if step_type is 'tool')")
    prompt_used: Optional[str] = Field(default=None, description="Prompt used (if step_type is 'llm')")
    output_model_class: Optional[str] = Field(default=None, description="Name of the Pydantic model class used for JSON output (if applicable)")
    error: Optional[str] = Field(default=None, description="Error message if the step failed")

    class Config:
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "step_type": "tool",
                "input_data": "https://youtube.com/watch?v=xyz",
                "output_data": "Video transcript text...",
                "duration_seconds": 2.5,
                "tool_used": "youtube_transcript",
                "prompt_used": None,
                "error": None
            }
        }


class FlowResult(BaseModel):
    """Result from a complete flow execution."""
    agent_name: str = Field(description="Name of the mini agent that executed the flow")
    total_steps: int = Field(description="Total number of steps executed")
    steps: List[StepResult] = Field(description="List of step results in execution order")
    total_duration_seconds: float = Field(description="Total time taken to execute the flow in seconds")
    final_output: Any = Field(description="Final output from the last step")
    success: bool = Field(description="Whether the flow completed successfully")
    error: Optional[str] = Field(default=None, description="Error message if the flow failed")
    executed_at: datetime = Field(default_factory=datetime.now, description="Timestamp of execution")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "YouTube Summarizer",
                "total_steps": 2,
                "steps": [],
                "total_duration_seconds": 5.8,
                "final_output": "Summary of the video...",
                "success": True,
                "error": None,
                "executed_at": "2025-01-15T10:30:00"
            }
        }
