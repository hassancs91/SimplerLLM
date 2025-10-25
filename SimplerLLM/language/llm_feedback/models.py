"""
Pydantic models for the LLM Feedback Loop system.

This module defines the data structures used for iterative refinement of LLM responses
through self-critique and improvement cycles.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TemperatureSchedule(str, Enum):
    """Temperature scheduling strategy."""
    FIXED = "fixed"           # Use same temperature for all iterations
    DECREASING = "decreasing" # Decrease temperature over iterations
    CUSTOM = "custom"         # Use custom temperature list


class Critique(BaseModel):
    """Structured critique for a single iteration."""
    strengths: List[str] = Field(
        description="Identified strengths in the current answer",
        default_factory=list
    )
    weaknesses: List[str] = Field(
        description="Identified weaknesses and areas for improvement",
        default_factory=list
    )
    improvement_suggestions: List[str] = Field(
        description="Specific, actionable suggestions for improvement",
        default_factory=list
    )
    quality_score: float = Field(
        description="Overall quality score from 1-10",
        ge=1.0,
        le=10.0
    )
    specific_issues: Dict[str, str] = Field(
        description="Specific issues for each criterion (criterion -> issue description)",
        default_factory=dict
    )
    reasoning: str = Field(
        description="Detailed reasoning behind the critique and score"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strengths": [
                    "Clear structure and logical flow",
                    "Good use of examples"
                ],
                "weaknesses": [
                    "Technical jargon could be simplified",
                    "Missing conclusion"
                ],
                "improvement_suggestions": [
                    "Replace technical terms with simpler alternatives",
                    "Add a concise summary at the end"
                ],
                "quality_score": 7.5,
                "specific_issues": {
                    "clarity": "Some terms are too technical for beginners",
                    "completeness": "No conclusion or summary provided"
                },
                "reasoning": "The answer demonstrates good structure but needs simplification..."
            }
        }


class IterationResult(BaseModel):
    """Result from a single iteration in the feedback loop."""
    iteration_number: int = Field(
        description="Iteration number (1-indexed)",
        ge=1
    )
    answer: str = Field(description="The answer generated in this iteration")
    critique: Critique = Field(description="Critique of this answer")
    provider_used: str = Field(description="Provider name used for this iteration")
    model_used: str = Field(description="Model name used for this iteration")
    temperature_used: float = Field(description="Temperature used for generation")
    execution_time: float = Field(description="Time taken for this iteration in seconds")
    improvement_from_previous: Optional[float] = Field(
        default=None,
        description="Improvement percentage from previous iteration (None for first iteration)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this iteration completed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "iteration_number": 2,
                "answer": "Improved answer text...",
                "critique": {},
                "provider_used": "OPENAI",
                "model_used": "gpt-4",
                "temperature_used": 0.7,
                "execution_time": 3.2,
                "improvement_from_previous": 0.15,
                "timestamp": "2025-01-15T10:30:00"
            }
        }


class FeedbackResult(BaseModel):
    """Complete result from feedback loop execution."""
    final_answer: str = Field(description="Final refined answer")
    all_iterations: List[IterationResult] = Field(
        description="Complete history of all iterations"
    )
    initial_score: float = Field(
        description="Quality score from first iteration",
        ge=1.0,
        le=10.0
    )
    final_score: float = Field(
        description="Quality score from final iteration",
        ge=1.0,
        le=10.0
    )
    improvement_trajectory: List[float] = Field(
        description="Quality scores across all iterations"
    )
    total_iterations: int = Field(
        description="Total number of iterations executed",
        ge=1
    )
    stopped_reason: str = Field(
        description="Why the loop stopped: 'max_iterations', 'converged', or 'threshold_met'"
    )
    convergence_detected: bool = Field(
        description="Whether convergence was detected"
    )
    total_execution_time: float = Field(
        description="Total time for all iterations in seconds"
    )
    architecture_used: str = Field(
        description="Architecture pattern used: 'single', 'dual', or 'multi_rotation'"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the feedback loop completed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "final_answer": "Final refined answer...",
                "all_iterations": [],
                "initial_score": 6.5,
                "final_score": 9.2,
                "improvement_trajectory": [6.5, 7.8, 8.5, 9.2],
                "total_iterations": 4,
                "stopped_reason": "converged",
                "convergence_detected": True,
                "total_execution_time": 15.3,
                "architecture_used": "single",
                "timestamp": "2025-01-15T10:35:00"
            }
        }


class FeedbackConfig(BaseModel):
    """Configuration options for feedback loop."""
    max_iterations: int = Field(
        description="Maximum number of iterations",
        ge=1,
        default=3
    )
    convergence_threshold: float = Field(
        description="Stop if improvement is less than this percentage (0.1 = 10%)",
        ge=0.0,
        le=1.0,
        default=0.1
    )
    quality_threshold: Optional[float] = Field(
        default=None,
        description="Stop if quality score reaches this threshold (1-10 scale)"
    )
    focus_criteria: Optional[List[str]] = Field(
        default=None,
        description="Specific criteria to focus improvements on"
    )
    temperature_schedule: Union[str, List[float]] = Field(
        default="fixed",
        description="Temperature schedule: 'fixed', 'decreasing', or list of floats"
    )
    check_convergence: bool = Field(
        default=True,
        description="Whether to check for convergence"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "max_iterations": 5,
                "convergence_threshold": 0.1,
                "quality_threshold": 9.0,
                "focus_criteria": ["clarity", "conciseness"],
                "temperature_schedule": "decreasing",
                "check_convergence": True
            }
        }
