"""
Pydantic models for the LLM Validator system.

This module defines the data structures used for multi-provider validation
of AI-generated content, including scoring, confidence, and aggregation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AggregationMethod(str, Enum):
    """Method for aggregating scores from multiple validators."""
    AVERAGE = "average"       # Simple mean of all scores
    WEIGHTED = "weighted"     # Weighted average (provider weights)
    MEDIAN = "median"         # Median score
    CONSENSUS = "consensus"   # Majority agreement threshold


class ValidatorScoreOutput(BaseModel):
    """Internal model for structured validator output from LLM."""
    score: float = Field(
        description="Validation score from 0.0 to 1.0 (0 = completely invalid, 1 = perfectly valid)",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        description="How confident you are in this score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    explanation: str = Field(
        description="Detailed explanation of why you gave this score"
    )


class ValidatorScore(BaseModel):
    """Individual validator's score for the content."""
    provider_name: str = Field(description="Name of the LLM provider (e.g., 'OPENAI', 'ANTHROPIC')")
    model_name: str = Field(description="Specific model used (e.g., 'gpt-4o', 'claude-3-5-sonnet')")
    score: float = Field(
        description="Validation score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        description="How confident the validator is in its score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    explanation: str = Field(description="Explanation for the score")
    is_valid: bool = Field(description="Whether the content passes validation (score >= threshold)")
    execution_time: float = Field(description="Time taken to validate in seconds")
    error: Optional[str] = Field(default=None, description="Error message if validation failed")

    class Config:
        json_schema_extra = {
            "example": {
                "provider_name": "OPENAI",
                "model_name": "gpt-4o",
                "score": 0.85,
                "confidence": 0.92,
                "explanation": "The content is factually accurate and well-structured.",
                "is_valid": True,
                "execution_time": 1.5,
                "error": None
            }
        }


class ValidationResult(BaseModel):
    """Complete result from LLM Validator."""
    overall_score: float = Field(
        description="Aggregated validation score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    overall_confidence: float = Field(
        description="Aggregated confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    is_valid: bool = Field(description="Overall pass/fail based on threshold")
    validators: List[ValidatorScore] = Field(description="Individual validator scores")
    consensus: bool = Field(description="Whether validators agreed on the result")
    consensus_details: str = Field(description="Explanation of validator agreement")
    aggregation_method: AggregationMethod = Field(description="Method used for aggregation")
    content_validated: str = Field(description="The content that was validated")
    validation_prompt: str = Field(description="The validation instructions used")
    original_question: Optional[str] = Field(default=None, description="The original question if provided")
    total_execution_time: float = Field(description="Total time for all validators in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When validation completed")

    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 0.85,
                "overall_confidence": 0.90,
                "is_valid": True,
                "validators": [],
                "consensus": True,
                "consensus_details": "All validators scored within 0.15 of each other",
                "aggregation_method": "average",
                "content_validated": "Paris is the capital of France.",
                "validation_prompt": "Check if the facts are accurate.",
                "original_question": "What is the capital of France?",
                "total_execution_time": 3.2,
                "timestamp": "2025-01-15T10:30:00"
            }
        }
