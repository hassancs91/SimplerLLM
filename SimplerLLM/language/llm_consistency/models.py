"""
Pydantic models for the Self-Consistency system.

This module defines the data structures used for self-consistency voting,
including sample responses, answer groups, and the final result.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AnswerType(str, Enum):
    """Type of answer comparison to use."""
    EXACT = "exact"       # Numbers, bools, specific short values - direct comparison
    SEMANTIC = "semantic" # Text/explanations - LLM-based semantic comparison


class ExtractedAnswer(BaseModel):
    """Structured answer extracted from LLM response."""
    answer: str = Field(description="The core answer extracted from the response")
    answer_type: AnswerType = Field(description="Whether this is an exact or semantic answer")
    reasoning: Optional[str] = Field(default=None, description="Brief reasoning if extracted")


class SampleResponse(BaseModel):
    """Individual response from one sample run."""
    index: int = Field(description="Sample index (0-based)")
    response: str = Field(description="Full response from the LLM")
    extracted_answer: str = Field(description="The core answer extracted from the response")
    execution_time: float = Field(description="Time taken to generate in seconds")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")

    class Config:
        json_schema_extra = {
            "example": {
                "index": 0,
                "response": "Let me calculate: 17 x 24 = 408. The answer is 408.",
                "extracted_answer": "408",
                "execution_time": 1.2,
                "error": None
            }
        }


class AnswerGroup(BaseModel):
    """Group of similar/identical answers."""
    answer: str = Field(description="Representative answer for this group")
    count: int = Field(description="Number of samples with this answer")
    sample_indices: List[int] = Field(description="Indices of samples in this group")
    responses: List[str] = Field(description="Full responses in this group")
    percentage: float = Field(description="Percentage of total samples (0-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "408",
                "count": 4,
                "sample_indices": [0, 1, 2, 4],
                "responses": ["The answer is 408.", "17 x 24 = 408", "408", "Result: 408"],
                "percentage": 80.0
            }
        }


class SemanticGrouping(BaseModel):
    """Result of LLM-based semantic grouping."""
    groups: List[List[int]] = Field(description="List of groups, each containing sample indices")
    representative_answers: List[str] = Field(description="Representative answer for each group")


class ConsistencyResult(BaseModel):
    """Complete result from self-consistency voting."""
    final_answer: str = Field(description="The consensus answer (most common)")
    confidence: float = Field(
        description="Agreement ratio (0.0 to 1.0) - proportion of samples that agree",
        ge=0.0,
        le=1.0
    )
    num_samples: int = Field(description="Total number of samples generated")
    num_agreeing: int = Field(description="Number of samples matching the final answer")
    is_tie: bool = Field(description="True if multiple answers had the same highest count")
    tied_answers: Optional[List[str]] = Field(
        default=None,
        description="All answers that tied for highest count (if is_tie=True)"
    )
    all_samples: List[SampleResponse] = Field(description="All generated sample responses")
    answer_groups: List[AnswerGroup] = Field(
        description="Answers grouped by similarity, sorted by count descending"
    )
    answer_type: AnswerType = Field(description="The comparison method used")
    prompt: str = Field(description="The original prompt used")
    execution_time: float = Field(description="Total time for all samples in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When generation completed")

    class Config:
        json_schema_extra = {
            "example": {
                "final_answer": "408",
                "confidence": 0.8,
                "num_samples": 5,
                "num_agreeing": 4,
                "is_tie": False,
                "tied_answers": None,
                "all_samples": [],
                "answer_groups": [],
                "answer_type": "exact",
                "prompt": "What is 17 x 24?",
                "execution_time": 5.5,
                "timestamp": "2025-01-15T10:30:00"
            }
        }
