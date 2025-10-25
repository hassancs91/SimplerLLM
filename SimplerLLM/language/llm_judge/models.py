"""
Pydantic models for the LLM Judge system.

This module defines the data structures used for multi-provider evaluation,
comparison, and synthesis of LLM responses.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class JudgeMode(str, Enum):
    """Mode for judge evaluation."""
    SELECT_BEST = "select_best"      # Judge picks the best answer
    SYNTHESIZE = "synthesize"        # Judge combines all answers into improved response
    COMPARE = "compare"              # Judge provides detailed comparative analysis


class ProviderResponse(BaseModel):
    """Individual provider's response to a prompt."""
    provider_name: str = Field(description="Name of the LLM provider (e.g., 'OPENAI', 'ANTHROPIC')")
    model_name: str = Field(description="Specific model used (e.g., 'gpt-4', 'claude-sonnet-4')")
    response_text: str = Field(description="The actual response text from the provider")
    execution_time: float = Field(description="Time taken to generate response in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")

    class Config:
        json_schema_extra = {
            "example": {
                "provider_name": "OPENAI",
                "model_name": "gpt-4",
                "response_text": "Quantum computing uses quantum mechanics principles...",
                "execution_time": 2.3,
                "timestamp": "2025-01-15T10:30:00",
                "error": None
            }
        }


class ProviderEvaluation(BaseModel):
    """Evaluation metrics for a single provider's response."""
    provider_name: str = Field(description="Name of the provider being evaluated")
    overall_score: float = Field(
        description="Overall quality score from 1-10",
        ge=1.0,
        le=10.0
    )
    rank: int = Field(
        description="Ranking position (1 = best, 2 = second best, etc.)",
        ge=1
    )
    criterion_scores: Dict[str, float] = Field(
        description="Scores for individual criteria (e.g., {'accuracy': 9.0, 'clarity': 8.5})",
        default_factory=dict
    )
    reasoning: str = Field(description="Judge's explanation for the scores and ranking")
    strengths: Optional[List[str]] = Field(
        default=None,
        description="List of identified strengths in this response"
    )
    weaknesses: Optional[List[str]] = Field(
        default=None,
        description="List of identified weaknesses in this response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "provider_name": "OPENAI",
                "overall_score": 9.2,
                "rank": 1,
                "criterion_scores": {
                    "accuracy": 9.5,
                    "clarity": 9.0,
                    "completeness": 9.0
                },
                "reasoning": "Excellent technical accuracy with clear explanations...",
                "strengths": ["Clear examples", "Accurate definitions"],
                "weaknesses": ["Could be more concise"]
            }
        }


class JudgeEvaluation(BaseModel):
    """Internal model for judge's structured evaluation output."""
    evaluations: List[ProviderEvaluation] = Field(
        description="List of evaluations for each provider"
    )
    final_answer: str = Field(
        description="Judge's final answer (selected best or synthesized response)"
    )
    overall_reasoning: str = Field(
        description="Judge's overall reasoning and decision process"
    )
    confidence_scores: Dict[str, float] = Field(
        description="Confidence scores for each provider (0-1 scale)",
        default_factory=dict
    )


class JudgeResult(BaseModel):
    """Complete result from LLM Judge evaluation."""
    final_answer: str = Field(description="Judge's final answer (selected or synthesized)")
    all_responses: List[ProviderResponse] = Field(description="All provider responses")
    evaluations: List[ProviderEvaluation] = Field(description="Evaluation for each provider")
    judge_reasoning: str = Field(description="Judge's overall reasoning")
    confidence_scores: Dict[str, float] = Field(
        description="Confidence score for each provider (0-1)",
        default_factory=dict
    )
    mode: JudgeMode = Field(description="Mode used for evaluation")
    criteria_used: List[str] = Field(description="Criteria used for evaluation")
    total_execution_time: float = Field(description="Total time for all providers + judge in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When evaluation completed")

    class Config:
        json_schema_extra = {
            "example": {
                "final_answer": "Quantum computing leverages quantum mechanical phenomena...",
                "all_responses": [],
                "evaluations": [],
                "judge_reasoning": "GPT-4 provided the most accurate and clear explanation...",
                "confidence_scores": {"OPENAI": 0.92, "ANTHROPIC": 0.85, "GEMINI": 0.78},
                "mode": "select_best",
                "criteria_used": ["accuracy", "clarity", "completeness"],
                "total_execution_time": 8.5,
                "timestamp": "2025-01-15T10:30:15"
            }
        }


class RouterSummary(BaseModel):
    """Summary data for LLMRouter training and configuration."""
    query_type: str = Field(description="Inferred type of query (e.g., 'technical_explanation', 'creative_writing')")
    winning_provider: str = Field(description="Provider that performed best")
    provider_scores: Dict[str, float] = Field(description="Score for each provider")
    recommendation: str = Field(description="Recommendation for router configuration")
    criteria_winners: Dict[str, str] = Field(
        description="Which provider won for each criterion",
        default_factory=dict
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_type": "technical_explanation",
                "winning_provider": "OPENAI",
                "provider_scores": {"OPENAI": 9.2, "ANTHROPIC": 8.5, "GEMINI": 7.8},
                "recommendation": "Use OPENAI (GPT-4) for technical explanations",
                "criteria_winners": {
                    "accuracy": "OPENAI",
                    "clarity": "OPENAI",
                    "completeness": "ANTHROPIC"
                }
            }
        }


class EvaluationReport(BaseModel):
    """Statistical summary for batch evaluations and benchmarking."""
    total_queries: int = Field(description="Total number of queries evaluated")
    provider_win_counts: Dict[str, int] = Field(
        description="Number of times each provider won (ranked #1)",
        default_factory=dict
    )
    average_scores: Dict[str, float] = Field(
        description="Average overall score for each provider",
        default_factory=dict
    )
    best_provider_overall: str = Field(description="Provider with highest average score")
    best_provider_by_criteria: Dict[str, str] = Field(
        description="Best provider for each evaluation criterion",
        default_factory=dict
    )
    evaluation_data: List[Dict[str, Any]] = Field(
        description="Detailed evaluation data for export",
        default_factory=list
    )
    generated_at: datetime = Field(default_factory=datetime.now, description="When report was generated")

    class Config:
        json_schema_extra = {
            "example": {
                "total_queries": 50,
                "provider_win_counts": {"OPENAI": 30, "ANTHROPIC": 15, "GEMINI": 5},
                "average_scores": {"OPENAI": 8.7, "ANTHROPIC": 8.2, "GEMINI": 7.5},
                "best_provider_overall": "OPENAI",
                "best_provider_by_criteria": {
                    "accuracy": "OPENAI",
                    "clarity": "ANTHROPIC",
                    "completeness": "OPENAI"
                },
                "evaluation_data": [],
                "generated_at": "2025-01-15T11:00:00"
            }
        }
