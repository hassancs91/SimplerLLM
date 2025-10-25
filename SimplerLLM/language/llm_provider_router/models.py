"""
Pydantic models for the LLM Provider Router system.

This module defines the data structures used for intelligent provider routing
based on query classification.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

from SimplerLLM.language.llm.base import LLM


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    # Note: LLM instance is not a Pydantic field (not serializable)
    # It's stored separately and referenced by index

    llm_provider: str = Field(description="Provider name (e.g., 'OPENAI', 'ANTHROPIC')")
    llm_model: str = Field(description="Model name (e.g., 'gpt-4', 'claude-sonnet-4')")
    specialties: List[str] = Field(
        description="Query types this provider handles well",
        default_factory=list
    )
    description: str = Field(
        description="Description of provider's strengths and use cases"
    )
    priority: int = Field(
        description="Priority level 1-10 (higher = preferred when multiple match)",
        ge=1,
        le=10,
        default=5
    )
    has_fallback: bool = Field(
        default=False,
        description="Whether this provider has a fallback LLM configured"
    )
    fallback_provider: Optional[str] = Field(
        default=None,
        description="Fallback provider name if primary fails"
    )
    fallback_model: Optional[str] = Field(
        default=None,
        description="Fallback model name if primary fails"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this provider is currently enabled"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "llm_provider": "OPENAI",
                "llm_model": "gpt-4",
                "specialties": ["coding", "technical_writing", "debugging"],
                "description": "Best for programming and technical tasks",
                "priority": 8,
                "has_fallback": True,
                "fallback_provider": "OPENAI",
                "fallback_model": "gpt-4o-mini",
                "enabled": True
            }
        }


class QueryClassification(BaseModel):
    """Result of query classification."""
    query_type: str = Field(description="Classified query type (e.g., 'coding', 'creative')")
    confidence: float = Field(
        description="Confidence in classification (0-1)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Why this classification was chosen")
    matched_by: str = Field(
        description="Classification method used: 'pattern', 'llm', or 'cache'"
    )
    alternative_types: Optional[List[str]] = Field(
        default=None,
        description="Other possible query types (if any)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_type": "coding",
                "confidence": 0.95,
                "reasoning": "Query contains 'write code' which matches coding patterns",
                "matched_by": "pattern",
                "alternative_types": ["technical_explanation"]
            }
        }


class RoutingResult(BaseModel):
    """Complete result from provider routing and execution."""
    answer: str = Field(description="The LLM's response to the query")
    provider_used: str = Field(description="Provider name that generated the answer")
    model_used: str = Field(description="Model name that generated the answer")
    query_classification: QueryClassification = Field(description="Query classification details")
    routing_confidence: float = Field(
        description="Confidence in provider selection (0-1)",
        ge=0.0,
        le=1.0
    )
    routing_reasoning: str = Field(description="Why this provider was selected")
    used_fallback: bool = Field(
        default=False,
        description="Whether fallback provider was used"
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Reason fallback was needed (if applicable)"
    )
    execution_time: float = Field(description="Total execution time in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the routing completed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Here's a Python function to reverse a string...",
                "provider_used": "OPENAI",
                "model_used": "gpt-4",
                "query_classification": {},
                "routing_confidence": 0.92,
                "routing_reasoning": "OpenAI GPT-4 specializes in coding tasks",
                "used_fallback": False,
                "fallback_reason": None,
                "execution_time": 2.3,
                "timestamp": "2025-01-15T10:30:00"
            }
        }


class RouterConfig(BaseModel):
    """Configuration for LLMProviderRouter (for export/import)."""
    providers: List[ProviderConfig] = Field(description="List of provider configurations")
    default_provider_index: Optional[int] = Field(
        default=None,
        description="Index of default provider (fallback)"
    )
    classification_method: str = Field(
        default="hybrid",
        description="Classification method: 'pattern', 'llm', or 'hybrid'"
    )
    cache_enabled: bool = Field(
        default=False,
        description="Whether classification caching is enabled"
    )
    cache_ttl: Optional[int] = Field(
        default=None,
        description="Cache TTL in seconds (None = no expiration)"
    )
    pattern_rules: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Custom pattern rules: query_type -> list of regex patterns"
    )
    classifier_provider: Optional[str] = Field(
        default=None,
        description="Provider name for LLM-based classification"
    )
    classifier_model: Optional[str] = Field(
        default=None,
        description="Model name for LLM-based classification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "providers": [],
                "default_provider_index": 0,
                "classification_method": "hybrid",
                "cache_enabled": True,
                "cache_ttl": 3600,
                "pattern_rules": {
                    "coding": ["write.*code", "debug", "implement"]
                },
                "classifier_provider": "OPENAI",
                "classifier_model": "gpt-4o-mini"
            }
        }


class CachedClassification(BaseModel):
    """Cached query classification entry."""
    query_hash: str = Field(description="Hash of the query (for cache key)")
    classification: QueryClassification = Field(description="The cached classification")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this was cached"
    )
    hits: int = Field(
        default=1,
        description="Number of times this cache entry was used"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_hash": "abc123def456",
                "classification": {},
                "timestamp": "2025-01-15T10:30:00",
                "hits": 5
            }
        }


# Internal model for LLM-based classification (structured output)
class LLMClassificationResponse(BaseModel):
    """Structured response from LLM classifier."""
    query_type: str = Field(description="Primary query type classification")
    confidence: float = Field(
        description="Confidence in this classification (0-1)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Reasoning behind the classification")
    alternative_types: List[str] = Field(
        default_factory=list,
        description="Alternative possible query types",
        max_items=3
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_type": "technical_explanation",
                "confidence": 0.88,
                "reasoning": "Query asks for explanation of a technical concept",
                "alternative_types": ["general", "educational"]
            }
        }
