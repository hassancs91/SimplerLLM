from pydantic import BaseModel
from typing import Any, Optional, List, Dict
from datetime import datetime


class LLMFullResponse(BaseModel):
    generated_text: str
    model: str
    process_time: float
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    llm_provider_response: Any
    model_object: Optional[Any] = None
    provider: Optional[Any] = None
    model_name: Optional[str] = None
    guardrails_metadata: Optional[Dict[str, Any]] = None


class LLMEmbeddingsResponse(BaseModel):
    generated_embedding: Any
    model: str
    process_time: float
    llm_provider_response: Any


class PatternMatch(BaseModel):
    """Represents a single pattern match extracted from text."""
    value: str
    """The extracted value as found in the text"""

    normalized_value: Optional[str] = None
    """The normalized version of the value (if normalization was applied)"""

    pattern_type: str
    """The type of pattern matched (e.g., 'email', 'phone', 'custom')"""

    position: int
    """The starting position of the match in the original text"""

    is_valid: bool = True
    """Whether the match passed validation checks beyond regex"""

    validation_message: Optional[str] = None
    """Validation details or error message if validation failed"""

    confidence: Optional[float] = None
    """Match quality score (0-1), if applicable"""

    class Config:
        json_schema_extra = {
            "example": {
                "value": "john.doe@example.com",
                "normalized_value": "john.doe@example.com",
                "pattern_type": "email",
                "position": 42,
                "is_valid": True,
                "validation_message": "Valid email format",
                "confidence": 1.0
            }
        }


class PatternExtractionResult(BaseModel):
    """Result of a pattern extraction operation from LLM output."""
    matches: List[PatternMatch]
    """List of all extracted pattern matches"""

    total_matches: int
    """Total number of matches found"""

    pattern_used: str
    """The regex pattern that was used for extraction"""

    original_text: str
    """The original text from the LLM response"""

    extraction_timestamp: datetime
    """When the extraction was performed"""

    class Config:
        json_schema_extra = {
            "example": {
                "matches": [
                    {
                        "value": "john@example.com",
                        "normalized_value": "john@example.com",
                        "pattern_type": "email",
                        "position": 0,
                        "is_valid": True,
                        "validation_message": "Valid email format",
                        "confidence": 1.0
                    }
                ],
                "total_matches": 1,
                "pattern_used": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "original_text": "Contact us at john@example.com for more information.",
                "extraction_timestamp": "2025-01-15T10:30:00"
            }
        }
