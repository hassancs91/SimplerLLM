"""
Models for the LLM Router system.
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class RouterChoice(BaseModel):
    """Response model for router predictions"""
    selected_index: int = Field(
        description="Index of the selected choice",
        ge=0
    )
    confidence_score: float = Field(
        description="Confidence score for the selection",
        ge=0,
        le=1
    )
    reasoning: str = Field(
        description="Explanation for why this choice was selected"
    )

class RouterMultiResponse(BaseModel):
    choices: List[RouterChoice] = Field(
        description="List of selected choices in order of confidence",
        max_items=3
    )

class RouterResponse(BaseModel):
    """Response model for router predictions"""
    selected_index: int = Field(
        description="Index of the selected choice",
        ge=0
    )
    confidence_score: float = Field(
        description="Confidence score for the selection",
        ge=0,
        le=1
    )
    reasoning: str = Field(
        description="Explanation for why this choice was selected"
    )

class Choice:
    """Represents a single choice in the router"""
    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = self._clean_string(content)
        self.metadata = metadata or {}

    @staticmethod
    def _clean_string(text: str) -> str:
        """Clean and normalize string content"""
        if not text:
            raise ValueError("Choice content cannot be empty")
        return " ".join(text.split())
    
    def __repr__(self):
        return f"Choice(content={self.content[:50]}...)"

class PromptTemplate:
    """Handles prompt generation for the router"""
    def __init__(self, template: Optional[str] = None):
        self.template = template or self._default_template()
    
    @staticmethod
    def _default_template() -> str:
        return """Your task is to choose the best choice from the list below based on the following input: {input}

Choice List:
{choices}

Select the most appropriate choice based on relevance to the input.
Provide a clear reasoning for your selection."""

    @staticmethod
    def _default_template_top_k() -> str:
        return """Your task is to choose the top {k} most appropriate choices from the list below based on the following input: {input}

Choice List:
{choices}

Select the {k} most appropriate choices based on relevance to the input.
Provide reasoning for each selection.

"""

    def format(self, input_text: str, choices_text: str, k: Optional[int] = None) -> str:
        if k is not None:
            return self._default_template_top_k().format(
                input=input_text,
                choices=choices_text,
                k=k
            )
        return self.template.format(
            input=input_text,
            choices=choices_text
        )
