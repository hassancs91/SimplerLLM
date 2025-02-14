"""
LLM Router - A system for intelligent content routing using LLMs

This module provides a flexible routing system that uses Large Language Models to select
the most appropriate choice from a collection based on input text. It supports batching,
multiple returns, custom prompting, and confidence thresholds.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
from abc import ABC, abstractmethod

class RouterResponse(BaseModel):
    """Response model for router predictions"""
    selected_index: int
    confidence_score: float = Field(ge=0, le=1)
    reasoning: str

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

Return your response in the following JSON format:
{
    "selected_index": <integer>,
    "confidence_score": <float between 0 and 1>,
    "reasoning": <string explaining why this choice was selected>
}"""

    def format(self, input_text: str, choices_text: str) -> str:
        return self.template.format(
            input=input_text,
            choices=choices_text
        )

class LLMRouter:
    """Main router class for handling choice selection via LLM"""
    
    def __init__(
        self, 
        llm_client: Any,
        confidence_threshold: float = 0.5,
        max_choices_per_batch: int = 50
    ):
        self._choices: List[Choice] = []
        self.llm = llm_client
        self.confidence_threshold = confidence_threshold
        self.max_choices_per_batch = max_choices_per_batch
        self.prompt_template = PromptTemplate()

    def add_choice(self, content: str, metadata: Optional[Dict] = None) -> int:
        """Add a new choice and return its index"""
        cleaned_content = Choice._clean_string(content)
        
        # Check for duplicates
        if any(choice.content == cleaned_content for choice in self._choices):
            raise ValueError("Duplicate choice content")
            
        choice = Choice(content, metadata)
        self._choices.append(choice)
        return len(self._choices) - 1

    def remove_choice(self, index: int) -> None:
        """Remove a choice by index"""
        if 0 <= index < len(self._choices):
            self._choices.pop(index)
        else:
            raise IndexError("Choice index out of range")

    def update_choice(self, index: int, content: str, metadata: Optional[Dict] = None) -> None:
        """Update an existing choice"""
        if 0 <= index < len(self._choices):
            cleaned_content = Choice._clean_string(content)
            if any(i != index and choice.content == cleaned_content for i, choice in enumerate(self._choices)):
                raise ValueError("Duplicate choice content")
            self._choices[index] = Choice(content, metadata)
        else:
            raise IndexError("Choice index out of range")

    def set_prompt_template(self, template: str) -> None:
        """Set custom prompt template"""
        self.prompt_template = PromptTemplate(template)

    def _chunk_choices(self) -> List[List[Choice]]:
        """Split choices into manageable batches"""
        return [
            self._choices[i:i + self.max_choices_per_batch]
            for i in range(0, len(self._choices), self.max_choices_per_batch)
        ]

    def _format_choices_text(self, choices: List[Choice]) -> str:
        """Format choices for prompt"""
        return "\n\n".join([
            f"ID: {i}\nChoice: {choice.content}" 
            for i, choice in enumerate(choices)
        ])

    async def _route_batch(
        self, 
        input_text: str, 
        choices: List[Choice]
    ) -> RouterResponse:
        """Route within a single batch of choices"""
        choices_text = self._format_choices_text(choices)
        prompt = self.prompt_template.format(input_text, choices_text)
        
        # LLM call implementation here
        # This should be implemented based on your LLM client
        response = await self.llm.generate(prompt)
        
        # Parse LLM response and return RouterResponse
        # Implementation needed
        pass

    async def route(self, input_text: str) -> Optional[RouterResponse]:
        """Find best matching choice for input"""
        if not self._choices:
            raise ValueError("No choices available for routing")

        best_response = None
        
        for batch in self._chunk_choices():
            response = await self._route_batch(input_text, batch)
            
            if response.confidence_score > self.confidence_threshold:
                if not best_response or response.confidence_score > best_response.confidence_score:
                    best_response = response

        return best_response

    async def route_top_k(
        self, 
        input_text: str, 
        k: int = 3
    ) -> List[RouterResponse]:
        """Return top K matching choices"""
        if not self._choices:
            raise ValueError("No choices available for routing")
        
        if k < 1:
            raise ValueError("k must be >= 1")
        
        all_responses = []
        
        for batch in self._chunk_choices():
            response = await self._route_batch(input_text, batch)
            if response.confidence_score > self.confidence_threshold:
                all_responses.append(response)
        
        # Sort by confidence score and return top k
        return sorted(
            all_responses, 
            key=lambda x: x.confidence_score, 
            reverse=True
        )[:k]

# Example usage:
"""
# Initialize router with your LLM client
router = LLMRouter(
    llm_client=your_llm_client,
    confidence_threshold=0.7,
    max_choices_per_batch=50
)

# Add choices
router.add_choice("Template 1...", metadata={"type": "tweet"})
router.add_choice("Template 2...", metadata={"type": "thread"})

# Custom prompt template if needed
router.set_prompt_template('''
Analyze the following input and select the most appropriate choice.
Input: {input}

Available choices:
{choices}

Respond in JSON format with selected_index, confidence_score, and reasoning.
''')

# Route input to best choice
best_match = await router.route("Some input text...")

# Get top 3 matches
top_matches = await router.route_top_k("Some input text...", k=3)
"""