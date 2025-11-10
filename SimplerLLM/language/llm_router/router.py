"""
LLM Router - A system for intelligent content routing using LLMs.

This module provides a flexible routing system that uses Large Language Models to select
the most appropriate choice from a collection based on input prompt.
"""

from typing import List, Optional, Dict, Any, Tuple
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import (
    generate_pydantic_json_model,
    generate_pydantic_json_model_reliable,
    generate_pydantic_json_model_async,
    generate_pydantic_json_model_reliable_async
)
from .models import RouterResponse, RouterMultiResponse, Choice, PromptTemplate

class LLMRouter:
    """Main router class for handling choice selection via LLM"""
    
    def __init__(
        self, 
        llm_instance: Any,
        confidence_threshold: float = 0.5,
        max_choices_per_batch: int = 100
    ):
        self._choices: List[Choice] = []
        self.llm = llm_instance
        self.confidence_threshold = confidence_threshold
        self.max_choices_per_batch = max_choices_per_batch
        self.prompt_template = PromptTemplate()

    def add_choices(self, choices: List[Tuple[str, Optional[Dict]]]) -> List[int]:
        """Add multiple choices at once and return their indices"""
        indices = []
        for content, metadata in choices:
            index = self.add_choice(content, metadata)
            indices.append(index)
        return indices

    def get_choices(self) -> List[Tuple[str, Dict]]:
        """Get all choices with their metadata"""
        return [(choice.content, choice.metadata) for choice in self._choices]

    def get_choice(self, index: int) -> Optional[Tuple[str, Dict]]:
        """Get a specific choice by index"""
        if 0 <= index < len(self._choices):
            choice = self._choices[index]
            return (choice.content, choice.metadata)
        return None

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

    def remove_all_choices(self) -> None:
        """Remove all choices from the router"""
        self._choices.clear()

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

    def _filter_choices_by_metadata(
        self,
        metadata_filter: Dict[str, Any]
    ) -> List[Choice]:
        """Filter choices based on metadata criteria"""
        return [
            choice for choice in self._choices
            if all(
                key in choice.metadata 
                and choice.metadata[key] == value
                for key, value in metadata_filter.items()
            )
        ]

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

    def _route_batch_sync(
        self, 
        input_text: str, 
        choices: List[Choice]
    ) -> Optional[RouterResponse]:
        """Synchronous version of batch routing"""
        choices_text = self._format_choices_text(choices)
        prompt = self.prompt_template.format(input_text, choices_text)
        
        if isinstance(self.llm, ReliableLLM):
            response, _ = generate_pydantic_json_model_reliable(
                model_class=RouterResponse,
                prompt=prompt,
                reliable_llm=self.llm,
                system_prompt="Select the most appropriate choice and provide reasoning"
            )
        else:
            response = generate_pydantic_json_model(
                model_class=RouterResponse,
                prompt=prompt,
                llm_instance=self.llm,
                system_prompt="Select the most appropriate choice and provide reasoning"
            )
            
        if isinstance(response, str):  # Error case
            return None
            
        return response

    async def _route_batch(
        self, 
        input_text: str, 
        choices: List[Choice]
    ) -> Optional[RouterResponse]:
        """Route within a single batch of choices"""
        choices_text = self._format_choices_text(choices)
        prompt = self.prompt_template.format(input_text, choices_text)
        
        if isinstance(self.llm, ReliableLLM):
            response, _ = await generate_pydantic_json_model_reliable_async(
                model_class=RouterResponse,
                prompt=prompt,
                reliable_llm=self.llm,
                system_prompt="Select the most appropriate choice and provide reasoning"
            )
        else:
            response = await generate_pydantic_json_model_async(
                model_class=RouterResponse,
                prompt=prompt,
                llm_instance=self.llm,
                system_prompt="Select the most appropriate choice and provide reasoning"
            )
            
        if isinstance(response, str):  # Error case
            return None
            
        return response

    def route(self, input_text: str) -> Optional[RouterResponse]:
        """Synchronous version of route"""
        if not self._choices:
            raise ValueError("No choices available for routing")

        best_response = None
        
        for batch in self._chunk_choices():
            response = self._route_batch_sync(input_text, batch)
            
            if response and response.confidence_score > self.confidence_threshold:
                if not best_response or response.confidence_score > best_response.confidence_score:
                    best_response = response

        return best_response

    async def route_async(self, input_text: str) -> Optional[RouterResponse]:
        """Asynchronous version of route"""
        if not self._choices:
            raise ValueError("No choices available for routing")

        best_response = None
        
        for batch in self._chunk_choices():
            response = await self._route_batch(input_text, batch)
            
            if response and response.confidence_score > self.confidence_threshold:
                if not best_response or response.confidence_score > best_response.confidence_score:
                    best_response = response

        return best_response

    def _route_top_k_sync(
        self,
        input_text: str,
        k: int,
        choices: List[Choice]
    ) -> Optional[RouterMultiResponse]:
        """Get top k choices in a single LLM call"""
        choices_text = self._format_choices_text(choices)
        prompt = self.prompt_template.format(input_text, choices_text, k=k)
        
        if isinstance(self.llm, ReliableLLM):
            response, _ = generate_pydantic_json_model_reliable(
                model_class=RouterMultiResponse,
                prompt=prompt,
                reliable_llm=self.llm,
                system_prompt=f"Select the top {k} most appropriate choices and provide reasoning for each"
            )
        else:
            response = generate_pydantic_json_model(
                model_class=RouterMultiResponse,
                prompt=prompt,
                llm_instance=self.llm,
                system_prompt=f"Select the top {k} most appropriate choices and provide reasoning for each"
            )
            
        if isinstance(response, str):  # Error case
            return None
            
        return response

    def route_top_k(self, input_text: str, k: int = 3) -> List[RouterResponse]:
        """Get top k matches using a single LLM call"""
        if not self._choices:
            raise ValueError("No choices available for routing")
        
        if k < 1:
            raise ValueError("k must be >= 1")
        
        # Cap k by the number of available choices
        effective_k = min(k, len(self._choices))
        if effective_k == 0:
            return []
            
        multi_response = self._route_top_k_sync(input_text, effective_k, self._choices)
        if not multi_response:
            return []
            
        confident_choices = [
            choice_item for choice_item in multi_response.choices
            if choice_item.confidence_score > self.confidence_threshold
        ]
        # Convert RouterMultiResponse choices to list of RouterResponse
        return [
            RouterResponse(
                selected_index=choice.selected_index,
                confidence_score=choice.confidence_score,
                reasoning=choice.reasoning
            )
            for choice in confident_choices
        ]

    async def _route_top_k_async(
        self,
        input_text: str,
        k: int,
        choices: List[Choice]
    ) -> Optional[RouterMultiResponse]:
        """Get top k choices in a single LLM call"""
        choices_text = self._format_choices_text(choices)
        prompt = self.prompt_template.format(input_text, choices_text, k=k)
        
        if isinstance(self.llm, ReliableLLM):
            response, _ = await generate_pydantic_json_model_reliable_async(
                model_class=RouterMultiResponse,
                prompt=prompt,
                reliable_llm=self.llm,
                system_prompt=f"Select the top {k} most appropriate choices and provide reasoning for each"
            )
        else:
            response = await generate_pydantic_json_model_async(
                model_class=RouterMultiResponse,
                prompt=prompt,
                llm_instance=self.llm,
                system_prompt=f"Select the top {k} most appropriate choices and provide reasoning for each"
            )
            
        if isinstance(response, str):  # Error case
            return None
            
        return response

    def route_with_metadata(
        self, 
        input_text: str, 
        metadata_filter: Dict[str, Any]
    ) -> Optional[RouterResponse]:
        """Route through choices that match the metadata filter"""
        filtered_choices = self._filter_choices_by_metadata(metadata_filter)
        if not filtered_choices:
            return None

        best_response = None
        
        # Split filtered choices into batches
        batches = [
            filtered_choices[i:i + self.max_choices_per_batch]
            for i in range(0, len(filtered_choices), self.max_choices_per_batch)
        ]
        
        for batch in batches:
            response = self._route_batch_sync(input_text, batch)
            
            if response and response.confidence_score > self.confidence_threshold:
                if not best_response or response.confidence_score > best_response.confidence_score:
                    best_response = response

        return best_response

    def route_top_k_with_metadata(
        self, 
        input_text: str, 
        metadata_filter: Dict[str, Any],
        k: int = 3
    ) -> List[RouterResponse]:
        """Get top k matches from choices that match the metadata filter"""
        filtered_choices = self._filter_choices_by_metadata(metadata_filter)
        if not filtered_choices:
            return []
        
        if k < 1:
            raise ValueError("k must be >= 1")
        
        # Cap k by the number of filtered choices
        effective_k = min(k, len(filtered_choices))
        if effective_k == 0:
            return []
            
        multi_response = self._route_top_k_sync(input_text, effective_k, filtered_choices)
        if not multi_response:
            return []
            
        confident_choices = [
            choice_item for choice_item in multi_response.choices
            if choice_item.confidence_score > self.confidence_threshold
        ]
        return [
            RouterResponse(
                selected_index=choice.selected_index,
                confidence_score=choice.confidence_score,
                reasoning=choice.reasoning
            )
            for choice in confident_choices
        ]

    async def route_with_metadata_async(
        self, 
        input_text: str, 
        metadata_filter: Dict[str, Any]
    ) -> Optional[RouterResponse]:
        """Async version of route_with_metadata"""
        filtered_choices = self._filter_choices_by_metadata(metadata_filter)
        if not filtered_choices:
            return None

        best_response = None
        
        # Split filtered choices into batches
        batches = [
            filtered_choices[i:i + self.max_choices_per_batch]
            for i in range(0, len(filtered_choices), self.max_choices_per_batch)
        ]
        
        for batch in batches:
            response = await self._route_batch(input_text, batch)
            
            if response and response.confidence_score > self.confidence_threshold:
                if not best_response or response.confidence_score > best_response.confidence_score:
                    best_response = response

        return best_response

    async def route_top_k_with_metadata_async(
        self, 
        input_text: str, 
        metadata_filter: Dict[str, Any],
        k: int = 3
    ) -> List[RouterResponse]:
        """Async version of route_top_k_with_metadata"""
        filtered_choices = self._filter_choices_by_metadata(metadata_filter)
        if not filtered_choices:
            return []
        
        if k < 1:
            raise ValueError("k must be >= 1")
            
        # Cap k by the number of filtered choices
        effective_k = min(k, len(filtered_choices))
        if effective_k == 0:
            return []
            
        multi_response = await self._route_top_k_async(input_text, effective_k, filtered_choices)
        if not multi_response:
            return []
            
        confident_choices = [
            choice_item for choice_item in multi_response.choices
            if choice_item.confidence_score > self.confidence_threshold
        ]
        return [
            RouterResponse(
                selected_index=choice.selected_index,
                confidence_score=choice.confidence_score,
                reasoning=choice.reasoning
            )
            for choice in confident_choices
        ]

    async def route_top_k_async(
        self, 
        input_text: str, 
        k: int = 3
    ) -> List[RouterResponse]:
        """Get top k matches using a single LLM call"""
        if not self._choices:
            raise ValueError("No choices available for routing")
        
        if k < 1:
            raise ValueError("k must be >= 1")
            
        # Cap k by the number of available choices
        effective_k = min(k, len(self._choices))
        if effective_k == 0:
            return []
            
        multi_response = await self._route_top_k_async(input_text, effective_k, self._choices)
        if not multi_response:
            return []
            
        confident_choices = [
            choice_item for choice_item in multi_response.choices
            if choice_item.confidence_score > self.confidence_threshold
        ]
        # Convert RouterMultiResponse choices to list of RouterResponse
        return [
            RouterResponse(
                selected_index=choice.selected_index,
                confidence_score=choice.confidence_score,
                reasoning=choice.reasoning
            )
            for choice in confident_choices
        ]
