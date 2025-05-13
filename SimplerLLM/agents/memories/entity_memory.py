"""
Entity memory implementation for storing important information.

This module provides a memory system that extracts and stores entities
(people, places, things, etc.) and their attributes from conversations.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from SimplerLLM.language.llm import LLM
from SimplerLLM.agents.memory_interface import BaseMemory

class Entity:
    """Represents an entity with attributes."""
    
    def __init__(self, name: str, entity_type: str):
        """
        Initialize an entity.
        
        Args:
            name: The entity name
            entity_type: The type of entity (person, place, etc.)
        """
        self.name = name
        self.type = entity_type
        self.attributes: Dict[str, Any] = {}
        self.mentions = 0
        self.last_mentioned = None  # Timestamp
        
    def update_attribute(self, key: str, value: Any) -> None:
        """
        Update an attribute of the entity.
        
        Args:
            key: Attribute name
            value: Attribute value
        """
        self.attributes[key] = value
        
    def increment_mentions(self) -> None:
        """Increment the mention count and update timestamp."""
        self.mentions += 1
        self.last_mentioned = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of the entity
        """
        return {
            "name": self.name,
            "type": self.type,
            "attributes": self.attributes,
            "mentions": self.mentions,
            "last_mentioned": self.last_mentioned
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """
        Create an entity from a dictionary.
        
        Args:
            data: Dictionary with entity data
            
        Returns:
            Entity instance
        """
        entity = cls(data["name"], data["type"])
        entity.attributes = data["attributes"]
        entity.mentions = data["mentions"]
        entity.last_mentioned = data["last_mentioned"]
        return entity

class EntityMemory(BaseMemory):
    """
    Memory system for storing and retrieving entities.
    """
    
    def __init__(self, llm: LLM, file_path: Optional[str] = None):
        """
        Initialize entity memory.
        
        Args:
            llm: LLM instance for entity extraction
            file_path: Optional path to save/load entities
        """
        self.llm = llm
        self.file_path = file_path
        self.entities: Dict[str, Entity] = {}
        
        # Load from file if provided
        if file_path and os.path.exists(file_path):
            self._load_from_file()
    
    def add_entity(self, entity_data: Dict[str, Any]) -> None:
        """
        Add or update an entity.
        
        Args:
            entity_data: Dictionary with entity information
        """
        name = entity_data.get("name")
        entity_type = entity_data.get("type")
        attributes = entity_data.get("attributes", {})
        
        if not name or not entity_type:
            return
            
        # Create or update entity
        if name in self.entities:
            entity = self.entities[name]
            entity.increment_mentions()
            
            # Update attributes
            for key, value in attributes.items():
                entity.update_attribute(key, value)
        else:
            entity = Entity(name, entity_type)
            entity.increment_mentions()
            
            # Set attributes
            for key, value in attributes.items():
                entity.update_attribute(key, value)
                
            self.entities[name] = entity
            
        # Save to file if path provided
        if self.file_path:
            self._save_to_file()
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """
        Get an entity by name.
        
        Args:
            name: Entity name
            
        Returns:
            Entity or None if not found
        """
        return self.entities.get(name)
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of entities of the specified type
        """
        return [e for e in self.entities.values() if e.type.lower() == entity_type.lower()]
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using LLM.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        prompt = f"""
Extract important entities from the following text. Focus on:
- People (names, roles)
- Organizations
- Locations
- Dates and times
- Numerical values
- Key facts or attributes

For each entity, provide:
1. The entity name/value
2. The entity type
3. Any associated attributes

Text: {text}

Format as a JSON list of entity objects with "name", "type", and "attributes" fields.
Example:
[
  {{
    "name": "John Smith",
    "type": "person",
    "attributes": {{"role": "CEO", "company": "Acme Inc."}}
  }},
  {{
    "name": "Acme Inc.",
    "type": "organization",
    "attributes": {{"industry": "technology", "founded": "1999"}}
  }}
]
"""
        
        try:
            response = self.llm.generate_response(
                prompt=prompt, 
                max_tokens=500,
                json_mode=True
            )
            
            # Parse JSON response
            import json
            entities = json.loads(response)
            return entities
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for entities relevant to the query.
        
        Args:
            query: The search query
            
        Returns:
            List of relevant entities
        """
        # Use LLM to determine relevant entity types for the query
        prompt = f"""
For the following query, determine which types of entities would be most relevant:

Query: {query}

List the entity types (e.g., person, organization, location, date, etc.) that would be most relevant to answering this query.
"""
        
        response = self.llm.generate_response(prompt=prompt, max_tokens=100)
        
        # Extract entity types from response
        import re
        entity_types = re.findall(r'\b(\w+)\b', response.lower())
        
        # Filter common words
        common_words = {"the", "and", "or", "for", "to", "a", "an", "in", "on", "at", "by", "with", "about", "types", "entity", "entities", "would", "be", "relevant"}
        entity_types = [t for t in entity_types if t not in common_words and len(t) > 2]
        
        # Find entities matching these types
        matching_entities = []
        for entity in self.entities.values():
            if entity.type.lower() in entity_types:
                matching_entities.append(entity.to_dict())
                
        # Also include entities mentioned in the query
        query_terms = query.lower().split()
        for entity in self.entities.values():
            if entity.name.lower() in query.lower() and entity.to_dict() not in matching_entities:
                matching_entities.append(entity.to_dict())
                
        return matching_entities
    
    def _save_to_file(self) -> None:
        """Save entities to file."""
        if not self.file_path:
            return
            
        data = {name: entity.to_dict() for name, entity in self.entities.items()}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        
        # Save to file
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_from_file(self) -> None:
        """Load entities from file."""
        if not self.file_path or not os.path.exists(self.file_path):
            return
            
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for name, entity_data in data.items():
                self.entities[name] = Entity.from_dict(entity_data)
        except Exception as e:
            print(f"Error loading entities from file: {e}")
    
    # BaseMemory interface implementation
    def add_user_message(self, message: str) -> None:
        """
        Process a user message to extract entities.
        
        Args:
            message: The message content
        """
        # Extract entities from the message
        try:
            entities = self.extract_entities_from_text(message)
            
            # Add each entity
            for entity in entities:
                if isinstance(entity, dict) and "name" in entity and "type" in entity:
                    self.add_entity(entity)
        except Exception as e:
            print(f"Error processing entities from user message: {e}")
        
    def add_assistant_message(self, message: str) -> None:
        """
        Process an assistant message to extract entities.
        
        Args:
            message: The message content
        """
        # Extract entities from the message
        try:
            entities = self.extract_entities_from_text(message)
            
            # Add each entity
            for entity in entities:
                if isinstance(entity, dict) and "name" in entity and "type" in entity:
                    self.add_entity(entity)
        except Exception as e:
            print(f"Error processing entities from assistant message: {e}")
        
    def add_system_message(self, message: str) -> None:
        """
        Process a system message (no entity extraction for system messages).
        
        Args:
            message: The message content
        """
        # We don't extract entities from system messages
        pass
        
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Convert entities to message format for compatibility.
        
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Create a summary of important entities
        for name, entity in self.entities.items():
            if entity.mentions > 1:  # Only include frequently mentioned entities
                attr_str = ", ".join([f"{k}: {v}" for k, v in entity.attributes.items()])
                messages.append({
                    "role": "system",
                    "content": f"Entity: {name} (Type: {entity.type}), Attributes: {attr_str}"
                })
                
        return messages
    
    def get_chat_history(self) -> str:
        """
        Get a formatted string of entities (not applicable for entity memory).
        
        Returns:
            Empty string (not applicable)
        """
        return ""
    
    def clear(self) -> None:
        """Clear all entities."""
        self.entities = {}
        if self.file_path:
            self._save_to_file()
