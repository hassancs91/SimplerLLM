"""
Memory interface for agent systems.

This module defines the abstract base class for agent memory systems,
allowing for different implementations (simple history, vector DB, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseMemory(ABC):
    """Abstract base class for agent memory systems."""
    
    @abstractmethod
    def add_user_message(self, message: str) -> None:
        """Add a user message to memory."""
        pass
        
    @abstractmethod
    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to memory."""
        pass
        
    @abstractmethod
    def add_system_message(self, message: str) -> None:
        """Add a system message to memory."""
        pass
        
    @abstractmethod
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in memory."""
        pass
    
    @abstractmethod
    def get_chat_history(self) -> str:
        """Get a formatted string of the chat history."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory except system messages."""
        pass
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search memory for relevant information.
        
        Default implementation returns empty list. Override this method
        in implementations that support semantic search.
        
        Args:
            query: The search query
            
        Returns:
            List of relevant memory items
        """
        # Default implementation - no search capability
        return []
