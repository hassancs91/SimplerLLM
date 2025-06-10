"""
Memory manager for coordinating multiple memory systems.

This module provides a manager that coordinates between different memory systems
and provides a unified interface for the agent.
"""

from typing import Dict, Any, Optional, List
from SimplerLLM.language.llm import LLM
from SimplerLLM.agents.memory_interface import BaseMemory
from SimplerLLM.agents.memories.conversation_memory import ConversationMemory
from SimplerLLM.agents.memories.entity_memory import EntityMemory

class MemoryManager:
    """
    Manages multiple memory systems and coordinates between them.
    """
    
    def __init__(self, llm: LLM):
        """
        Initialize the memory manager with various memory systems.
        
        Args:
            llm: LLM instance for memory operations
        """
        self.llm = llm
        self.memories: Dict[str, BaseMemory] = {}
        
        # Initialize default memories
        self.add_memory("conversation", ConversationMemory(llm))
        self.add_memory("entity", EntityMemory(llm))
        
    def add_memory(self, name: str, memory: BaseMemory) -> None:
        """
        Add a memory system.
        
        Args:
            name: Name to identify the memory system
            memory: Memory system instance
        """
        self.memories[name] = memory
        
    def add_user_message(self, message: str) -> None:
        """
        Process a user message across all memory systems.
        
        Args:
            message: The message content
        """
        # Add to all memory systems
        for memory in self.memories.values():
            memory.add_user_message(message)
    
    def add_assistant_message(self, message: str) -> None:
        """
        Process an assistant message across all memory systems.
        
        Args:
            message: The message content
        """
        # Add to all memory systems
        for memory in self.memories.values():
            memory.add_assistant_message(message)
    
    def add_system_message(self, message: str) -> None:
        """
        Process a system message across all memory systems.
        
        Args:
            message: The message content
        """
        # Add to all memory systems
        for memory in self.memories.values():
            memory.add_system_message(message)
    
    def get_relevant_memories(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant information from all memory systems.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary with results from each memory system
        """
        results = {}
        
        # Get results from each memory system
        for name, memory in self.memories.items():
            results[name] = memory.search(query)
        
        return results
    
    def get_combined_messages(self) -> List[Dict[str, str]]:
        """
        Get combined messages from all memory systems.
        
        Returns:
            Combined list of messages
        """
        # Start with system messages
        system_messages = []
        for memory in self.memories.values():
            for msg in memory.get_messages():
                if msg["role"] == "system" and msg not in system_messages:
                    system_messages.append(msg)
        
        # Get conversation messages (non-system)
        conversation_messages = []
        if "conversation" in self.memories:
            conversation_messages = [
                msg for msg in self.memories["conversation"].get_messages()
                if msg["role"] != "system"
            ]
        
        # Combine messages
        return system_messages + conversation_messages
    
    def clear(self) -> None:
        """Clear all memory systems."""
        for memory in self.memories.values():
            memory.clear()
