"""
Enhanced agent with multi-memory capabilities.

This module provides an enhanced agent implementation that uses multiple
memory systems to improve context retention and response quality.
"""

from typing import Dict, Any, Optional, List, Callable, Union
from SimplerLLM.language.llm import LLM
from SimplerLLM.agents.memory_interface import BaseMemory
from SimplerLLM.agents.memory_manager import MemoryManager
from SimplerLLM.agents.tool_registry import ToolRegistry
from SimplerLLM.agents.brain import AgentBrain
from SimplerLLM.agents.models import AgentRole

class Agent:
    """
    Agent with enhanced memory capabilities.
    
    This agent uses multiple memory systems to improve context retention
    and response quality.
    """
    
    def __init__(
        self,
        llm: LLM,
        memory_manager: Optional[MemoryManager] = None,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the enhanced agent.
        
        Args:
            llm: LLM instance for generating responses
            memory_manager: Memory manager for coordinating memory systems
            tools: Dictionary of tools
            system_prompt: System prompt for the agent
            verbose: Whether to print verbose output
        """
        # Create memory manager if not provided
        self.memory_manager = memory_manager or MemoryManager(llm)
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        
        # Add tools if provided
        if tools:
            for name, tool in tools.items():
                self.tool_registry.register_tool(
                    name=name,
                    func=tool["func"],
                    description=tool["description"],
                    parameters=tool.get("parameters") or {}
                )
        
        # Set default system prompt if none provided
        default_system_prompt = """You are a helpful AI assistant with access to tools and excellent memory.
Follow these steps for each user request:
1. Think carefully about what the user is asking
2. Recall relevant information from previous conversations
3. Determine if you need to use a tool or can answer directly
4. If using a tool, specify which tool and what parameters to use
5. If answering directly, provide a helpful response"""

        # Set system prompt in all memory systems
        if system_prompt:
            self.memory_manager.add_system_message(system_prompt)
        else:
            self.memory_manager.add_system_message(default_system_prompt)
            
        # Create a memory adapter that provides the BaseMemory interface
        # but uses the memory manager internally
        memory_adapter = _MemoryManagerAdapter(self.memory_manager)
            
        # Initialize the brain
        self.brain = AgentBrain(
            llm=llm,
            memory=memory_adapter,
            tool_registry=self.tool_registry,
            verbose=verbose
        )
        
        self.llm = llm
        self.verbose = verbose
    
    def add_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a tool to the agent's toolkit.
        
        Args:
            name: Name of the tool
            func: Function to call when the tool is used
            description: Description of what the tool does
            parameters: Dictionary of parameter names to descriptions
        """
        self.tool_registry.register_tool(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {}
        )
    
    def run(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Process a user query with enhanced memory capabilities.
        
        Args:
            user_input: The user's input
            max_iterations: Maximum number of iterations
            
        Returns:
            The agent's response
        """
        # Add user input to memory manager
        self.memory_manager.add_user_message(user_input)
        
        # Get relevant memories
        memories = self.memory_manager.get_relevant_memories(user_input)
        
        # Enhance the query with memory context if available
        if memories.get("entity") and len(memories["entity"]) > 0:
            # Format entity information
            entity_info = "\n".join([
                f"Entity: {e['name']} (Type: {e['type']}), " +
                f"Attributes: {', '.join([f'{k}: {v}' for k, v in e['attributes'].items()])}"
                for e in memories["entity"]
            ])
            
            # Add entity information to system messages temporarily
            self.memory_manager.memories["conversation"].add_system_message(
                f"Relevant entities for this query:\n{entity_info}"
            )
        
        # Process the query using the brain
        response = self.brain.process_query(user_input, max_iterations)
        
        # Add assistant response to memory manager
        self.memory_manager.add_assistant_message(response)
        
        # Remove temporary entity information from system messages if added
        if memories.get("entity") and len(memories["entity"]) > 0:
            # Get system messages
            system_messages = [
                msg for msg in self.memory_manager.memories["conversation"].messages 
                if msg["role"] == "system" and msg["content"].startswith("Relevant entities for this query:")
            ]
            
            # Remove the entity info message
            if system_messages:
                self.memory_manager.memories["conversation"].messages.remove(system_messages[0])
        
        return response
    
    async def run_async(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Asynchronous version of the main execution loop.
        
        This is a placeholder for future implementation.
        
        Args:
            user_input: The user's input
            max_iterations: Maximum number of iterations
                
        Returns:
            The agent's response
        """
        # This is a placeholder for future async implementation
        # For now, just call the synchronous version
        return self.run(user_input, max_iterations)
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set a simple system prompt directly.
        
        Args:
            prompt: The system prompt to set
        """
        # Clear existing system messages
        for memory in self.memory_manager.memories.values():
            memory.clear()
            
        # Add new system prompt
        self.memory_manager.add_system_message(prompt)
        
    def set_role(self, role: Union[str, AgentRole]) -> None:
        """
        Set the agent's role.
        
        Args:
            role: Either a string prompt or an AgentRole object
        """
        if isinstance(role, str):
            self.set_system_prompt(role)
        else:
            # It's an AgentRole object
            self.set_system_prompt(role.system_prompt)
            # In the future, we could use more properties from the role


class _MemoryManagerAdapter(BaseMemory):
    """
    Adapter that provides the BaseMemory interface for the MemoryManager.
    
    This allows the AgentBrain to work with the MemoryManager as if it
    were a single memory system.
    """
    
    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize the adapter.
        
        Args:
            memory_manager: The memory manager to adapt
        """
        self.memory_manager = memory_manager
        
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to all memory systems.
        
        Args:
            message: The message content
        """
        self.memory_manager.add_user_message(message)
        
    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to all memory systems.
        
        Args:
            message: The message content
        """
        self.memory_manager.add_assistant_message(message)
        
    def add_system_message(self, message: str) -> None:
        """
        Add a system message to all memory systems.
        
        Args:
            message: The message content
        """
        self.memory_manager.add_system_message(message)
        
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get combined messages from all memory systems.
        
        Returns:
            Combined list of messages
        """
        return self.memory_manager.get_combined_messages()
    
    def get_chat_history(self) -> str:
        """
        Get a formatted string of the chat history.
        
        Returns:
            Formatted chat history from the conversation memory
        """
        if "conversation" in self.memory_manager.memories:
            return self.memory_manager.memories["conversation"].get_chat_history()
        return ""
    
    def clear(self) -> None:
        """Clear all memory systems."""
        self.memory_manager.clear()
        
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for relevant information across all memory systems.
        
        Args:
            query: The search query
            
        Returns:
            Combined list of relevant items
        """
        results = []

        # Get results from all memory systems
        all_results = self.memory_manager.get_relevant_memories(query)

        # Combine results
        for memory_type, memory_results in all_results.items():
            for result in memory_results:
                result["source"] = memory_type  # Add source information
                results.append(result)

        return results
