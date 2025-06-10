"""
Tool registry for agent systems.

This module provides a registry for managing agent tools, including
registration, retrieval, and execution.
"""

from typing import Dict, Any, Optional, List, Callable

class ToolRegistry:
    """Registry for managing agent tools."""
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a tool with the registry.
        
        Args:
            name: Name of the tool
            func: Function to call when the tool is used
            description: Description of what the tool does
            parameters: Dictionary of parameter names to descriptions
        """
        self.tools[name] = {
            "func": func,
            "description": description,
            "parameters": parameters or {}
        }
        
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            Tool dictionary or None if not found
        """
        return self.tools.get(name)
        
    def list_tools(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
        
    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools.
        
        Returns:
            Formatted string with tool descriptions
        """
        tool_descriptions = []
        for name, tool in self.tools.items():
            params_str = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            tool_descriptions.append(
                f"Tool: {name}\nDescription: {tool['description']}\nParameters: {params_str}"
            )
        
        return "\n\n".join(tool_descriptions) if tool_descriptions else "No tools available."
        
    def execute_tool(self, name: str, **parameters) -> Any:
        """
        Execute a tool with the given parameters.
        
        Args:
            name: Name of the tool to execute
            **parameters: Parameters to pass to the tool
            
        Returns:
            Result of tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
            
        try:
            return tool["func"](**parameters)
        except Exception as e:
            return f"Error executing tool {name}: {str(e)}"
