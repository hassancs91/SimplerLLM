"""
Tool Registry for Mini Agent Flows

Maps tool names to actual tool functions from SimplerLLM.tools
"""

from SimplerLLM.tools.python_func import execute_python_code
from SimplerLLM.tools.file_functions import save_text_to_file
from SimplerLLM.tools.json_helpers import (
    extract_json_from_text,
    validate_json_with_pydantic_model,
    convert_json_to_pydantic_model,
    generate_json_example_from_pydantic,
)
from SimplerLLM.tools.youtube import get_youtube_transcript, get_youtube_transcript_with_timing
from SimplerLLM.tools.serp import (
    search_with_serper_api,
    search_with_value_serp,
    search_with_duck_duck_go,
)
from SimplerLLM.tools.file_loader import read_csv_file
from SimplerLLM.tools.text_chunker import (
    chunk_by_max_chunk_size,
    chunk_by_sentences,
    chunk_by_paragraphs,
)
from SimplerLLM.tools.brainstorm import (
    recursive_brainstorm_tool,
    simple_brainstorm,
)


class ToolRegistry:
    """Registry of available tools for Mini Agent flows."""

    TOOLS = {
        # Python execution
        "execute_python_code": execute_python_code,

        # File operations
        "save_text_to_file": save_text_to_file,
        "read_csv_file": read_csv_file,

        # JSON operations
        "extract_json_from_text": extract_json_from_text,
        "validate_json_with_pydantic_model": validate_json_with_pydantic_model,
        "convert_json_to_pydantic_model": convert_json_to_pydantic_model,
        "generate_json_example_from_pydantic": generate_json_example_from_pydantic,

        # YouTube tools
        "youtube_transcript": get_youtube_transcript,
        "youtube_transcript_with_timing": get_youtube_transcript_with_timing,

        # Search tools
        "web_search_serper": search_with_serper_api,
        "web_search_value_serp": search_with_value_serp,
        "web_search_duckduckgo": search_with_duck_duck_go,

        # Text chunking
        "chunk_by_max_size": chunk_by_max_chunk_size,
        "chunk_by_sentences": chunk_by_sentences,
        "chunk_by_paragraphs": chunk_by_paragraphs,

        # Brainstorming tools
        "recursive_brainstorm": recursive_brainstorm_tool,
        "simple_brainstorm": simple_brainstorm,
    }

    @classmethod
    def get_tool(cls, tool_name: str):
        """
        Get a tool function by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            The tool function

        Raises:
            ValueError: If the tool name is not found in the registry
        """
        if tool_name not in cls.TOOLS:
            available_tools = ", ".join(cls.TOOLS.keys())
            raise ValueError(
                f"Tool '{tool_name}' not found in registry. "
                f"Available tools: {available_tools}"
            )
        return cls.TOOLS[tool_name]

    @classmethod
    def list_tools(cls):
        """Return a list of all available tool names."""
        return list(cls.TOOLS.keys())

    @classmethod
    def register_tool(cls, name: str, func):
        """
        Register a custom tool.

        Args:
            name: Name for the tool
            func: The tool function to register
        """
        cls.TOOLS[name] = func
