"""
Tests for the SimplerLLM Agent Builder.

This file contains tests for the Agent class and related functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.agents import Agent, AgentMemory, AgentAction, ToolCall

class MockLLM(LLM):
    """Mock LLM for testing."""
    
    def __init__(self, response_text="Mock response"):
        super().__init__(provider=LLMProvider.OPENAI, model_name="mock-model")
        self.response_text = response_text
        
    def generate_response(self, **kwargs):
        """Return the mock response."""
        return self.response_text

class TestAgentMemory(unittest.TestCase):
    """Tests for the AgentMemory class."""
    
    def test_add_messages(self):
        """Test adding messages to memory."""
        memory = AgentMemory()
        
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there")
        memory.add_system_message("System message")
        
        messages = memory.get_messages()
        
        # System message should be first
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System message")
        
        # User and assistant messages should follow
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello")
        
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Hi there")
        
    def test_memory_trimming(self):
        """Test that memory is trimmed when it exceeds the token limit."""
        memory = AgentMemory(max_tokens=10)  # Very small token limit for testing
        
        # Add a system message
        memory.add_system_message("System message")
        
        # Add a long user message that should trigger trimming
        memory.add_user_message("This is a very long message that should exceed the token limit")
        
        # The system message should still be there
        messages = memory.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        
        # Add another system message to test trimming of system messages
        memory.add_system_message("Another system message")
        memory.add_system_message("Yet another system message")
        
        # Should trim the oldest system message if we have too many
        messages = memory.get_messages()
        self.assertLess(len(messages), 4)  # Should have trimmed at least one message

class TestAgent(unittest.TestCase):
    """Tests for the Agent class."""
    
    @patch('SimplerLLM.language.llm_addons.generate_pydantic_json_model')
    def test_agent_run_with_direct_response(self, mock_generate):
        """Test agent run with a direct response (no tool call)."""
        # Create a mock response
        mock_response = AgentAction(
            thought="I should answer directly.",
            response="This is a direct response."
        )
        mock_generate.return_value = mock_response
        
        # Create an agent
        llm = MockLLM()
        agent = Agent(llm=llm)
        
        # Run the agent
        response = agent.run("Hello")
        
        # Check that the response is correct
        self.assertEqual(response, "This is a direct response.")
        
        # Check that generate_pydantic_json_model was called
        mock_generate.assert_called_once()
        
    @patch('SimplerLLM.language.llm_addons.generate_pydantic_json_model')
    def test_agent_run_with_tool_call(self, mock_generate):
        """Test agent run with a tool call."""
        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.return_value = "Tool result"
        
        # Create a mock response with a tool call, then a direct response
        tool_call_response = AgentAction(
            thought="I should use a tool.",
            action=ToolCall(
                tool_name="mock_tool",
                parameters={"param1": "value1"}
            )
        )
        
        direct_response = AgentAction(
            thought="Now I can answer directly.",
            response="Final response after tool use."
        )
        
        # Set up the mock to return the tool call response first, then the direct response
        mock_generate.side_effect = [tool_call_response, direct_response]
        
        # Create an agent with the mock tool
        llm = MockLLM()
        agent = Agent(llm=llm)
        agent.add_tool("mock_tool", mock_tool, "A mock tool", {"param1": "A parameter"})
        
        # Run the agent
        response = agent.run("Use a tool")
        
        # Check that the response is correct
        self.assertEqual(response, "Final response after tool use.")
        
        # Check that generate_pydantic_json_model was called twice
        self.assertEqual(mock_generate.call_count, 2)
        
        # Check that the tool was called with the correct parameters
        mock_tool.assert_called_once_with(param1="value1")
        
    def test_add_tool(self):
        """Test adding a tool to an agent."""
        llm = MockLLM()
        agent = Agent(llm=llm)
        
        # Add a tool
        def mock_tool(param1):
            return f"Tool called with {param1}"
        
        agent.add_tool(
            "mock_tool",
            mock_tool,
            "A mock tool",
            {"param1": "A parameter"}
        )
        
        # Check that the tool was added
        self.assertIn("mock_tool", agent.tools)
        self.assertEqual(agent.tools["mock_tool"]["func"], mock_tool)
        self.assertEqual(agent.tools["mock_tool"]["description"], "A mock tool")
        self.assertEqual(agent.tools["mock_tool"]["parameters"], {"param1": "A parameter"})

if __name__ == '__main__':
    unittest.main()
