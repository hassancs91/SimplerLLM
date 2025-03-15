"""
Tests for the new modular agent system.

This module contains tests for the new agent architecture with
modular components and decision tree workflow.
"""

import unittest
from unittest.mock import MagicMock, patch
from SimplerLLM.agents import (
    Agent, 
    AgentMemory, 
    ToolRegistry, 
    AgentBrain,
    RouterDecision,
    AgentRole,
    AgentAction,
    ToolCall
)

class TestNewAgentSystem(unittest.TestCase):
    """Test cases for the new agent system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock LLM
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = "This is a test response"
        
        # Create a simple test tool
        def echo_tool(message):
            return f"Echo: {message}"
        
        # Create agent with the mock LLM
        self.agent = Agent(llm=self.mock_llm, verbose=True)
        
        # Add a test tool
        self.agent.add_tool(
            name="echo",
            func=echo_tool,
            description="Echo back the input message",
            parameters={"message": "The message to echo back"}
        )
    
    def test_agent_initialization(self):
        """Test that the agent initializes correctly with all components."""
        self.assertIsInstance(self.agent.memory, AgentMemory)
        self.assertIsInstance(self.agent.tool_registry, ToolRegistry)
        self.assertIsInstance(self.agent.brain, AgentBrain)
        
        # Check that the tool was registered
        self.assertIn("echo", self.agent.tool_registry.list_tools())
        
        # Check that the system prompt was set
        system_messages = [msg for msg in self.agent.memory.get_messages() if msg["role"] == "system"]
        self.assertEqual(len(system_messages), 1)
    
    def test_set_system_prompt(self):
        """Test setting a custom system prompt."""
        custom_prompt = "This is a custom system prompt"
        self.agent.set_system_prompt(custom_prompt)
        
        # Check that the system prompt was updated
        system_messages = [msg for msg in self.agent.memory.get_messages() if msg["role"] == "system"]
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(system_messages[0]["content"], custom_prompt)
    
    def test_set_role_with_string(self):
        """Test setting a role using a string."""
        role_prompt = "You are a test assistant"
        self.agent.set_role(role_prompt)
        
        # Check that the system prompt was updated
        system_messages = [msg for msg in self.agent.memory.get_messages() if msg["role"] == "system"]
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(system_messages[0]["content"], role_prompt)
    
    def test_set_role_with_agent_role(self):
        """Test setting a role using an AgentRole object."""
        role = AgentRole(
            name="Test Role",
            description="A role for testing",
            system_prompt="You are a test assistant with a specific role",
            responsibilities=["Testing"],
            constraints=["Be concise"],
            allowed_tools=["echo"],
            priority_level=1,
            fallback_behavior="Default to simple responses"
        )
        
        self.agent.set_role(role)
        
        # Check that the system prompt was updated
        system_messages = [msg for msg in self.agent.memory.get_messages() if msg["role"] == "system"]
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(system_messages[0]["content"], role.system_prompt)
    
    @patch('SimplerLLM.agents.brain.AgentBrain._make_initial_decision')
    def test_direct_answer_workflow(self, mock_decision):
        """Test the direct answer workflow."""
        # Mock the decision to return direct_answer
        mock_decision.return_value = RouterDecision(
            decision_type="direct_answer",
            confidence=0.9,
            reasoning="This can be answered directly"
        )
        
        # Run the agent
        response = self.agent.run("What is 2+2?")
        
        # Check that the LLM was called to generate a response
        self.mock_llm.generate.assert_called_once()
        self.assertEqual(response, "This is a test response")
    
    @patch('SimplerLLM.agents.brain.AgentBrain._make_initial_decision')
    @patch('SimplerLLM.agents.brain.AgentBrain._select_tool')
    def test_tool_use_workflow(self, mock_select_tool, mock_decision):
        """Test the tool use workflow."""
        # Mock the decision to return use_tools
        mock_decision.return_value = RouterDecision(
            decision_type="use_tools",
            confidence=0.9,
            reasoning="This requires a tool"
        )
        
        # Mock the tool selection to return the echo tool
        mock_select_tool.return_value = AgentAction(
            thought="I should use the echo tool",
            action=ToolCall(
                tool_name="echo",
                parameters={"message": "Hello, world!"}
            )
        )
        
        # Run the agent
        response = self.agent.run("Echo 'Hello, world!'")
        
        # Check that the tool was called and the response was generated
        self.mock_llm.generate.assert_called_once()
        self.assertEqual(response, "This is a test response")

if __name__ == '__main__':
    unittest.main()
