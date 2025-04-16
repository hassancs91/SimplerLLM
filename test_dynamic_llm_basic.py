import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our DynamicLLM implementation
from dynamic_llm import DynamicLLM, MultiLLMRequest, LLMConfig, Message
from SimplerLLM.language.llm import LLMProvider

class TestDynamicLLMBasic(unittest.TestCase):
    """Basic tests for the DynamicLLM class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.request = MultiLLMRequest(
            history=[
                Message(role="system", content="You are a helpful AI assistant"),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!")
            ],
            message="How are you?",
            llm_configs={
                "test_model": LLMConfig(
                    provider="OPENAI",
                    model="gpt-4o-mini",
                    temperature=0.7,
                    top_p=0.9
                )
            },
            max_tokens=100,
            top_p=0.95
        )
    
    @patch('SimplerLLM.language.llm.base.LLM.create')
    def test_chat_with_multiple_llms(self, mock_create):
        """Test the chat_with_multiple_llms method"""
        # Mock the LLM.create method
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "I'm doing well, thank you for asking!"
        mock_create.return_value = mock_llm
        
        # Call the method
        responses = DynamicLLM.chat_with_multiple_llms(self.request)
        
        # Assertions
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses["test_model"], "I'm doing well, thank you for asking!")
        
        # Verify the LLM.create was called with correct parameters
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        self.assertEqual(kwargs["provider"], getattr(LLMProvider, "OPENAI"))
        self.assertEqual(kwargs["model_name"], "gpt-4o-mini")
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_p"], 0.9)
        
        # Verify generate_response was called with correct parameters
        mock_llm.generate_response.assert_called_once()
        args, kwargs = mock_llm.generate_response.call_args
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertEqual(kwargs["top_p"], 0.9)
        
        # Check that the messages were formatted correctly
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 4)  # system + 2 history + 1 new message
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], "How are you?")
    
    @patch('SimplerLLM.language.llm.base.LLM.create')
    def test_multiple_llm_configs(self, mock_create):
        """Test handling multiple LLM configurations"""
        # Create a request with multiple LLM configs
        request = MultiLLMRequest(
            message="Hello",
            llm_configs={
                "model1": LLMConfig(provider="OPENAI", model="gpt-4o-mini"),
                "model2": LLMConfig(provider="ANTHROPIC", model="claude-3-haiku-20240307")
            }
        )
        
        # Mock the LLM.create method to return different responses for different models
        def create_side_effect(**kwargs):
            mock_llm = MagicMock()
            if kwargs["provider"] == getattr(LLMProvider, "OPENAI"):
                mock_llm.generate_response.return_value = "Response from OpenAI"
            else:
                mock_llm.generate_response.return_value = "Response from Anthropic"
            return mock_llm
        
        mock_create.side_effect = create_side_effect
        
        # Call the method
        responses = DynamicLLM.chat_with_multiple_llms(request)
        
        # Assertions
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses["model1"], "Response from OpenAI")
        self.assertEqual(responses["model2"], "Response from Anthropic")
        
        # Verify LLM.create was called twice
        self.assertEqual(mock_create.call_count, 2)
    
    def test_max_tokens_validator(self):
        """Test the max_tokens validator"""
        # Create a request with max_tokens > 2048
        request = MultiLLMRequest(
            message="Hello",
            llm_configs=LLMConfig(provider="OPENAI", model="gpt-4o-mini"),
            max_tokens=3000
        )
        
        # Check that max_tokens was capped at 2048
        self.assertEqual(request.max_tokens, 2048)

if __name__ == "__main__":
    unittest.main()
