"""
Conversation memory implementation with summarization capabilities.

This module provides a memory system that stores conversation history and
can summarize it when it exceeds a certain length.
"""

import time
from typing import List, Dict, Any, Optional
from SimplerLLM.language.llm import LLM
from SimplerLLM.agents.memory_interface import BaseMemory

class ConversationMemory(BaseMemory):
    """
    Memory system for storing and summarizing conversation history.
    """
    
    def __init__(self, llm: LLM, max_tokens: int = 4000, summarize_threshold: int = 3000):
        """
        Initialize conversation memory.
        
        Args:
            llm: LLM instance for summarization
            max_tokens: Maximum number of tokens to store
            summarize_threshold: Token count that triggers summarization
        """
        self.llm = llm
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self.messages: List[Dict[str, str]] = []
        self.summary = ""
        self.last_summarized = 0  # Index of last summarized message
        
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to memory.
        
        Args:
            message: The message content
        """
        self.messages.append({"role": "user", "content": message})
        self._check_and_summarize()
        
    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to memory.
        
        Args:
            message: The message content
        """
        self.messages.append({"role": "assistant", "content": message})
        self._check_and_summarize()
        
    def add_system_message(self, message: str) -> None:
        """
        Add a system message to memory.
        
        Args:
            message: The message content
        """
        # Check if we already have a system message with the same content
        for msg in self.messages:
            if msg["role"] == "system" and msg["content"] == message:
                return  # Skip if identical system message exists
                
        # Insert at beginning to maintain system message priority
        self.messages.insert(0, {"role": "system", "content": message})
        
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in memory.
        
        Returns:
            List of message dictionaries
        """
        if self.summary:
            # If we have a summary, insert it at the beginning (after system messages)
            system_messages = [m for m in self.messages if m["role"] == "system"]
            other_messages = [m for m in self.messages if m["role"] != "system"]
            
            summary_message = {"role": "system", "content": f"Conversation summary: {self.summary}"}
            
            return system_messages + [summary_message] + other_messages
        else:
            return self.messages
    
    def get_chat_history(self) -> str:
        """
        Get a formatted string of the chat history.
        
        Returns:
            Formatted chat history
        """
        history = []
        
        # Add summary if available
        if self.summary:
            history.append(f"Summary: {self.summary}\n")
            
        # Add messages (excluding system messages)
        for msg in self.messages:
            if msg["role"] != "system":
                history.append(f"{msg['role'].capitalize()}: {msg['content']}")
                
        return "\n".join(history)
    
    def clear(self) -> None:
        """Clear all messages from memory except system messages."""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages
        self.summary = ""
        self.last_summarized = 0
        
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for relevant messages.
        
        Args:
            query: The search query
            
        Returns:
            List of relevant messages
        """
        results = []
        for msg in self.messages:
            if msg["role"] != "system":
                # Use LLM to determine relevance
                prompt = f"""
Determine how relevant the following message is to the query.
Provide a relevance score between 0.0 (not relevant) and 1.0 (highly relevant). Just output the score.

Query: {query}
Message: {msg['content']}

Relevance score:"""
                try:
                    relevance_str = self.llm.generate_response(prompt=prompt, max_tokens=50)
                    relevance = float(relevance_str.strip())
                    if 0.0 <= relevance <= 1.0:
                        results.append({
                            "content": msg["content"],
                            "role": msg["role"],
                            "relevance": relevance
                        })
                except Exception as e:
                    print(f"Error determining relevance: {e}")
                    # If there's an error, assume it's not relevant
                    pass
                
        # Also include summary if it contains the query
        if self.summary and query.lower() in self.summary.lower():
            results.append({
                "content": f"From summary: {self.summary}",
                "role": "system",
                "relevance": 0.8
            })
            
        return results
    
    def _check_and_summarize(self) -> None:
        """Check if summarization is needed and summarize if necessary."""
        # Count tokens in non-system messages
        non_system_messages = [m for m in self.messages if m["role"] != "system"]
        token_count = sum(len(m["content"]) for m in non_system_messages) // 4  # Rough estimate
        
        if token_count > self.summarize_threshold:
            self._summarize_conversation()
            
    def _summarize_conversation(self) -> None:
        """Summarize the conversation and remove older messages."""
        # Get non-system messages for summarization
        non_system_messages = [m for m in self.messages if m["role"] != "system"]
        
        # Keep the most recent messages
        keep_count = min(10, len(non_system_messages) // 3)
        keep_count = max(keep_count, 3)  # Always keep at least 3 messages
        recent_messages = non_system_messages[-keep_count:] if keep_count > 0 else []
        
        # Messages to summarize (only those not previously summarized)
        to_summarize = non_system_messages[self.last_summarized:-keep_count] if keep_count > 0 else non_system_messages[self.last_summarized:]
        
        if not to_summarize:
            return
            
        # Format conversation for summarization
        conversation = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in to_summarize
        ])
        
        # Create summarization prompt
        prompt = f"""
Summarize the following conversation, extracting the most important information, 
key facts, and any decisions or conclusions reached. Maintain the essential context 
needed for future conversation.

Previous summary: {self.summary if self.summary else "None"}

Conversation to summarize:
{conversation}

Provide a comprehensive summary that captures all critical information.
"""
        
        # Generate summary
        new_summary = self.llm.generate_response(prompt=prompt, max_tokens=800)
        
        # Update summary
        if self.summary:
            self.summary = f"{self.summary}\n\nUpdated with: {new_summary}"
        else:
            self.summary = new_summary
            
        # Update last summarized index
        self.last_summarized = len(non_system_messages) - keep_count
            
        # Update messages - keep system messages and recent messages
        system_messages = [m for m in self.messages if msg["role"] == "system"]
        self.messages = system_messages + recent_messages
