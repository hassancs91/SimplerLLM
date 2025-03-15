"""
Example demonstrating a custom memory implementation with the SimplerLLM Agent.

This example shows how to create a custom memory system that extends the
BaseMemory interface and use it with the agent.
"""

from typing import List, Dict, Any, Optional
from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.agents import Agent, BaseMemory

class KeywordMemory(BaseMemory):
    """
    A custom memory implementation that adds keyword extraction.
    
    This memory system extracts keywords from messages and allows
    searching by keyword relevance.
    """
    
    def __init__(self, max_tokens: int = 4000):
        """Initialize the keyword memory."""
        self.messages: List[Dict[str, str]] = []
        self.max_tokens: int = max_tokens
        self.keywords: Dict[str, List[int]] = {}  # keyword -> list of message indices
        
    def add_user_message(self, message: str) -> None:
        """Add a user message to memory and extract keywords."""
        self.messages.append({"role": "user", "content": message})
        self._extract_keywords(message, len(self.messages) - 1)
        self._trim_if_needed()
        
    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to memory and extract keywords."""
        self.messages.append({"role": "assistant", "content": message})
        self._extract_keywords(message, len(self.messages) - 1)
        self._trim_if_needed()
        
    def add_system_message(self, message: str) -> None:
        """Add a system message to memory."""
        # Insert at beginning to maintain system message priority
        self.messages.insert(0, {"role": "system", "content": message})
        self._trim_if_needed()
        
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in memory."""
        return self.messages
    
    def get_chat_history(self) -> str:
        """Get a formatted string of the chat history."""
        history = []
        for msg in self.messages:
            if msg["role"] != "system":  # Skip system messages in the formatted history
                history.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(history)
    
    def clear(self) -> None:
        """Clear all messages from memory except system messages."""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages
        self.keywords = {}
        
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search memory for relevant information based on keywords.
        
        Args:
            query: The search query
            
        Returns:
            List of relevant memory items
        """
        # Extract keywords from the query
        query_keywords = self._simple_keyword_extraction(query)
        
        # Find messages that contain the keywords
        relevant_indices = set()
        for keyword in query_keywords:
            if keyword in self.keywords:
                relevant_indices.update(self.keywords[keyword])
        
        # Convert indices to messages
        results = []
        for idx in relevant_indices:
            if idx < len(self.messages):
                msg = self.messages[idx]
                if msg["role"] != "system":  # Skip system messages in search results
                    results.append({
                        "content": msg["content"],
                        "role": msg["role"],
                        "relevance": 1.0  # Simple relevance score
                    })
        
        return results
    
    def _extract_keywords(self, message: str, message_idx: int) -> None:
        """
        Extract keywords from a message and update the keyword index.
        
        Args:
            message: The message to extract keywords from
            message_idx: The index of the message in the messages list
        """
        keywords = self._simple_keyword_extraction(message)
        for keyword in keywords:
            if keyword not in self.keywords:
                self.keywords[keyword] = []
            self.keywords[keyword].append(message_idx)
    
    def _simple_keyword_extraction(self, text: str) -> List[str]:
        """
        Simple keyword extraction using common words filtering.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Remove common words and punctuation
        common_words = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "like", 
                        "from", "of", "as", "i", "you", "he", "she", "it", "we", "they"}
        
        keywords = []
        for word in words:
            # Remove punctuation
            word = word.strip(".,;:!?\"'()[]{}")
            # Check if it's a valid keyword
            if word and word not in common_words and len(word) > 2:
                keywords.append(word)
        
        return keywords
    
    def _trim_if_needed(self) -> None:
        """
        Trim oldest messages if exceeding max tokens.
        
        This uses a simple approximation where 4 characters ≈ 1 token.
        System messages are preserved if possible.
        """
        # Simple implementation - could be improved with actual token counting
        total_chars = sum(len(m["content"]) for m in self.messages)
        # Rough approximation: 4 chars ≈ 1 token
        while total_chars > self.max_tokens * 4 and len(self.messages) > 1:
            # Don't remove system messages if possible
            for i, msg in enumerate(self.messages):
                if msg["role"] != "system":
                    # Remove message from keyword index
                    for keyword, indices in list(self.keywords.items()):
                        self.keywords[keyword] = [idx for idx in indices if idx != i]
                        # Remove keyword if no more messages contain it
                        if not self.keywords[keyword]:
                            del self.keywords[keyword]
                    
                    # Remove message
                    removed = self.messages.pop(i)
                    total_chars -= len(removed["content"])
                    
                    # Update indices in keyword index
                    for keyword, indices in self.keywords.items():
                        self.keywords[keyword] = [idx if idx < i else idx - 1 for idx in indices]
                    
                    break
            else:
                # If we only have system messages left but still over the limit
                if self.messages and total_chars > self.max_tokens * 4:
                    removed = self.messages.pop(-1)  # Remove the oldest system message
                    total_chars -= len(removed["content"])


def main():
    # Create an LLM instance
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        temperature=0.7,
        verbose=False,
    )
    
    # Create a custom memory instance
    custom_memory = KeywordMemory(max_tokens=4000)
    
    # Create an agent with custom memory
    agent = Agent(llm=llm, memory=custom_memory, verbose=True)
    
    # Set a role that emphasizes memory capabilities
    agent.set_system_prompt(
        "You are a helpful assistant with excellent memory. When asked about previous "
        "parts of the conversation, refer back to what was discussed earlier."
    )
    
    # First interaction - introduce some information about programming
    print("\n--- First Interaction ---")
    query1 = "I'm learning Python programming and working on a machine learning project."
    print(f"User: {query1}")
    response1 = agent.run(query1)
    print(f"Agent: {response1}")
    
    # Second interaction - introduce information about a different topic
    print("\n--- Second Interaction ---")
    query2 = "I also enjoy hiking in the mountains on weekends."
    print(f"User: {query2}")
    response2 = agent.run(query2)
    print(f"Agent: {response2}")
    
    # Third interaction - ask about programming (should find relevant info)
    print("\n--- Third Interaction (Keyword Search Test: Programming) ---")
    query3 = "Tell me again what programming language I'm learning?"
    print(f"User: {query3}")
    response3 = agent.run(query3)
    print(f"Agent: {response3}")
    
    # Fourth interaction - ask about hiking (should find relevant info)
    print("\n--- Fourth Interaction (Keyword Search Test: Hiking) ---")
    query4 = "Where do I like to go hiking?"
    print(f"User: {query4}")
    response4 = agent.run(query4)
    print(f"Agent: {response4}")
    
    # Fifth interaction - ask about something not mentioned
    print("\n--- Fifth Interaction (Keyword Search Test: Not Mentioned) ---")
    query5 = "What's my favorite food?"
    print(f"User: {query5}")
    response5 = agent.run(query5)
    print(f"Agent: {response5}")
    
    # Print keyword index for demonstration
    print("\n--- Keyword Index ---")
    for keyword, indices in custom_memory.keywords.items():
        message_snippets = [custom_memory.messages[idx]["content"][:30] + "..." for idx in indices if idx < len(custom_memory.messages) and custom_memory.messages[idx]["role"] != "system"]
        print(f"Keyword: '{keyword}' -> Messages: {message_snippets}")

if __name__ == "__main__":
    main()
