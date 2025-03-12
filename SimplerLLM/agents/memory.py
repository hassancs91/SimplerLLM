from typing import List, Dict, Any, Optional

class AgentMemory:
    """
    Memory system for agents to store conversation history and other state.
    
    This class provides methods for adding and retrieving messages, as well as
    managing memory size to prevent token limits from being exceeded.
    """
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize the agent memory.
        
        Args:
            max_tokens (int): Maximum number of tokens to store in memory.
                              This is an approximate limit based on character count.
        """
        self.messages: List[Dict[str, str]] = []
        self.max_tokens: int = max_tokens
        self.metadata: Dict[str, Any] = {}  # For storing additional information
        
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to memory.
        
        Args:
            message (str): The message content.
        """
        self.messages.append({"role": "user", "content": message})
        self._trim_if_needed()
        
    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to memory.
        
        Args:
            message (str): The message content.
        """
        self.messages.append({"role": "assistant", "content": message})
        self._trim_if_needed()
        
    def add_system_message(self, message: str) -> None:
        """
        Add a system message to memory.
        
        System messages are inserted at the beginning to maintain their priority.
        
        Args:
            message (str): The message content.
        """
        # Insert at beginning to maintain system message priority
        self.messages.insert(0, {"role": "system", "content": message})
        self._trim_if_needed()
        
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in memory.
        
        Returns:
            List[Dict[str, str]]: List of message dictionaries.
        """
        return self.messages
    
    def get_chat_history(self) -> str:
        """
        Get a formatted string of the chat history.
        
        Returns:
            str: Formatted chat history.
        """
        history = []
        for msg in self.messages:
            if msg["role"] != "system":  # Skip system messages in the formatted history
                history.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(history)
    
    def clear(self) -> None:
        """
        Clear all messages from memory except system messages.
        """
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages
        
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata value.
        
        Args:
            key (str): Metadata key.
            value (Any): Metadata value.
        """
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value.
        
        Args:
            key (str): Metadata key.
            default (Any, optional): Default value if key doesn't exist.
            
        Returns:
            Any: The metadata value or default.
        """
        return self.metadata.get(key, default)
        
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
                    removed = self.messages.pop(i)
                    total_chars -= len(removed["content"])
                    break
            else:
                # If we only have system messages left but still over the limit
                if self.messages and total_chars > self.max_tokens * 4:
                    removed = self.messages.pop(-1)  # Remove the oldest system message
                    total_chars -= len(removed["content"])
