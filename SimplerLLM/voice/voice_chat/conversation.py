from typing import List, Dict, Optional
from .models import ConversationMessage, ConversationRole
from SimplerLLM.utils.custom_verbose import verbose_print


class ConversationManager:
    """
    Manages conversation history with automatic truncation.

    Handles:
    - Message storage
    - History length limits
    - Conversion to LLM message format
    - System message preservation
    """

    def __init__(self, max_length: int = 20, verbose: bool = False):
        """
        Initialize conversation manager.

        Args:
            max_length: Maximum number of messages to keep (excluding system message)
            verbose: Enable verbose logging
        """
        self.max_length = max_length
        self.verbose = verbose
        self.messages: List[ConversationMessage] = []

    def add_message(
        self,
        role: ConversationRole,
        content: str,
        audio_file: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add message to conversation history.

        Args:
            role: Message role (USER, ASSISTANT, SYSTEM)
            content: Message content
            audio_file: Optional path to audio file
            metadata: Optional metadata dictionary
        """
        message = ConversationMessage(
            role=role,
            content=content,
            audio_file=audio_file,
            metadata=metadata
        )

        self.messages.append(message)

        # Truncate if needed (keep system message + max_length recent messages)
        if len(self.messages) > self.max_length + 1:  # +1 for system message
            # Find system message (should be first)
            system_msg = None
            other_msgs = []

            for msg in self.messages:
                if msg.role == ConversationRole.SYSTEM:
                    system_msg = msg
                else:
                    other_msgs.append(msg)

            # Keep system message + last max_length messages
            if system_msg:
                self.messages = [system_msg] + other_msgs[-self.max_length:]
            else:
                self.messages = other_msgs[-self.max_length:]

            if self.verbose:
                verbose_print(
                    f"Truncated conversation history to {self.max_length} messages (+ system)",
                    "debug"
                )

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Convert conversation history to LLM message format.

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.messages
        ]

    def get_all_messages(self) -> List[ConversationMessage]:
        """
        Get all messages in conversation history.

        Returns:
            Copy of all messages
        """
        return self.messages.copy()

    def clear(self):
        """Clear all messages from conversation history."""
        self.messages = []
        if self.verbose:
            verbose_print("Conversation history cleared", "debug")

    def get_message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)

    def get_user_message_count(self) -> int:
        """Get number of user messages."""
        return sum(1 for msg in self.messages if msg.role == ConversationRole.USER)

    def get_assistant_message_count(self) -> int:
        """Get number of assistant messages."""
        return sum(1 for msg in self.messages if msg.role == ConversationRole.ASSISTANT)
