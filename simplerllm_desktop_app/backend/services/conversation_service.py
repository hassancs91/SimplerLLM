"""
Conversation Service - Manages chat history
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime


class ConversationService:
    """Service for managing conversation history."""

    def __init__(self):
        # In-memory storage for conversations
        # In production, this could be persisted to a database
        self._conversations: Dict[str, Dict] = {}

    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        conversation_id = str(uuid.uuid4())
        self._conversations[conversation_id] = {
            'id': conversation_id,
            'created_at': datetime.now().isoformat(),
            'messages': []
        }
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """Add a message to a conversation."""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return False

        conversation['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages in a conversation."""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return []
        return conversation['messages']

    def get_messages_for_llm(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get messages formatted for the LLM API.
        Returns list of {'role': 'user'|'assistant', 'content': '...'}
        """
        messages = self.get_messages(conversation_id)
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in messages
        ]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    def clear_all(self):
        """Clear all conversations."""
        self._conversations.clear()
