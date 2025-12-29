"""
Compare Service - Handles parallel LLM calls for model comparison with conversation support.
"""
import uuid
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
from services.llm_service import LLMService


class CompareConversation:
    """Stores dual conversation state for model comparison."""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.messages_left: List[Dict] = []   # [{role, content, usage, timestamp}]
        self.messages_right: List[Dict] = []
        self.model_left: Optional[Dict] = None    # {provider, model}
        self.model_right: Optional[Dict] = None
        self.created_at = datetime.now().isoformat()

    def add_message(self, side: str, role: str, content: str, usage: Dict = None):
        """Add a message to one side of the conversation."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        if usage:
            message['usage'] = usage

        if side == 'left':
            self.messages_left.append(message)
        else:
            self.messages_right.append(message)

    def get_messages_for_llm(self, side: str) -> List[Dict[str, str]]:
        """Get messages formatted for the LLM API."""
        messages = self.messages_left if side == 'left' else self.messages_right
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in messages
        ]

    def to_dict(self) -> Dict:
        """Serialize conversation to dict."""
        return {
            'id': self.id,
            'created_at': self.created_at,
            'model_left': self.model_left,
            'model_right': self.model_right,
            'messages_left': self.messages_left,
            'messages_right': self.messages_right
        }


class CompareService:
    """Service for running parallel LLM comparisons with real-time streaming."""

    def __init__(self):
        self.llm_service = LLMService()
        self.conversations: Dict[str, CompareConversation] = {}

    def create_conversation(self) -> str:
        """Create a new compare conversation and return its ID."""
        conversation = CompareConversation()
        self.conversations[conversation.id] = conversation
        return conversation.id

    def get_conversation(self, conversation_id: str) -> Optional[CompareConversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    def run_compare_streaming(
        self,
        message: str,
        models: List[Dict[str, str]],
        conversation_id: str = None,
        settings: Dict = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run parallel LLM calls and yield events as they occur.

        Args:
            message: The user message to send to both models
            models: List of exactly 2 dicts with 'provider' and 'model' keys
            conversation_id: Optional existing conversation ID
            settings: Dict with temperature, max_tokens, system_prompt

        Yields:
            Dict events with types: 'start', 'model_start', 'model_complete',
            'complete', 'error'
        """
        settings = settings or {}
        temperature = settings.get('temperature', 0.7)
        max_tokens = settings.get('max_tokens', 1000)
        system_prompt = settings.get('system_prompt', 'You are a helpful assistant.')

        # Get or create conversation
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
        else:
            conversation = CompareConversation()
            self.conversations[conversation.id] = conversation

        # Store model configurations
        conversation.model_left = models[0]
        conversation.model_right = models[1]

        # Add user message to both conversations
        conversation.add_message('left', 'user', message)
        conversation.add_message('right', 'user', message)

        # Event queue for communication between threads and generator
        event_queue = queue.Queue()

        def call_model(model_index: int, model_config: Dict, side: str):
            """Call a single model and put results in queue."""
            provider = model_config['provider']
            model = model_config['model']

            try:
                start_time = time.time()

                # Get conversation history for this side
                messages = conversation.get_messages_for_llm(side)

                # Generate response
                result = self.llm_service.generate_response(
                    provider=provider,
                    model=model,
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                execution_time = time.time() - start_time

                if result['success']:
                    response_text = result['response']
                    usage = result.get('usage', {})

                    # Add assistant response to conversation
                    conversation.add_message(side, 'assistant', response_text, usage)

                    event_queue.put({
                        'type': 'model_complete',
                        'model_index': model_index,
                        'side': side,
                        'provider': provider,
                        'model': model,
                        'response': response_text,
                        'usage': usage,
                        'execution_time': round(execution_time, 2)
                    })
                else:
                    event_queue.put({
                        'type': 'model_error',
                        'model_index': model_index,
                        'side': side,
                        'provider': provider,
                        'model': model,
                        'error': result.get('error', 'Unknown error'),
                        'execution_time': round(execution_time, 2)
                    })

            except Exception as e:
                event_queue.put({
                    'type': 'model_error',
                    'model_index': model_index,
                    'side': side,
                    'provider': provider,
                    'model': model,
                    'error': str(e)
                })

        # Yield start event
        yield {
            'type': 'start',
            'conversation_id': conversation.id,
            'timestamp': time.time(),
            'total_models': 2
        }

        # Yield model_start events
        for idx, model_config in enumerate(models):
            side = 'left' if idx == 0 else 'right'
            yield {
                'type': 'model_start',
                'model_index': idx,
                'side': side,
                'provider': model_config['provider'],
                'model': model_config['model']
            }

        # Start both model calls in parallel threads
        threads = []
        for idx, model_config in enumerate(models):
            side = 'left' if idx == 0 else 'right'
            thread = threading.Thread(
                target=call_model,
                args=(idx, model_config, side)
            )
            thread.start()
            threads.append(thread)

        # Wait for both results
        results_received = 0
        start_time = time.time()

        while results_received < 2:
            try:
                event = event_queue.get(timeout=300)  # 5 minute timeout
                yield event

                if event['type'] in ('model_complete', 'model_error'):
                    results_received += 1

            except queue.Empty:
                yield {
                    'type': 'error',
                    'error': 'Model comparison timed out'
                }
                break

        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=5)

        total_time = time.time() - start_time

        # Yield complete event
        yield {
            'type': 'complete',
            'conversation_id': conversation.id,
            'total_time': round(total_time, 2)
        }

    def run_compare(
        self,
        message: str,
        models: List[Dict[str, str]],
        conversation_id: str = None,
        settings: Dict = None
    ) -> Dict[str, Any]:
        """
        Run parallel LLM calls and return complete result.
        Non-streaming fallback endpoint.
        """
        responses = []
        error = None
        final_conversation_id = None
        total_time = 0

        for event in self.run_compare_streaming(message, models, conversation_id, settings):
            if event['type'] == 'start':
                final_conversation_id = event['conversation_id']
            elif event['type'] == 'model_complete':
                responses.append({
                    'model_index': event['model_index'],
                    'side': event['side'],
                    'provider': event['provider'],
                    'model': event['model'],
                    'response': event['response'],
                    'usage': event.get('usage', {}),
                    'execution_time': event.get('execution_time', 0)
                })
            elif event['type'] == 'model_error':
                responses.append({
                    'model_index': event['model_index'],
                    'side': event['side'],
                    'provider': event['provider'],
                    'model': event['model'],
                    'error': event['error'],
                    'execution_time': event.get('execution_time', 0)
                })
            elif event['type'] == 'complete':
                total_time = event['total_time']
            elif event['type'] == 'error':
                error = event['error']

        if error:
            return {
                'success': False,
                'error': error
            }

        # Sort responses by model_index
        responses.sort(key=lambda x: x['model_index'])

        return {
            'success': True,
            'conversation_id': final_conversation_id,
            'responses': responses,
            'total_time': total_time
        }


# Singleton instance
compare_service = CompareService()
