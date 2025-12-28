"""
Chat API Endpoints
"""
from flask import Blueprint, request, jsonify
from services.llm_service import LLMService
from services.conversation_service import ConversationService

chat_bp = Blueprint('chat', __name__)

# Initialize services
llm_service = LLMService()
conversation_service = ConversationService()


@chat_bp.route('/chat', methods=['POST'])
def chat():
    """
    Send a message and get an LLM response.

    Request body:
    {
        "message": "Hello!",
        "provider": "openai",
        "model": "gpt-4o",
        "conversation_id": "optional-uuid",
        "settings": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant."
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        message = data.get('message', '').strip()
        if not message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400

        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')
        conversation_id = data.get('conversation_id')
        settings = data.get('settings', {})

        # Get or create conversation
        if conversation_id:
            conversation = conversation_service.get_conversation(conversation_id)
        else:
            conversation_id = conversation_service.create_conversation()
            conversation = conversation_service.get_conversation(conversation_id)

        # Add user message to conversation
        conversation_service.add_message(conversation_id, 'user', message)

        # Build messages list for the LLM
        messages = conversation_service.get_messages_for_llm(conversation_id)

        # Generate response
        response_data = llm_service.generate_response(
            provider=provider,
            model=model,
            messages=messages,
            system_prompt=settings.get('system_prompt', 'You are a helpful assistant.'),
            temperature=settings.get('temperature', 0.7),
            max_tokens=settings.get('max_tokens', 1000)
        )

        if response_data.get('success'):
            # Add assistant response to conversation
            conversation_service.add_message(
                conversation_id,
                'assistant',
                response_data['response']
            )

        return jsonify({
            'success': response_data.get('success', False),
            'response': response_data.get('response', ''),
            'conversation_id': conversation_id,
            'provider_used': provider,
            'model_used': model,
            'usage': response_data.get('usage', {}),
            'error': response_data.get('error')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chat_bp.route('/chat/<conversation_id>', methods=['DELETE'])
def clear_conversation(conversation_id):
    """Clear a conversation's history."""
    try:
        conversation_service.delete_conversation(conversation_id)
        return jsonify({'success': True, 'message': 'Conversation cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@chat_bp.route('/chat/<conversation_id>/history', methods=['GET'])
def get_history(conversation_id):
    """Get conversation history."""
    try:
        messages = conversation_service.get_messages(conversation_id)
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'messages': messages
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
