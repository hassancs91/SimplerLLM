"""
Compare API Endpoints with Server-Sent Events (SSE) support.
"""
import json
from flask import Blueprint, request, jsonify, Response
from services.compare_service import compare_service

compare_bp = Blueprint('compare', __name__)


@compare_bp.route('/compare/stream', methods=['POST'])
def stream_compare():
    """
    Send a message to two models simultaneously with SSE streaming.

    Request body:
    {
        "message": "Hello!",
        "models": [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"}
        ],
        "conversation_id": "optional-uuid",
        "settings": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant."
        }
    }

    SSE Events:
    - start: Session started with conversation_id
    - model_start: Model is generating (model_index: 0 or 1, side: left or right)
    - model_complete: Model finished with response, usage, execution_time
    - model_error: Model encountered an error
    - complete: Both models finished
    - error: General error occurred
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        message = data.get('message', '').strip()
        if not message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400

        models = data.get('models', [])
        if not models or len(models) != 2:
            return jsonify({'success': False, 'error': 'Exactly 2 models are required'}), 400

        # Validate model configurations
        for idx, model_config in enumerate(models):
            if not model_config.get('provider') or not model_config.get('model'):
                return jsonify({
                    'success': False,
                    'error': f'Model {idx + 1} must have provider and model specified'
                }), 400

        conversation_id = data.get('conversation_id')
        settings = data.get('settings', {})

        def generate():
            """Generator that yields SSE events."""
            for event in compare_service.run_compare_streaming(
                message=message,
                models=models,
                conversation_id=conversation_id,
                settings=settings
            ):
                yield f"data: {json.dumps(event)}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@compare_bp.route('/compare/run', methods=['POST'])
def run_compare():
    """
    Run model comparison without streaming (fallback).

    Returns the complete result when both models finish.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        message = data.get('message', '').strip()
        if not message:
            return jsonify({'success': False, 'error': 'Message is required'}), 400

        models = data.get('models', [])
        if not models or len(models) != 2:
            return jsonify({'success': False, 'error': 'Exactly 2 models are required'}), 400

        # Validate model configurations
        for idx, model_config in enumerate(models):
            if not model_config.get('provider') or not model_config.get('model'):
                return jsonify({
                    'success': False,
                    'error': f'Model {idx + 1} must have provider and model specified'
                }), 400

        conversation_id = data.get('conversation_id')
        settings = data.get('settings', {})

        result = compare_service.run_compare(
            message=message,
            models=models,
            conversation_id=conversation_id,
            settings=settings
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@compare_bp.route('/compare/<conversation_id>', methods=['DELETE'])
def delete_compare_conversation(conversation_id):
    """Clear a compare conversation history."""
    try:
        success = compare_service.delete_conversation(conversation_id)

        if success:
            return jsonify({
                'success': True,
                'message': 'Conversation cleared'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Conversation not found'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@compare_bp.route('/compare/<conversation_id>/history', methods=['GET'])
def get_compare_history(conversation_id):
    """Get conversation history for both models."""
    try:
        conversation = compare_service.get_conversation(conversation_id)

        if not conversation:
            return jsonify({
                'success': False,
                'error': 'Conversation not found'
            }), 404

        return jsonify({
            'success': True,
            'conversation': conversation.to_dict()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
