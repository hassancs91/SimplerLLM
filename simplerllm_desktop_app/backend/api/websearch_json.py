"""
Web Search + JSON API Routes
"""
from flask import Blueprint, request, jsonify
from services.websearch_json_service import websearch_json_service

websearch_json_bp = Blueprint('websearch_json', __name__)


@websearch_json_bp.route('/websearch-json/providers', methods=['GET'])
def get_providers():
    """Get list of providers that support web search."""
    try:
        providers = websearch_json_service.get_web_search_providers()
        return jsonify({
            'success': True,
            'providers': providers
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@websearch_json_bp.route('/websearch-json/generate', methods=['POST'])
def generate():
    """
    Generate structured JSON from web search.

    Request body:
    {
        "prompt": "Search query/prompt",
        "provider": "openai" or "perplexity",
        "model": "gpt-4o" or "sonar-pro",
        "schema_mode": "form" or "code",
        "schema": {  // Required if schema_mode is "form"
            "fields": [
                {"name": "field_name", "type": "string|number|boolean|list", "description": "...", "item_type": "string"}
            ]
        },
        "schema_code": "...",  // Required if schema_mode is "code" - Pydantic model definition
        "settings": {
            "temperature": 0.7,
            "max_tokens": 2000
        }
    }

    Response:
    {
        "success": true,
        "result": {
            "data": {...},
            "sources": [{"title": "...", "url": "..."}],
            "tokens": {"input": N, "output": N},
            "process_time": N.NN,
            "provider_used": "...",
            "model_used": "..."
        }
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400

        provider = data.get('provider', '').strip()
        if not provider:
            return jsonify({
                'success': False,
                'error': 'Provider is required'
            }), 400

        model = data.get('model', '').strip()
        if not model:
            return jsonify({
                'success': False,
                'error': 'Model is required'
            }), 400

        # Get schema mode (default to "form" for backward compatibility)
        schema_mode = data.get('schema_mode', 'form').strip()
        schema = data.get('schema', {})
        schema_code = data.get('schema_code', '').strip()

        # Validate based on mode
        if schema_mode == 'code':
            if not schema_code:
                return jsonify({
                    'success': False,
                    'error': 'Pydantic model code is required when using code mode'
                }), 400
        else:
            # Form mode
            if not schema.get('fields'):
                return jsonify({
                    'success': False,
                    'error': 'Schema with at least one field is required'
                }), 400

        settings = data.get('settings', {})

        # Generate structured output with web search
        result = websearch_json_service.generate_with_web_search(
            prompt=prompt,
            provider=provider,
            model=model,
            schema=schema if schema_mode == 'form' else None,
            schema_code=schema_code if schema_mode == 'code' else None,
            schema_mode=schema_mode,
            settings=settings
        )

        return jsonify({
            'success': True,
            'result': result
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Generation failed: {str(e)}'
        }), 500


@websearch_json_bp.route('/websearch-json/code-example', methods=['POST'])
def get_code_example():
    """
    Generate Python code example for the current configuration.

    Request body: Same as /generate endpoint
    Response: { "success": true, "code": "..." }
    """
    try:
        data = request.get_json()

        prompt = data.get('prompt', '').strip()
        provider = data.get('provider', 'openai').strip()
        model = data.get('model', 'gpt-4o').strip()
        schema = data.get('schema', {'fields': []})

        code = websearch_json_service.generate_code_example(
            prompt=prompt,
            schema=schema,
            provider=provider,
            model=model
        )

        return jsonify({
            'success': True,
            'code': code
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
