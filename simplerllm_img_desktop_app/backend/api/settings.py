"""
Settings API - Endpoints for managing application settings
"""
from flask import Blueprint, request, jsonify
from services.shared import settings_service, image_service

settings_bp = Blueprint('settings', __name__)


@settings_bp.route('/settings', methods=['GET'])
def get_settings():
    """
    Get current settings (excluding API keys for security).

    Response:
        {
            "success": true,
            "settings": {
                "default_provider": "google",
                "default_model": "gemini-2.5-flash-image-preview",
                "default_settings": { ... },
                "has_api_keys": {"google": true}
            }
        }
    """
    try:
        settings = settings_service.get_settings()

        # Don't expose actual API keys, just whether they're configured
        has_api_keys = {}
        for provider in ['google']:
            has_api_keys[provider] = settings_service.has_api_key(provider)

        # Remove actual keys from response
        safe_settings = {k: v for k, v in settings.items() if k != 'api_keys'}
        safe_settings['has_api_keys'] = has_api_keys

        return jsonify({
            'success': True,
            'settings': safe_settings
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.route('/settings', methods=['POST'])
def save_settings():
    """
    Save settings (excluding API keys - use dedicated endpoint).

    Request body:
        {
            "default_provider": "google",
            "default_model": "gemini-2.5-flash-image-preview"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Only allow specific settings to be updated
        allowed_keys = ['default_provider', 'default_model', 'default_settings']
        for key in allowed_keys:
            if key in data:
                settings_service.set_setting(key, data[key])

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.route('/settings/api-key/<provider>', methods=['POST'])
def set_api_key(provider):
    """
    Set API key for a provider.

    Request body:
        {
            "api_key": "your-api-key"
        }
    """
    try:
        data = request.get_json()
        if not data or 'api_key' not in data:
            return jsonify({'success': False, 'error': 'API key is required'}), 400

        api_key = data['api_key'].strip()
        if not api_key:
            return jsonify({'success': False, 'error': 'API key cannot be empty'}), 400

        # Validate the API key
        validation = image_service.validate_api_key(api_key, provider)
        if not validation['valid']:
            return jsonify({
                'success': False,
                'error': validation.get('error', 'Invalid API key')
            }), 400

        # Save the key
        settings_service.set_api_key(provider, api_key)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.route('/settings/api-key/<provider>', methods=['DELETE'])
def remove_api_key(provider):
    """Remove API key for a provider."""
    try:
        settings_service.remove_api_key(provider)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
