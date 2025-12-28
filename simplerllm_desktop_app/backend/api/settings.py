"""
Settings API Endpoints
"""
from flask import Blueprint, request, jsonify
from services.settings_service import SettingsService

settings_bp = Blueprint('settings', __name__)

settings_service = SettingsService()


@settings_bp.route('/settings', methods=['GET'])
def get_settings():
    """Get current settings (API keys are returned as booleans for security)."""
    settings = settings_service.get_settings()

    # Convert API keys to booleans (whether they're set or not)
    api_keys_status = {}
    for key, value in settings.get('api_keys', {}).items():
        api_keys_status[key] = bool(value)

    return jsonify({
        'api_keys': api_keys_status,
        'default_provider': settings.get('default_provider', 'openai'),
        'default_model': settings.get('default_model', 'gpt-4o'),
        'default_settings': settings.get('default_settings', {
            'temperature': 0.7,
            'max_tokens': 1000,
            'system_prompt': 'You are a helpful assistant.'
        })
    })


@settings_bp.route('/settings', methods=['POST'])
def save_settings():
    """Save settings."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Handle API keys
        api_keys = data.get('api_keys', {})
        for provider, key in api_keys.items():
            if key is None:
                # Remove the API key
                settings_service.remove_api_key(provider)
            elif key:
                # Set the API key
                settings_service.set_api_key(provider, key)

        # Handle other settings
        if 'default_provider' in data:
            settings_service.set_setting('default_provider', data['default_provider'])

        if 'default_model' in data:
            settings_service.set_setting('default_model', data['default_model'])

        if 'default_settings' in data:
            settings_service.set_setting('default_settings', data['default_settings'])

        return jsonify({'success': True, 'message': 'Settings saved successfully'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.route('/settings/api-key/<provider>', methods=['POST'])
def set_api_key(provider):
    """Set API key for a specific provider."""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip() if data else ''

        if not api_key:
            return jsonify({'success': False, 'error': 'API key is required'}), 400

        settings_service.set_api_key(provider, api_key)
        return jsonify({'success': True, 'message': f'API key for {provider} saved'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.route('/settings/api-key/<provider>', methods=['DELETE'])
def remove_api_key(provider):
    """Remove API key for a specific provider."""
    try:
        settings_service.remove_api_key(provider)
        return jsonify({'success': True, 'message': f'API key for {provider} removed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
