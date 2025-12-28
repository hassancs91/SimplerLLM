"""
Providers API - Endpoints for getting provider information
"""
from flask import Blueprint, jsonify
from services.shared import image_service

providers_bp = Blueprint('providers', __name__)


@providers_bp.route('/providers', methods=['GET'])
def get_providers():
    """
    Get list of available image generation providers.

    Response:
        {
            "success": true,
            "providers": [
                {
                    "id": "google",
                    "name": "Google AI",
                    "configured": true,
                    "models": [
                        {
                            "id": "gemini-2.5-flash-image-preview",
                            "name": "Gemini 2.5 Flash (Image Preview)",
                            "description": "Fast image generation with Gemini 2.5 Flash"
                        }
                    ]
                }
            ]
        }
    """
    try:
        result = image_service.get_providers()
        return jsonify({
            'success': True,
            'providers': result['providers']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@providers_bp.route('/providers/<provider_id>/models', methods=['GET'])
def get_provider_models(provider_id):
    """
    Get models for a specific provider.

    Response:
        {
            "success": true,
            "models": [ ... ]
        }
    """
    try:
        result = image_service.get_providers()
        for provider in result['providers']:
            if provider['id'] == provider_id:
                return jsonify({
                    'success': True,
                    'models': provider['models']
                })

        return jsonify({'success': False, 'error': 'Provider not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
