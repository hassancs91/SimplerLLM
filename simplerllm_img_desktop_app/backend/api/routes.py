"""
API Routes - Blueprint registration and health check
"""
import sys
from flask import Blueprint, jsonify

# Create main routes blueprint
main_bp = Blueprint('main', __name__)


@main_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'app': 'SimplerLLM Image Tools',
        'version': '1.0.0',
        'python_version': sys.version
    })


@main_bp.route('/debug/settings', methods=['GET'])
def debug_settings():
    """Debug endpoint to verify singleton and settings state."""
    from services.shared import settings_service, image_service

    settings = settings_service.get_settings()
    has_google_key = settings_service.has_api_key('google')
    api_key_preview = None

    # Show first/last 4 chars of API key if present (for debugging)
    google_key = settings_service.get_api_key('google')
    if google_key:
        if len(google_key) > 8:
            api_key_preview = f"{google_key[:4]}...{google_key[-4:]}"
        else:
            api_key_preview = "****"

    # Check image_service's view of the API key
    image_service_key = image_service._get_api_key('google')
    image_service_has_key = bool(image_service_key)

    return jsonify({
        'settings_file': str(settings_service._settings_file),
        'settings_dir': str(settings_service._settings_dir),
        'has_google_key_settings': has_google_key,
        'has_google_key_image_service': image_service_has_key,
        'api_key_preview': api_key_preview,
        'settings_service_id': id(settings_service),
        'image_service_settings_id': id(image_service._settings_service),
        'same_instance': id(settings_service) == id(image_service._settings_service),
        'raw_api_keys': list(settings.get('api_keys', {}).keys())
    })


def register_routes(app):
    """Register all API blueprints with the Flask app."""
    from api.image_generation import image_bp
    from api.gallery import gallery_bp
    from api.settings import settings_bp
    from api.providers import providers_bp

    # Register blueprints
    app.register_blueprint(main_bp, url_prefix='/api')
    app.register_blueprint(image_bp, url_prefix='/api')
    app.register_blueprint(gallery_bp, url_prefix='/api')
    app.register_blueprint(settings_bp, url_prefix='/api')
    app.register_blueprint(providers_bp, url_prefix='/api')

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
