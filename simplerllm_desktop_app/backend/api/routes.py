"""
API Routes Registration
"""
from flask import Blueprint, jsonify
from api.chat import chat_bp
from api.providers import providers_bp
from api.settings import settings_bp

def register_routes(app):
    """Register all API blueprints."""

    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        try:
            import simplerllm
            simplerllm_version = getattr(simplerllm, '__version__', 'unknown')
        except ImportError:
            simplerllm_version = 'not installed'

        import sys
        return jsonify({
            'status': 'healthy',
            'simplerllm_version': simplerllm_version,
            'python_version': sys.version.split()[0]
        })

    # Register blueprints
    app.register_blueprint(chat_bp, url_prefix='/api')
    app.register_blueprint(providers_bp, url_prefix='/api')
    app.register_blueprint(settings_bp, url_prefix='/api')

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
