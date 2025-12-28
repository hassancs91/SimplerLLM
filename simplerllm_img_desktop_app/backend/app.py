"""
SimplerLLM Image Tools - Flask Backend
"""
import os
import sys
from flask import Flask
from flask_cors import CORS

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.routes import register_routes


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Enable CORS for Electron frontend
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Configuration
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['JSON_SORT_KEYS'] = False

    # Register API routes
    register_routes(app)

    return app


def main():
    """Run the Flask development server."""
    port = int(os.environ.get('FLASK_PORT', 5124))

    app = create_app()

    print(f"Starting SimplerLLM Image Tools backend on port {port}...")
    print(f"Health check: http://localhost:{port}/api/health")

    app.run(
        host='127.0.0.1',
        port=port,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
