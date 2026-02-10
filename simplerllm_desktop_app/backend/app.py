"""
SimplerLLM Playground - Flask Backend
"""
import os
import sys

# Add backend directory to path for local imports (api, services)
# This is required for packaged builds where PYTHONPATH is ignored by embedded Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path for SimplerLLM imports (development mode)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask
from flask_cors import CORS

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
    port = int(os.environ.get('FLASK_PORT', 5123))

    app = create_app()

    print(f"Starting SimplerLLM Playground backend on port {port}...")
    print(f"Health check: http://localhost:{port}/api/health")

    app.run(
        host='127.0.0.1',
        port=port,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
