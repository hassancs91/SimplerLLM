#!/usr/bin/env python
"""
Script to run the Dynamic LLM API server with command-line arguments.
"""
import argparse
import os
import uvicorn
from dotenv import load_dotenv

def main():
    """Run the Dynamic LLM API server."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the Dynamic LLM API server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"], help="Log level")
    args = parser.parse_args()
    
    # Print server information
    print(f"Starting Dynamic LLM API server on http://{args.host}:{args.port}")
    print(f"API documentation will be available at http://{args.host}:{args.port}/docs")
    
    # Run the server
    uvicorn.run(
        "fastapi_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()
