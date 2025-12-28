"""
Brainstorm API Endpoints with Server-Sent Events (SSE) support.
"""
import json
from flask import Blueprint, request, jsonify, Response
from services.brainstorm_service import brainstorm_service

brainstorm_bp = Blueprint('brainstorm', __name__)


@brainstorm_bp.route('/brainstorm/stream', methods=['POST'])
def stream_brainstorm():
    """
    Start a brainstorming session with real-time SSE streaming.

    Request body:
    {
        "prompt": "Generate marketing strategies for...",
        "provider": "openai",
        "model": "gpt-4o",
        "params": {
            "max_depth": 2,
            "ideas_per_level": 5,
            "top_n": 3,
            "min_quality_threshold": 5.0
        }
    }

    SSE Events:
    - start: Session started
    - iteration_start: New iteration beginning
    - idea: Individual idea generated
    - iteration_complete: Iteration finished
    - complete: Session finished with full result
    - error: Error occurred
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')
        params = data.get('params', {})

        # Extract and validate parameters
        max_depth = min(max(int(params.get('max_depth', 2)), 1), 5)
        ideas_per_level = min(max(int(params.get('ideas_per_level', 5)), 3), 10)
        top_n = min(max(int(params.get('top_n', 3)), 1), 5)
        min_quality_threshold = min(max(float(params.get('min_quality_threshold', 5.0)), 1.0), 10.0)

        def generate():
            """Generator that yields SSE events."""
            for event in brainstorm_service.run_brainstorm_streaming(
                prompt=prompt,
                provider=provider,
                model=model,
                max_depth=max_depth,
                ideas_per_level=ideas_per_level,
                top_n=top_n,
                min_quality_threshold=min_quality_threshold
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


@brainstorm_bp.route('/brainstorm/run', methods=['POST'])
def run_brainstorm():
    """
    Run a brainstorming session without streaming (for fallback).

    Returns the complete result when finished.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')
        params = data.get('params', {})

        # Extract and validate parameters
        max_depth = min(max(int(params.get('max_depth', 2)), 1), 5)
        ideas_per_level = min(max(int(params.get('ideas_per_level', 5)), 3), 10)
        top_n = min(max(int(params.get('top_n', 3)), 1), 5)
        min_quality_threshold = min(max(float(params.get('min_quality_threshold', 5.0)), 1.0), 10.0)

        # Collect all events
        result = None
        error = None

        for event in brainstorm_service.run_brainstorm_streaming(
            prompt=prompt,
            provider=provider,
            model=model,
            max_depth=max_depth,
            ideas_per_level=ideas_per_level,
            top_n=top_n,
            min_quality_threshold=min_quality_threshold
        ):
            if event['type'] == 'complete':
                result = event['result']
            elif event['type'] == 'error':
                error = event['error']

        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
