"""
Judge API Endpoints with Server-Sent Events (SSE) support.
"""
import json
from flask import Blueprint, request, jsonify, Response
from services.judge_service import judge_service

judge_bp = Blueprint('judge', __name__)


@judge_bp.route('/judge/stream', methods=['POST'])
def stream_judge():
    """
    Start an LLM Judge evaluation with real-time SSE streaming.

    Request body:
    {
        "prompt": "Explain quantum computing",
        "contestants": [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"}
        ],
        "judge": {"provider": "anthropic", "model": "claude-3-opus-20240229"},
        "mode": "synthesize",
        "criteria": ["accuracy", "clarity", "completeness"]
    }

    SSE Events:
    - start: Session started with total_contestants count
    - contestant_start: Contestant is being initialized
    - contestant_ready: Contestant LLM is ready
    - contestants_running: All contestants are generating responses
    - contestant_complete: Contestant finished generating
    - judging_complete: Judge has evaluated all responses
    - complete: Full result with evaluations
    - error: Error occurred
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        contestants = data.get('contestants', [])
        if not contestants or len(contestants) < 2:
            return jsonify({'success': False, 'error': 'At least 2 contestants are required'}), 400

        judge_config = data.get('judge', {})
        if not judge_config.get('provider') or not judge_config.get('model'):
            return jsonify({'success': False, 'error': 'Judge provider and model are required'}), 400

        mode = data.get('mode', 'synthesize')
        if mode not in ['select_best', 'synthesize', 'compare']:
            return jsonify({'success': False, 'error': 'Invalid mode. Must be select_best, synthesize, or compare'}), 400

        criteria = data.get('criteria', ['accuracy', 'clarity', 'completeness'])

        def generate():
            """Generator that yields SSE events."""
            for event in judge_service.run_judge_streaming(
                prompt=prompt,
                contestants=contestants,
                judge_config=judge_config,
                mode=mode,
                criteria=criteria
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


@judge_bp.route('/judge/run', methods=['POST'])
def run_judge():
    """
    Run LLM Judge evaluation without streaming (fallback).

    Returns the complete result when finished.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        contestants = data.get('contestants', [])
        if not contestants or len(contestants) < 2:
            return jsonify({'success': False, 'error': 'At least 2 contestants are required'}), 400

        judge_config = data.get('judge', {})
        if not judge_config.get('provider') or not judge_config.get('model'):
            return jsonify({'success': False, 'error': 'Judge provider and model are required'}), 400

        mode = data.get('mode', 'synthesize')
        if mode not in ['select_best', 'synthesize', 'compare']:
            return jsonify({'success': False, 'error': 'Invalid mode. Must be select_best, synthesize, or compare'}), 400

        criteria = data.get('criteria', ['accuracy', 'clarity', 'completeness'])

        # Collect all events
        result = None
        error = None

        for event in judge_service.run_judge_streaming(
            prompt=prompt,
            contestants=contestants,
            judge_config=judge_config,
            mode=mode,
            criteria=criteria
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
