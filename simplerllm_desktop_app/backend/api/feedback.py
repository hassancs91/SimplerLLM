"""
Feedback API Endpoints with Server-Sent Events (SSE) support.
"""
import json
from flask import Blueprint, request, jsonify, Response
from services.feedback_service import feedback_service

feedback_bp = Blueprint('feedback', __name__)


@feedback_bp.route('/feedback/stream', methods=['POST'])
def stream_feedback():
    """
    Start an LLM Feedback loop with real-time SSE streaming.

    Request body:
    {
        "prompt": "Explain quantum computing",
        "architecture": "dual",
        "generator": {"provider": "openai", "model": "gpt-4o"},
        "critic": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        "providers": [],
        "max_iterations": 3,
        "criteria": ["accuracy", "clarity", "completeness"],
        "initial_answer": null,
        "convergence_threshold": 0.1,
        "quality_threshold": null
    }

    SSE Events:
    - start: Session started with max_iterations and architecture
    - llm_ready: LLM(s) initialized
    - running: Improvement loop started
    - iteration_complete: An iteration finished with critique and improved answer
    - complete: Full result with all iterations
    - error: Error occurred
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        architecture = data.get('architecture', 'single')
        if architecture not in ['single', 'dual', 'multi']:
            return jsonify({'success': False, 'error': 'Invalid architecture. Must be single, dual, or multi'}), 400

        generator_config = data.get('generator')
        critic_config = data.get('critic')
        providers_config = data.get('providers', [])

        # Validate based on architecture
        if architecture == 'single':
            if not generator_config or not generator_config.get('provider') or not generator_config.get('model'):
                return jsonify({'success': False, 'error': 'Generator provider and model required for single architecture'}), 400

        elif architecture == 'dual':
            if not generator_config or not generator_config.get('provider') or not generator_config.get('model'):
                return jsonify({'success': False, 'error': 'Generator provider and model required for dual architecture'}), 400
            if not critic_config or not critic_config.get('provider') or not critic_config.get('model'):
                return jsonify({'success': False, 'error': 'Critic provider and model required for dual architecture'}), 400

        elif architecture == 'multi':
            if not providers_config or len(providers_config) < 2:
                return jsonify({'success': False, 'error': 'At least 2 providers required for multi architecture'}), 400
            for i, p in enumerate(providers_config):
                if not p.get('provider') or not p.get('model'):
                    return jsonify({'success': False, 'error': f'Provider {i+1} missing provider or model'}), 400

        max_iterations = data.get('max_iterations', 3)
        if not isinstance(max_iterations, int) or max_iterations < 1 or max_iterations > 10:
            return jsonify({'success': False, 'error': 'max_iterations must be between 1 and 10'}), 400

        criteria = data.get('criteria', ['accuracy', 'clarity', 'completeness'])
        initial_answer = data.get('initial_answer')
        convergence_threshold = data.get('convergence_threshold', 0.1)
        quality_threshold = data.get('quality_threshold')

        def generate():
            """Generator that yields SSE events."""
            for event in feedback_service.run_feedback_streaming(
                prompt=prompt,
                architecture=architecture,
                generator_config=generator_config,
                critic_config=critic_config,
                providers_config=providers_config,
                max_iterations=max_iterations,
                criteria=criteria,
                initial_answer=initial_answer,
                convergence_threshold=convergence_threshold,
                quality_threshold=quality_threshold
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


@feedback_bp.route('/feedback/run', methods=['POST'])
def run_feedback():
    """
    Run LLM Feedback loop without streaming (fallback).

    Returns the complete result when finished.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        architecture = data.get('architecture', 'single')
        if architecture not in ['single', 'dual', 'multi']:
            return jsonify({'success': False, 'error': 'Invalid architecture'}), 400

        generator_config = data.get('generator')
        critic_config = data.get('critic')
        providers_config = data.get('providers', [])

        max_iterations = data.get('max_iterations', 3)
        criteria = data.get('criteria', ['accuracy', 'clarity', 'completeness'])
        initial_answer = data.get('initial_answer')
        convergence_threshold = data.get('convergence_threshold', 0.1)
        quality_threshold = data.get('quality_threshold')

        # Collect all events
        result = None
        error = None

        for event in feedback_service.run_feedback_streaming(
            prompt=prompt,
            architecture=architecture,
            generator_config=generator_config,
            critic_config=critic_config,
            providers_config=providers_config,
            max_iterations=max_iterations,
            criteria=criteria,
            initial_answer=initial_answer,
            convergence_threshold=convergence_threshold,
            quality_threshold=quality_threshold
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
