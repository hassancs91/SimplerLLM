"""
Retrieval API Endpoints with Server-Sent Events (SSE) support.
"""
import json
from flask import Blueprint, request, jsonify, Response
from services.retrieval_service import retrieval_service

retrieval_bp = Blueprint('retrieval', __name__)


@retrieval_bp.route('/retrieval/samples', methods=['GET'])
def get_samples():
    """
    Get available sample datasets.

    Returns:
        {
            "success": true,
            "samples": [
                {"id": "ai_news", "name": "AI News Articles", "description": "...", "chunk_count": 10},
                ...
            ]
        }
    """
    try:
        samples = retrieval_service.get_samples()
        return jsonify({
            'success': True,
            'samples': samples
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@retrieval_bp.route('/retrieval/status', methods=['GET'])
def get_status():
    """
    Get current index status.

    Returns:
        {
            "success": true,
            "has_index": true/false
        }
    """
    try:
        return jsonify({
            'success': True,
            'has_index': retrieval_service.has_index()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@retrieval_bp.route('/retrieval/build-index/stream', methods=['POST'])
def stream_build_index():
    """
    Build a retrieval index with real-time SSE streaming.

    Request body:
    {
        "source": "text" | "sample",
        "text": "...",           // if source="text"
        "sample_id": "ai_news",  // if source="sample"
        "provider": "openai",
        "model": "gpt-4o"
    }

    SSE Events:
    - start: Index building started
    - chunking_complete: Text has been chunked
    - clustering_progress: Clustering in progress
    - complete: Index built successfully with tree structure
    - error: Error occurred
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        source = data.get('source', 'text')
        text = data.get('text', '').strip() if source == 'text' else None
        sample_id = data.get('sample_id', '') if source == 'sample' else None
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')

        # Validation
        if source == 'text' and not text:
            return jsonify({'success': False, 'error': 'Text is required when source is "text"'}), 400
        if source == 'sample' and not sample_id:
            return jsonify({'success': False, 'error': 'Sample ID is required when source is "sample"'}), 400

        def generate():
            """Generator that yields SSE events."""
            for event in retrieval_service.build_index_streaming(
                source=source,
                text=text,
                sample_id=sample_id,
                provider=provider,
                model=model
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


@retrieval_bp.route('/retrieval/build-index', methods=['POST'])
def build_index():
    """
    Build a retrieval index without streaming (for fallback).

    Returns the complete result when finished.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        source = data.get('source', 'text')
        text = data.get('text', '').strip() if source == 'text' else None
        sample_id = data.get('sample_id', '') if source == 'sample' else None
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')

        # Validation
        if source == 'text' and not text:
            return jsonify({'success': False, 'error': 'Text is required when source is "text"'}), 400
        if source == 'sample' and not sample_id:
            return jsonify({'success': False, 'error': 'Sample ID is required when source is "sample"'}), 400

        # Collect all events
        result = None
        error = None

        for event in retrieval_service.build_index_streaming(
            source=source,
            text=text,
            sample_id=sample_id,
            provider=provider,
            model=model
        ):
            if event['type'] == 'complete':
                result = event
            elif event['type'] == 'error':
                error = event['error']

        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500

        return jsonify({
            'success': True,
            'tree': result.get('tree', {}),
            'stats': result.get('stats', {})
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@retrieval_bp.route('/retrieval/query/stream', methods=['POST'])
def stream_query():
    """
    Query the built index with real-time SSE streaming.

    Request body:
    {
        "query": "What are AI safety challenges?",
        "top_k": 3,
        "provider": "openai",
        "model": "gpt-4o"
    }

    SSE Events:
    - start: Query started
    - navigation_step: Navigating through cluster tree
    - result: A result chunk found
    - complete: Query complete with all results
    - error: Error occurred
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        query = data.get('query', '').strip()
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400

        top_k = min(max(int(data.get('top_k', 3)), 1), 10)
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')

        def generate():
            """Generator that yields SSE events."""
            for event in retrieval_service.query_streaming(
                query=query,
                top_k=top_k,
                provider=provider,
                model=model
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


@retrieval_bp.route('/retrieval/query', methods=['POST'])
def query():
    """
    Query the built index without streaming (for fallback).

    Returns the complete result when finished.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({'success': False, 'error': 'Query is required'}), 400

        top_k = min(max(int(data.get('top_k', 3)), 1), 10)
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')

        # Collect all events
        result = None
        error = None

        for event in retrieval_service.query_streaming(
            query=query_text,
            top_k=top_k,
            provider=provider,
            model=model
        ):
            if event['type'] == 'complete':
                result = event
            elif event['type'] == 'error':
                error = event['error']

        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500

        return jsonify({
            'success': True,
            'results': result.get('results', []),
            'navigation_path': result.get('navigation_path', []),
            'stats': result.get('stats', {})
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@retrieval_bp.route('/retrieval/clear', methods=['POST'])
def clear_index():
    """
    Clear the current index.
    """
    try:
        retrieval_service.clear_index()
        return jsonify({
            'success': True
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
