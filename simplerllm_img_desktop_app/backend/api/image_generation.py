"""
Image Generation API - Endpoints for generating images
"""
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service

image_bp = Blueprint('image', __name__)


@image_bp.route('/generate', methods=['POST'])
def generate_image():
    """
    Generate an image from a text prompt.

    Request body:
        {
            "prompt": "A beautiful sunset over mountains",
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional)
        }

    Response:
        {
            "success": true,
            "image_id": "uuid",
            "image_url": "/api/gallery/uuid/image",
            "metadata": { ... }
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')
        aspect_ratio = data.get('aspect_ratio', '1:1')

        # Generate the image
        result = image_service.generate_image(prompt, model, provider, aspect_ratio)

        if not result['success']:
            return jsonify(result), 400

        # Save to gallery
        image_bytes = result['image_bytes']
        metadata = gallery_service.save_image(image_bytes, prompt, model)

        return jsonify({
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'metadata': metadata
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
