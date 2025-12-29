"""
Sketch to Image API - Endpoints for generating images from sketches
"""
import base64
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service

sketch_bp = Blueprint('sketch', __name__)


@sketch_bp.route('/sketch-to-image', methods=['POST'])
def sketch_to_image():
    """
    Generate an image from a sketch and text prompt.

    Request body:
        {
            "sketch_data": "base64-encoded PNG data",
            "prompt": "Transform this sketch into a realistic image of...",
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional),
            "aspect_ratio": "1:1" (optional)
        }

    Response:
        {
            "success": true,
            "sketch_id": "uuid",
            "image_id": "uuid",
            "sketch_url": "/api/gallery/uuid/image",
            "image_url": "/api/gallery/uuid/image",
            "metadata": { ... }
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        sketch_data = data.get('sketch_data')
        prompt = data.get('prompt', '').strip()

        # Validate inputs
        if not sketch_data:
            return jsonify({'success': False, 'error': 'Sketch data is required'}), 400

        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')
        aspect_ratio = data.get('aspect_ratio', '1:1')

        # Decode base64 sketch data
        try:
            sketch_bytes = base64.b64decode(sketch_data)
        except Exception as e:
            return jsonify({'success': False, 'error': 'Invalid base64 sketch data'}), 400

        # Save sketch to gallery first
        sketch_metadata = gallery_service.save_image(
            image_bytes=sketch_bytes,
            prompt='User sketch',
            model=None,
            image_type='sketch',
            parent_id=None,
            source_type=None
        )

        # Generate image from sketch
        result = image_service.generate_from_sketch(
            sketch_bytes=sketch_bytes,
            prompt=prompt,
            model=model,
            provider=provider,
            aspect_ratio=aspect_ratio
        )

        if not result['success']:
            # Return error but still provide the sketch_id since it was saved
            return jsonify({
                'success': False,
                'error': result['error'],
                'sketch_id': sketch_metadata['id'],
                'sketch_url': f"/api/gallery/{sketch_metadata['id']}/image"
            }), 400

        # Save generated image to gallery with sketch as parent
        image_bytes = result['image_bytes']
        image_metadata = gallery_service.save_image(
            image_bytes=image_bytes,
            prompt=prompt,
            model=model,
            image_type='generated',
            parent_id=sketch_metadata['id'],
            source_type='sketch'
        )

        return jsonify({
            'success': True,
            'sketch_id': sketch_metadata['id'],
            'image_id': image_metadata['id'],
            'sketch_url': f"/api/gallery/{sketch_metadata['id']}/image",
            'image_url': f"/api/gallery/{image_metadata['id']}/image",
            'sketch_metadata': sketch_metadata,
            'metadata': image_metadata
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
