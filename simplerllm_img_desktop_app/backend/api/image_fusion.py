"""
Image Fusion API - Combine multiple reference images into a single generated image
"""
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service
from pathlib import Path

image_fusion_bp = Blueprint('image_fusion', __name__)


@image_fusion_bp.route('/fusion/generate', methods=['POST'])
def generate_fusion():
    """
    Generate a fused image from multiple reference images.

    Request body:
        {
            "image_sources": [
                { "type": "gallery", "id": "uuid" },
                { "type": "local", "path": "/path/to/file" }
            ],
            "prompt": "Combine these images into...",
            "aspect_ratio": "1:1" (optional, default 1:1),
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

        image_sources = data.get('image_sources', [])
        prompt = data.get('prompt', '').strip()
        aspect_ratio = data.get('aspect_ratio', '1:1')
        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')

        # Validate inputs
        if not image_sources:
            return jsonify({'success': False, 'error': 'At least one image is required'}), 400

        if len(image_sources) > 5:
            return jsonify({'success': False, 'error': 'Maximum 5 images allowed'}), 400

        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        # Load all reference images as bytes
        reference_images = []
        parent_ids = []

        for source in image_sources:
            source_type = source.get('type')

            if source_type == 'gallery':
                image_id = source.get('id')
                if not image_id:
                    return jsonify({'success': False, 'error': 'Gallery image missing id'}), 400

                image_path = gallery_service.get_image_path(image_id)
                if not image_path:
                    return jsonify({'success': False, 'error': f'Gallery image not found: {image_id}'}), 404

                with open(image_path, 'rb') as f:
                    reference_images.append(f.read())
                parent_ids.append(image_id)

            elif source_type == 'local':
                file_path = source.get('path')
                if not file_path:
                    return jsonify({'success': False, 'error': 'Local image missing path'}), 400

                path = Path(file_path)
                if not path.exists():
                    return jsonify({'success': False, 'error': f'Local file not found: {file_path}'}), 404

                with open(path, 'rb') as f:
                    reference_images.append(f.read())

                # Import to gallery for tracking
                try:
                    imported = gallery_service.import_image(file_path)
                    parent_ids.append(imported['id'])
                except Exception as e:
                    return jsonify({'success': False, 'error': f'Failed to import image: {str(e)}'}), 400

            else:
                return jsonify({'success': False, 'error': f'Invalid source type: {source_type}'}), 400

        # Generate the fused image
        result = image_service.generate_with_reference(
            prompt=prompt,
            reference_images=reference_images,
            model=model,
            provider=provider,
            aspect_ratio=aspect_ratio
        )

        if not result['success']:
            return jsonify(result), 400

        # Save to gallery
        image_bytes = result['image_bytes']

        # Use first parent as the primary parent_id for version tracking
        primary_parent_id = parent_ids[0] if parent_ids else None

        # Truncate prompt for storage if too long
        stored_prompt = prompt[:500] + ('...' if len(prompt) > 500 else '')

        metadata = gallery_service.save_image(
            image_bytes=image_bytes,
            prompt=stored_prompt,
            model=model,
            image_type='fusion',
            parent_id=primary_parent_id,
            source_type='fusion'
        )

        return jsonify({
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'source_count': len(reference_images),
            'metadata': metadata
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
