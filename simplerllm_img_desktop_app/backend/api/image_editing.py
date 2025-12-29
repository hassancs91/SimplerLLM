"""
Image Editing API - Endpoints for editing images and managing versions
"""
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service

image_editing_bp = Blueprint('image_editing', __name__)


@image_editing_bp.route('/edit', methods=['POST'])
def edit_image():
    """
    Edit an existing image using AI with text instructions.

    Request body:
        {
            "source_image_id": "uuid" (optional, use if editing from gallery),
            "source_path": "C:/path/to/image.jpg" (optional, use if editing local file),
            "prompt": "Make the sky more dramatic",
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional),
            "aspect_ratio": "1:1" (optional)
        }

    Note: Either source_image_id or source_path must be provided, not both.

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

        source_image_id = data.get('source_image_id')
        source_path = data.get('source_path')
        prompt = data.get('prompt', '').strip()

        # Validate inputs
        if not prompt:
            return jsonify({'success': False, 'error': 'Edit prompt is required'}), 400

        if not source_image_id and not source_path:
            return jsonify({'success': False, 'error': 'Either source_image_id or source_path is required'}), 400

        if source_image_id and source_path:
            return jsonify({'success': False, 'error': 'Provide only one of source_image_id or source_path'}), 400

        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')
        aspect_ratio = data.get('aspect_ratio', '1:1')

        # Get source image bytes
        source_type = None
        parent_id = None

        if source_image_id:
            # Load from gallery
            image_path = gallery_service.get_image_path(source_image_id)
            if not image_path:
                return jsonify({'success': False, 'error': 'Source image not found in gallery'}), 404

            with open(image_path, 'rb') as f:
                source_bytes = f.read()
            source_type = 'gallery'
            parent_id = source_image_id
        else:
            # Load from local file
            from pathlib import Path
            path = Path(source_path)
            if not path.exists():
                return jsonify({'success': False, 'error': 'Source file not found'}), 404

            with open(path, 'rb') as f:
                source_bytes = f.read()
            source_type = 'local'
            # For local files, we first import them, then use as parent
            # Import the source image first
            try:
                imported = gallery_service.import_image(source_path)
                parent_id = imported['id']
            except Exception as e:
                return jsonify({'success': False, 'error': f'Failed to import source image: {str(e)}'}), 400

        # Edit the image
        result = image_service.edit_image(source_bytes, prompt, model, provider, aspect_ratio)

        if not result['success']:
            return jsonify(result), 400

        # Save to gallery with version info
        image_bytes = result['image_bytes']
        metadata = gallery_service.save_image(
            image_bytes=image_bytes,
            prompt=prompt,
            model=model,
            image_type='edited',
            parent_id=parent_id,
            source_type=source_type
        )

        return jsonify({
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'metadata': metadata
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@image_editing_bp.route('/import', methods=['POST'])
def import_image():
    """
    Import a local image into the gallery.

    Request body:
        {
            "file_path": "C:/path/to/image.jpg"
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

        file_path = data.get('file_path')
        if not file_path:
            return jsonify({'success': False, 'error': 'file_path is required'}), 400

        # Import the image
        metadata = gallery_service.import_image(file_path)

        return jsonify({
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'metadata': metadata
        })

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@image_editing_bp.route('/gallery/<image_id>/versions', methods=['GET'])
def get_version_tree(image_id):
    """
    Get the version tree for an image.

    Response:
        {
            "success": true,
            "tree": {
                "id": "root-uuid",
                "prompt": "...",
                "timestamp": "...",
                "type": "generated",
                "children": [...]
            },
            "family": [...],
            "has_versions": true
        }
    """
    try:
        # Check if image exists
        image = gallery_service.get_image(image_id)
        if not image:
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Get version tree
        tree = gallery_service.get_version_tree(image_id)
        family = gallery_service.get_image_family(image_id)
        has_versions = gallery_service.has_versions(image_id)

        return jsonify({
            'success': True,
            'tree': tree,
            'family': family,
            'has_versions': has_versions,
            'current_id': image_id
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
