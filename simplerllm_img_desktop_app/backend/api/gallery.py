"""
Gallery API - Endpoints for managing generated images
"""
from flask import Blueprint, jsonify, send_file
from services.shared import gallery_service

gallery_bp = Blueprint('gallery', __name__)


@gallery_bp.route('/gallery', methods=['GET'])
def get_gallery():
    """
    Get all generated images.

    Response:
        {
            "success": true,
            "images": [
                {
                    "id": "uuid",
                    "filename": "uuid.png",
                    "prompt": "...",
                    "model": "gemini-2.5-flash-image-preview",
                    "timestamp": "2024-01-01T12:00:00",
                    "size": {"width": 1024, "height": 1024}
                }
            ]
        }
    """
    try:
        images = gallery_service.get_all_images()
        return jsonify({
            'success': True,
            'images': images
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@gallery_bp.route('/gallery/<image_id>', methods=['GET'])
def get_image_metadata(image_id):
    """
    Get metadata for a specific image.

    Response:
        {
            "success": true,
            "image": { ... metadata ... }
        }
    """
    try:
        metadata = gallery_service.get_image(image_id)
        if metadata:
            return jsonify({
                'success': True,
                'image': metadata
            })
        else:
            return jsonify({'success': False, 'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@gallery_bp.route('/gallery/<image_id>/image', methods=['GET'])
def get_image_file(image_id):
    """
    Serve the actual image file.

    Response: Image file (PNG)
    """
    try:
        filepath = gallery_service.get_image_path(image_id)
        if filepath and filepath.exists():
            return send_file(filepath, mimetype='image/png')
        else:
            return jsonify({'success': False, 'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@gallery_bp.route('/gallery/<image_id>/path', methods=['GET'])
def get_image_filepath(image_id):
    """
    Get the local file path for an image.

    Response:
        {
            "success": true,
            "path": "C:/Users/.../images/uuid.png"
        }
    """
    try:
        filepath = gallery_service.get_image_path(image_id)
        if filepath and filepath.exists():
            return jsonify({
                'success': True,
                'path': str(filepath)
            })
        else:
            return jsonify({'success': False, 'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@gallery_bp.route('/gallery/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    """
    Delete an image.

    Response:
        {
            "success": true
        }
    """
    try:
        deleted = gallery_service.delete_image(image_id)
        if deleted:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
