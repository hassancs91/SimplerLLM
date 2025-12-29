"""
Character Generator API - Generate consistent characters with style and pose presets
"""
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service

character_generator_bp = Blueprint('character_generator', __name__)

# Style presets for initial character generation
STYLE_PRESETS = {
    'anime': {
        'name': 'Anime',
        'description': 'Japanese anime/manga style with expressive features',
        'prompt_modifier': 'in anime art style, vibrant colors, clean linework, expressive eyes, manga-inspired'
    },
    'realistic': {
        'name': 'Realistic',
        'description': 'Photorealistic character with detailed features',
        'prompt_modifier': 'photorealistic, highly detailed, natural lighting, lifelike skin texture, realistic proportions'
    },
    'cartoon': {
        'name': 'Cartoon',
        'description': 'Western cartoon style with bold shapes',
        'prompt_modifier': 'cartoon style, bold outlines, exaggerated features, vibrant saturated colors, stylized'
    },
    'pixel_art': {
        'name': 'Pixel Art',
        'description': 'Retro pixel art style for games',
        'prompt_modifier': 'pixel art style, 16-bit aesthetic, limited color palette, crisp pixels, retro game character'
    },
    'fantasy': {
        'name': 'Fantasy',
        'description': 'Epic fantasy illustration style',
        'prompt_modifier': 'fantasy art style, painterly, dramatic lighting, epic atmosphere, detailed armor and clothing'
    }
}

# Pose presets for variations
POSE_PRESETS = {
    'front_view': {
        'name': 'Front View',
        'description': 'Character facing forward',
        'prompt_modifier': 'front view, facing the camera, symmetrical pose'
    },
    'side_profile': {
        'name': 'Side Profile',
        'description': 'Character from the side',
        'prompt_modifier': 'side profile view, 90-degree angle, profile portrait'
    },
    'three_quarter': {
        'name': 'Three-Quarter View',
        'description': '45-degree angle view',
        'prompt_modifier': 'three-quarter view, 45-degree angle, dynamic perspective'
    },
    'back_view': {
        'name': 'Back View',
        'description': 'Character from behind',
        'prompt_modifier': 'back view, facing away, showing back details'
    },
    'action_pose': {
        'name': 'Action Pose',
        'description': 'Dynamic action stance',
        'prompt_modifier': 'dynamic action pose, dramatic stance, movement, energy'
    },
    'sitting': {
        'name': 'Sitting',
        'description': 'Character in seated position',
        'prompt_modifier': 'sitting pose, relaxed position, seated character'
    },
    'full_body': {
        'name': 'Full Body',
        'description': 'Complete character view',
        'prompt_modifier': 'full body shot, head to toe, complete character visible'
    },
    'portrait_closeup': {
        'name': 'Portrait Close-up',
        'description': 'Face and shoulders focus',
        'prompt_modifier': 'portrait close-up, face and shoulders, detailed facial features'
    }
}


@character_generator_bp.route('/character/generate', methods=['POST'])
def generate_character():
    """
    Generate an initial character image with style preset.

    Request body:
        {
            "prompt": "character description" (required),
            "style": "anime" (optional, one of STYLE_PRESETS keys),
            "aspect_ratio": "1:1" (optional),
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional)
        }

    Response:
        {
            "success": true,
            "image_id": "uuid",
            "image_url": "/api/gallery/uuid/image",
            "metadata": { ... },
            "style_used": "anime"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        style = data.get('style')
        aspect_ratio = data.get('aspect_ratio', '1:1')
        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')

        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        # Build full prompt with style modifier
        if style and style in STYLE_PRESETS:
            style_modifier = STYLE_PRESETS[style]['prompt_modifier']
            full_prompt = f"{prompt}, {style_modifier}"
        else:
            full_prompt = prompt

        # Generate the character
        result = image_service.generate_image(
            prompt=full_prompt,
            model=model,
            provider=provider,
            aspect_ratio=aspect_ratio
        )

        if not result['success']:
            return jsonify(result), 400

        # Save to gallery
        image_bytes = result['image_bytes']
        stored_prompt = prompt[:500] + ('...' if len(prompt) > 500 else '')

        metadata = gallery_service.save_image(
            image_bytes=image_bytes,
            prompt=stored_prompt,
            model=model,
            image_type='character',
            parent_id=None,
            source_type='generated'
        )

        return jsonify({
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'metadata': metadata,
            'style_used': style
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@character_generator_bp.route('/character/variation', methods=['POST'])
def generate_variation():
    """
    Generate a pose/angle variation using original character as reference.

    Request body:
        {
            "reference_image_id": "uuid" (required - original character from gallery),
            "pose": "front_view" (required, one of POSE_PRESETS keys),
            "custom_prompt": "additional instructions" (optional),
            "aspect_ratio": "1:1" (optional),
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional)
        }

    Response:
        {
            "success": true,
            "image_id": "uuid",
            "image_url": "/api/gallery/uuid/image",
            "metadata": { ... },
            "pose_used": "front_view"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        reference_image_id = data.get('reference_image_id')
        pose = data.get('pose')
        custom_prompt = data.get('custom_prompt', '').strip()
        aspect_ratio = data.get('aspect_ratio', '1:1')
        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')

        if not reference_image_id:
            return jsonify({'success': False, 'error': 'Reference image ID is required'}), 400

        if not pose or pose not in POSE_PRESETS:
            return jsonify({'success': False, 'error': 'Valid pose preset is required'}), 400

        # Load reference image
        image_path = gallery_service.get_image_path(reference_image_id)
        if not image_path:
            return jsonify({'success': False, 'error': 'Reference image not found'}), 404

        with open(image_path, 'rb') as f:
            reference_bytes = f.read()

        # Get original prompt from metadata if available
        original_metadata = gallery_service.get_image(reference_image_id)
        original_prompt = original_metadata.get('prompt', 'this character') if original_metadata else 'this character'

        # Build variation prompt
        pose_modifier = POSE_PRESETS[pose]['prompt_modifier']
        variation_prompt = f"Generate the same character as in the reference image, maintaining exact appearance, face, clothing, and style. New pose: {pose_modifier}"

        if custom_prompt:
            variation_prompt += f". Additional: {custom_prompt}"

        # Generate with reference
        result = image_service.generate_with_reference(
            prompt=variation_prompt,
            reference_images=[reference_bytes],
            model=model,
            provider=provider,
            aspect_ratio=aspect_ratio
        )

        if not result['success']:
            return jsonify(result), 400

        # Save to gallery
        image_bytes = result['image_bytes']
        stored_prompt = f"[{POSE_PRESETS[pose]['name']}] {original_prompt}"[:500]

        metadata = gallery_service.save_image(
            image_bytes=image_bytes,
            prompt=stored_prompt,
            model=model,
            image_type='character_variation',
            parent_id=reference_image_id,
            source_type='character_reference'
        )

        return jsonify({
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'metadata': metadata,
            'pose_used': pose
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@character_generator_bp.route('/character/presets', methods=['GET'])
def get_presets():
    """
    Get available style and pose presets.

    Response:
        {
            "success": true,
            "styles": [...],
            "poses": [...]
        }
    """
    styles = [
        {'id': key, 'name': val['name'], 'description': val['description']}
        for key, val in STYLE_PRESETS.items()
    ]

    poses = [
        {'id': key, 'name': val['name'], 'description': val['description']}
        for key, val in POSE_PRESETS.items()
    ]

    return jsonify({
        'success': True,
        'styles': styles,
        'poses': poses
    })
