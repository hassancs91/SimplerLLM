"""
Product Enhancer API - Endpoints for enhancing product images with AI
"""
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service

product_enhancer_bp = Blueprint('product_enhancer', __name__)

# Preset enhancement prompts
ENHANCEMENT_PRESETS = {
    'full_enhancement': {
        'name': 'Full Enhancement',
        'description': 'Complete enhancement including background, lighting, color correction, and polish',
        'prompt': '''Enhance this product image professionally:
1. Clean up and perfect the background
2. Optimize lighting to highlight product features
3. Correct colors for accurate product representation
4. Remove any imperfections or distractions
5. Add subtle shadows for depth
6. Ensure the product looks premium and appealing
Keep the product itself accurate and unchanged.'''
    },
    'ecommerce_white': {
        'name': 'E-commerce White Background',
        'description': 'Clean white background perfect for online stores',
        'prompt': '''Transform this product image for e-commerce:
1. Replace the background with a pure, clean white background
2. Add soft, professional lighting that highlights the product
3. Ensure consistent shadow placement below the product
4. Remove any dust, scratches, or imperfections
5. Maintain accurate product colors
6. Center the product with appropriate padding
The result should be Amazon/Shopify marketplace ready.'''
    },
    'social_media_lifestyle': {
        'name': 'Social Media Lifestyle',
        'description': 'Lifestyle setting for social media appeal',
        'prompt': '''Enhance this product for social media marketing:
1. Place the product in an appealing lifestyle context
2. Add warm, inviting lighting with soft shadows
3. Include subtle complementary props or textures
4. Create depth with a slightly blurred background
5. Make colors vibrant and eye-catching
6. Give it an Instagram-worthy aesthetic
Maintain product accuracy while making it visually compelling.'''
    },
    'studio_professional': {
        'name': 'Studio Professional',
        'description': 'High-end studio photography look',
        'prompt': '''Give this product a professional studio photography look:
1. Create a gradient or seamless studio backdrop
2. Apply dramatic, professional lighting setup
3. Add precise reflections and highlights
4. Ensure razor-sharp product details
5. Create magazine-quality presentation
6. Add subtle vignette for focus
The result should look like a high-end catalog or advertisement.'''
    }
}


@product_enhancer_bp.route('/enhance', methods=['POST'])
def enhance_product():
    """
    Enhance a product image using AI with preset or custom instructions.

    Request body:
        {
            "source_image_id": "uuid" (optional, use if from gallery),
            "source_path": "C:/path/to/image.jpg" (optional, use if local file),
            "preset": "ecommerce_white" (optional, one of the preset keys),
            "custom_prompt": "Custom enhancement instructions" (optional),
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional),
            "aspect_ratio": "1:1" (optional)
        }

    Note: Either preset or custom_prompt must be provided.
    If both are provided, custom_prompt is appended to preset prompt.

    Response:
        {
            "success": true,
            "image_id": "uuid",
            "image_url": "/api/gallery/uuid/image",
            "metadata": { ... },
            "preset_used": "ecommerce_white"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        source_image_id = data.get('source_image_id')
        source_path = data.get('source_path')
        preset = data.get('preset')
        custom_prompt = data.get('custom_prompt', '').strip()

        # Validate source
        if not source_image_id and not source_path:
            return jsonify({'success': False, 'error': 'Either source_image_id or source_path is required'}), 400

        if source_image_id and source_path:
            return jsonify({'success': False, 'error': 'Provide only one of source_image_id or source_path'}), 400

        # Build enhancement prompt
        if preset and preset in ENHANCEMENT_PRESETS:
            base_prompt = ENHANCEMENT_PRESETS[preset]['prompt']
            if custom_prompt:
                enhancement_prompt = f"{base_prompt}\n\nAdditional instructions: {custom_prompt}"
            else:
                enhancement_prompt = base_prompt
        elif custom_prompt:
            enhancement_prompt = f"Enhance this product image: {custom_prompt}"
        else:
            return jsonify({'success': False, 'error': 'Either preset or custom_prompt is required'}), 400

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
            # Import the source image first
            try:
                imported = gallery_service.import_image(source_path)
                parent_id = imported['id']
            except Exception as e:
                return jsonify({'success': False, 'error': f'Failed to import source image: {str(e)}'}), 400

        # Enhance the image using edit_image
        result = image_service.edit_image(source_bytes, enhancement_prompt, model, provider, aspect_ratio)

        if not result['success']:
            return jsonify(result), 400

        # Save to gallery with version info
        image_bytes = result['image_bytes']

        # Truncate prompt for storage if too long
        stored_prompt = enhancement_prompt[:200] + ('...' if len(enhancement_prompt) > 200 else '')

        metadata = gallery_service.save_image(
            image_bytes=image_bytes,
            prompt=stored_prompt,
            model=model,
            image_type='enhanced',
            parent_id=parent_id,
            source_type=source_type
        )

        return jsonify({
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'metadata': metadata,
            'preset_used': preset
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@product_enhancer_bp.route('/enhance/presets', methods=['GET'])
def get_presets():
    """
    Get available enhancement presets.

    Response:
        {
            "success": true,
            "presets": [
                {
                    "id": "ecommerce_white",
                    "name": "E-commerce White Background",
                    "description": "Clean white background..."
                },
                ...
            ]
        }
    """
    presets = [
        {
            'id': key,
            'name': value['name'],
            'description': value['description']
        }
        for key, value in ENHANCEMENT_PRESETS.items()
    ]
    return jsonify({'success': True, 'presets': presets})
