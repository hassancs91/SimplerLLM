"""
YouTube Thumbnail Generator API - Endpoints for generating YouTube thumbnails with face reference
"""
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service

youtube_thumbnail_bp = Blueprint('youtube_thumbnail', __name__)

# Style presets with optimized prompts for YouTube thumbnails
THUMBNAIL_PRESETS = {
    'reaction': {
        'name': 'Reaction',
        'description': 'Expressive reaction face with dramatic lighting',
        'icon': '😱',
        'prompt_template': '''Create a YouTube thumbnail in 16:9 aspect ratio with a dramatic reaction style:
- Feature the person from the reference image with an exaggerated shocked and surprised expression
- Use high contrast lighting with dramatic shadows
- Background should be a gradient with bright colors and abstract shapes
- Add visual impact elements like motion blur or zoom effects
- Style: Bold, attention-grabbing, high energy
{text_instruction}
{custom_instructions}'''
    },
    'tutorial': {
        'name': 'Tutorial',
        'description': 'Clean, professional tutorial thumbnail',
        'icon': '📚',
        'prompt_template': '''Create a YouTube thumbnail in 16:9 aspect ratio with a professional tutorial style:
- Feature the person from the reference image looking friendly and approachable
- Clean, well-lit studio background with subtle tech elements
- Professional color grading with blue and white professional tones
- Include visual elements that suggest learning or education
- Style: Clean, professional, trustworthy
{text_instruction}
{custom_instructions}'''
    },
    'gaming': {
        'name': 'Gaming',
        'description': 'Dynamic gaming-style thumbnail with effects',
        'icon': '🎮',
        'prompt_template': '''Create a YouTube thumbnail in 16:9 aspect ratio with an epic gaming style:
- Feature the person from the reference image with an intense, focused expression
- Dynamic background with game characters, explosions, and action scenes
- Neon lighting effects, lens flares, and particle effects
- High contrast colors with neon blue and purple accents
- Style: Epic, dynamic, high-energy gaming aesthetic
{text_instruction}
{custom_instructions}'''
    },
    'vlog': {
        'name': 'Vlog',
        'description': 'Casual, lifestyle vlog thumbnail',
        'icon': '📹',
        'prompt_template': '''Create a YouTube thumbnail in 16:9 aspect ratio with a lifestyle vlog style:
- Feature the person from the reference image with a natural, candid expression
- Lifestyle background showing an interesting travel or lifestyle setting
- Warm, inviting color grading with natural lighting
- Authentic, relatable aesthetic
- Style: Casual, authentic, inviting
{text_instruction}
{custom_instructions}'''
    },
    'clickbait': {
        'name': 'Clickbait',
        'description': 'Maximum impact attention-grabbing thumbnail',
        'icon': '🔥',
        'prompt_template': '''Create a YouTube thumbnail in 16:9 aspect ratio with maximum clickbait impact:
- Feature the person from the reference image with a shocked/surprised expression, mouth open
- Bright, saturated colors with red and yellow accents
- Add dramatic elements like arrows, circles, or spotlight effects
- High contrast, oversaturated look
- Background with money, explosions, or shocking imagery
- Style: Maximum impact, impossible to ignore, viral-worthy
{text_instruction}
{custom_instructions}'''
    }
}


def _build_thumbnail_prompt(user_prompt: str, preset: str = None, thumbnail_text: str = None) -> str:
    """Build the final thumbnail generation prompt."""

    # Start with base instructions for YouTube thumbnails
    base_instructions = """Generate a YouTube thumbnail image in exactly 16:9 aspect ratio (1280x720 pixels).
The thumbnail must be eye-catching, professional, and optimized for YouTube's thumbnail display."""

    # Add text rendering instructions if text is provided
    text_instruction = ""
    if thumbnail_text:
        text_instruction = f"""
IMPORTANT - Render the following text directly on the thumbnail:
Text to display: "{thumbnail_text}"
- Make the text large, bold, and highly readable
- Use contrasting colors so text stands out against the background
- Position text prominently (typically upper third or center)
- Use a bold, impactful font style
- Add text effects like outline, shadow, or glow for visibility"""

    # If preset is selected, use preset template
    if preset and preset in THUMBNAIL_PRESETS:
        preset_config = THUMBNAIL_PRESETS[preset]
        template = preset_config['prompt_template']

        # Build the prompt from template
        final_prompt = template.format(
            text_instruction=text_instruction,
            custom_instructions=f"\nAdditional instructions: {user_prompt}" if user_prompt else ""
        )
        return f"{base_instructions}\n\n{final_prompt}"

    # No preset - use user's full prompt
    full_prompt = f"{base_instructions}\n\n{user_prompt}"
    if text_instruction:
        full_prompt += f"\n{text_instruction}"

    return full_prompt


@youtube_thumbnail_bp.route('/thumbnail/generate', methods=['POST'])
def generate_thumbnail():
    """
    Generate a YouTube thumbnail with face reference.

    Request body:
        {
            "face_image_id": "uuid" (optional, use if from gallery),
            "face_image_path": "C:/path/to/face.jpg" (optional, use if local file),
            "prompt": "Full description of the thumbnail" (required if no preset),
            "preset": "reaction" (optional, one of the preset keys),
            "thumbnail_text": "Text to render on thumbnail" (optional),
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional)
        }

    Response:
        {
            "success": true,
            "image_id": "uuid",
            "image_url": "/api/gallery/uuid/image",
            "metadata": { ... },
            "preset_used": "reaction"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        face_image_id = data.get('face_image_id')
        face_image_path = data.get('face_image_path')
        prompt = data.get('prompt', '').strip()
        preset = data.get('preset')
        thumbnail_text = data.get('thumbnail_text', '').strip()

        # Validate: either prompt or preset is required
        if not prompt and not preset:
            return jsonify({'success': False, 'error': 'Either prompt or preset is required'}), 400

        # Build the final prompt
        final_prompt = _build_thumbnail_prompt(prompt, preset, thumbnail_text)

        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')

        # Get face reference image bytes if provided
        face_reference = None
        parent_id = None

        if face_image_id:
            # Load from gallery
            image_path = gallery_service.get_image_path(face_image_id)
            if not image_path:
                return jsonify({'success': False, 'error': 'Face reference image not found in gallery'}), 404

            with open(image_path, 'rb') as f:
                face_reference = f.read()
            parent_id = face_image_id

        elif face_image_path:
            # Load from local file
            from pathlib import Path
            path = Path(face_image_path)
            if not path.exists():
                return jsonify({'success': False, 'error': 'Face reference file not found'}), 404

            with open(path, 'rb') as f:
                face_reference = f.read()

            # Import the source image first
            try:
                imported = gallery_service.import_image(face_image_path)
                parent_id = imported['id']
            except Exception as e:
                return jsonify({'success': False, 'error': f'Failed to import face image: {str(e)}'}), 400

        # Generate thumbnail with or without face reference
        if face_reference:
            result = image_service.generate_with_reference(
                prompt=final_prompt,
                reference_images=[face_reference],
                model=model,
                provider=provider,
                aspect_ratio='16:9'
            )
        else:
            # Generate without face reference
            result = image_service.generate_image(
                prompt=final_prompt,
                model=model,
                provider=provider,
                aspect_ratio='16:9'
            )

        if not result['success']:
            return jsonify(result), 400

        # Save to gallery
        image_bytes = result['image_bytes']

        # Truncate prompt for storage if too long
        stored_prompt = final_prompt[:500] + ('...' if len(final_prompt) > 500 else '')

        metadata = gallery_service.save_image(
            image_bytes=image_bytes,
            prompt=stored_prompt,
            model=model,
            image_type='thumbnail',
            parent_id=parent_id,
            source_type='face_reference' if face_reference else 'generated'
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


@youtube_thumbnail_bp.route('/thumbnail/presets', methods=['GET'])
def get_presets():
    """
    Get available thumbnail style presets.

    Response:
        {
            "success": true,
            "presets": [
                {
                    "id": "reaction",
                    "name": "Reaction",
                    "description": "Expressive reaction face...",
                    "icon": "😱"
                },
                ...
            ]
        }
    """
    presets = [
        {
            'id': key,
            'name': value['name'],
            'description': value['description'],
            'icon': value['icon']
        }
        for key, value in THUMBNAIL_PRESETS.items()
    ]
    return jsonify({'success': True, 'presets': presets})
