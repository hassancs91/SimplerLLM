"""
Portrait Studio API - Generate images with face reference for character consistency
"""
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service

portrait_studio_bp = Blueprint('portrait_studio', __name__)


# ============================================
# Portrait Transformation Presets
# ============================================

PORTRAIT_PRESETS = {
    # Age Transformations
    'elderly': {
        'name': '80 Years Old',
        'description': 'Age to look elderly with wrinkles',
        'icon': '👴',
        'prompt_template': '''Transform the person from the reference image to appear as an 80-year-old version of themselves:
- Add realistic aging: deep wrinkles, age spots, sagging skin
- Gray/white hair, possibly thinning
- Maintain their core facial structure and recognizable features
- Natural elderly expression, wise and dignified
- Realistic lighting and skin texture
{custom_instructions}'''
    },
    'young': {
        'name': 'Young/Teen',
        'description': 'Make the person look like a teenager',
        'icon': '👦',
        'prompt_template': '''Transform the person from the reference image to appear as a teenager (14-17 years old):
- Smooth, youthful skin without wrinkles
- Fuller face with softer features
- Bright, clear eyes
- Maintain recognizable facial characteristics
- Natural teenage expression and energy
{custom_instructions}'''
    },
    'baby': {
        'name': 'Baby Face',
        'description': 'Transform into a cute baby version',
        'icon': '👶',
        'prompt_template': '''Transform the person from the reference image into an adorable baby version (1-2 years old):
- Chubby cheeks and small features
- Big, expressive eyes
- Soft, smooth baby skin
- Adorable, innocent expression
- Maintain some recognizable features from the original
- Cute baby proportions
{custom_instructions}'''
    },

    # Funny/Horror
    'zombie': {
        'name': 'Zombie',
        'description': 'Spooky zombie transformation',
        'icon': '🧟',
        'prompt_template': '''Transform the person from the reference image into a terrifying zombie:
- Pale, decaying skin with greenish/grayish tinge
- Dark circles around sunken, lifeless eyes
- Some visible decay or wounds
- Torn or disheveled clothing/hair
- Spooky, horror atmosphere
- Maintain recognizable features despite the transformation
{custom_instructions}'''
    },

    # Art Styles
    'pixar': {
        'name': 'Cartoon/Pixar',
        'description': '3D animated Pixar-style character',
        'icon': '🎬',
        'prompt_template': '''Transform the person from the reference image into a Pixar/Disney 3D animated character:
- Stylized proportions with slightly larger eyes
- Expressive, appealing features
- Smooth, clean 3D rendering style
- Vibrant colors and soft, appealing lighting
- Maintain personality and recognizable features
- High-quality CGI movie aesthetic like Pixar films
{custom_instructions}'''
    },
    'anime': {
        'name': 'Anime',
        'description': 'Japanese anime/manga art style',
        'icon': '🎌',
        'prompt_template': '''Transform the person from the reference image into anime/manga style:
- Large, expressive eyes with detailed highlights and reflections
- Simplified but expressive facial features
- Clean linework and cel-shaded coloring
- Stylized anime hair with dynamic flow
- Japanese animation aesthetic
- Maintain the person's essence and recognizable traits
{custom_instructions}'''
    },
    'oil_painting': {
        'name': 'Oil Painting',
        'description': 'Classical oil painting portrait',
        'icon': '🖼️',
        'prompt_template': '''Transform the person from the reference image into a classical oil painting portrait:
- Rich, textured brushstrokes visible in the painting
- Renaissance or Baroque style lighting with dramatic chiaroscuro
- Deep, warm color palette with earth tones
- Elegant, timeless composition
- Museum-worthy fine art quality
- Maintain the subject's likeness and character
{custom_instructions}'''
    },
    'blocks': {
        'name': '3D Blocks',
        'description': 'Minecraft/voxel block style',
        'icon': '🧱',
        'prompt_template': '''Transform the person from the reference image into Minecraft/voxel block style:
- Cubic, pixelated features made of 3D blocks
- Block-based construction like Minecraft characters
- 8-bit/16-bit aesthetic with limited color palette
- Maintain recognizable features in blocky voxel form
- Game-like rendering with clean edges
- Fun, playful video game aesthetic
{custom_instructions}'''
    },

    # Character/Costume
    'superhero': {
        'name': 'Superhero',
        'description': 'Transform into a superhero character',
        'icon': '🦸',
        'prompt_template': '''Transform the person from the reference image into a powerful superhero:
- Heroic pose with confident, determined expression
- Custom superhero costume (colorful, with cape optional)
- Dramatic lighting with lens flares and energy effects
- Action-ready stance showing strength
- Comic book hero aesthetic
- Maintain the person's facial features
- Epic, powerful atmosphere
{custom_instructions}'''
    },
    'knight': {
        'name': 'Medieval Knight',
        'description': 'Noble knight in shining armor',
        'icon': '⚔️',
        'prompt_template': '''Transform the person from the reference image into a medieval knight:
- Detailed plate armor in silver or gold
- Noble, brave expression
- Sword, shield, or lance as props
- Castle or battlefield background
- Historical fantasy aesthetic
- Maintain the person's facial features visible through the helmet visor or with helmet off
- Epic medieval warrior appearance
{custom_instructions}'''
    },
    'astronaut': {
        'name': 'Astronaut',
        'description': 'Space explorer in a spacesuit',
        'icon': '🚀',
        'prompt_template': '''Transform the person from the reference image into an astronaut:
- Detailed NASA-style space suit with patches and equipment
- Helmet with visor up showing the face clearly
- Space, stars, Earth, or moon in background
- Adventurous, determined explorer expression
- Realistic sci-fi aesthetic
- Maintain the person's facial features
- Inspiring space exploration atmosphere
{custom_instructions}'''
    }
}


def _build_portrait_prompt(preset_id: str, custom_prompt: str) -> str:
    """
    Build the final prompt by combining preset template with custom instructions.

    Args:
        preset_id: The preset ID to use (or None for custom prompt only)
        custom_prompt: User's custom prompt/instructions

    Returns:
        The complete prompt string for image generation
    """
    if preset_id and preset_id in PORTRAIT_PRESETS:
        template = PORTRAIT_PRESETS[preset_id]['prompt_template']
        custom_part = f"\nAdditional instructions: {custom_prompt}" if custom_prompt else ""
        return template.format(custom_instructions=custom_part)
    return custom_prompt


@portrait_studio_bp.route('/portrait/presets', methods=['GET'])
def get_portrait_presets():
    """
    Get available portrait transformation presets.

    Response:
        {
            "success": true,
            "presets": [
                {"id": "elderly", "name": "80 Years Old", "description": "...", "icon": "👴"},
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
        for key, value in PORTRAIT_PRESETS.items()
    ]
    return jsonify({'success': True, 'presets': presets})


@portrait_studio_bp.route('/portrait/generate', methods=['POST'])
def generate_portrait():
    """
    Generate an image with optional face reference for character consistency.

    Request body:
        {
            "face_image_id": "uuid" (optional, use if from gallery),
            "face_image_path": "path" (optional, use if local file),
            "prompt": "description" (optional if preset provided),
            "preset": "preset_id" (optional, e.g. 'elderly', 'pixar', 'zombie'),
            "aspect_ratio": "1:1" (optional, default 1:1),
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional)
        }

    Response:
        {
            "success": true,
            "image_id": "uuid",
            "image_url": "/api/gallery/uuid/image",
            "preset_used": "preset_id" (if preset was used),
            "metadata": { ... }
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        face_image_id = data.get('face_image_id')
        face_image_path = data.get('face_image_path')
        custom_prompt = data.get('prompt', '').strip()
        preset_id = data.get('preset')
        aspect_ratio = data.get('aspect_ratio', '1:1')
        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')

        # Validate: either preset or prompt is required
        if not preset_id and not custom_prompt:
            return jsonify({'success': False, 'error': 'Either a preset or prompt is required'}), 400

        # Validate preset if provided
        if preset_id and preset_id not in PORTRAIT_PRESETS:
            return jsonify({'success': False, 'error': f'Invalid preset: {preset_id}'}), 400

        # Build the final prompt
        prompt = _build_portrait_prompt(preset_id, custom_prompt)

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

        # Generate image with or without face reference
        if face_reference:
            result = image_service.generate_with_reference(
                prompt=prompt,
                reference_images=[face_reference],
                model=model,
                provider=provider,
                aspect_ratio=aspect_ratio
            )
        else:
            # Generate without face reference
            result = image_service.generate_image(
                prompt=prompt,
                model=model,
                provider=provider,
                aspect_ratio=aspect_ratio
            )

        if not result['success']:
            return jsonify(result), 400

        # Save to gallery
        image_bytes = result['image_bytes']

        # Truncate prompt for storage if too long
        stored_prompt = prompt[:500] + ('...' if len(prompt) > 500 else '')

        metadata = gallery_service.save_image(
            image_bytes=image_bytes,
            prompt=stored_prompt,
            model=model,
            image_type='portrait',
            parent_id=parent_id,
            source_type='face_reference' if face_reference else 'generated'
        )

        response = {
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'metadata': metadata
        }

        # Include preset info if used
        if preset_id:
            response['preset_used'] = preset_id

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
