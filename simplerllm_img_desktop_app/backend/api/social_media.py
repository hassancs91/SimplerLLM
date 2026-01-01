"""
Social Media Post Generator API - Generate platform-optimized social media images
"""
from flask import Blueprint, request, jsonify
from services.shared import image_service, gallery_service

social_media_bp = Blueprint('social_media', __name__)

# Platform presets with aspect ratios optimized for each social network
PLATFORM_PRESETS = {
    'instagram_feed': {
        'name': 'Instagram Feed',
        'platform': 'instagram',
        'icon': '📷',
        'aspect_ratio': '1:1',
        'dimensions': '1080x1080',
        'description': 'Square post for Instagram feed'
    },
    'instagram_portrait': {
        'name': 'Instagram Portrait',
        'platform': 'instagram',
        'icon': '📷',
        'aspect_ratio': '9:16',
        'dimensions': '1080x1350',
        'description': 'Vertical post for Instagram feed'
    },
    'instagram_story': {
        'name': 'Instagram Story/Reel',
        'platform': 'instagram',
        'icon': '📱',
        'aspect_ratio': '9:16',
        'dimensions': '1080x1920',
        'description': 'Full-screen vertical for Stories/Reels'
    },
    'facebook_link': {
        'name': 'Facebook Link',
        'platform': 'facebook',
        'icon': '📘',
        'aspect_ratio': '16:9',
        'dimensions': '1200x628',
        'description': 'Landscape preview for link shares'
    },
    'facebook_post': {
        'name': 'Facebook Post',
        'platform': 'facebook',
        'icon': '📘',
        'aspect_ratio': '1:1',
        'dimensions': '1200x1200',
        'description': 'Square post for Facebook feed'
    },
    'twitter_landscape': {
        'name': 'X/Twitter Landscape',
        'platform': 'twitter',
        'icon': '🐦',
        'aspect_ratio': '16:9',
        'dimensions': '1600x900',
        'description': 'Landscape image for X/Twitter'
    },
    'twitter_square': {
        'name': 'X/Twitter Square',
        'platform': 'twitter',
        'icon': '🐦',
        'aspect_ratio': '1:1',
        'dimensions': '1080x1080',
        'description': 'Square image for X/Twitter'
    },
    'linkedin_link': {
        'name': 'LinkedIn Link',
        'platform': 'linkedin',
        'icon': '💼',
        'aspect_ratio': '16:9',
        'dimensions': '1200x628',
        'description': 'Landscape preview for link shares'
    },
    'linkedin_post': {
        'name': 'LinkedIn Post',
        'platform': 'linkedin',
        'icon': '💼',
        'aspect_ratio': '1:1',
        'dimensions': '1200x1200',
        'description': 'Square post for LinkedIn feed'
    }
}

# Content type presets with prompt modifiers for different post styles
CONTENT_TYPE_PRESETS = {
    'promotional': {
        'name': 'Promotional',
        'icon': '🎯',
        'description': 'Product/service promotion with call-to-action',
        'prompt_modifier': 'promotional marketing post style, eye-catching design, vibrant colors, clear call-to-action elements, modern professional marketing aesthetic, bold impactful visuals'
    },
    'announcement': {
        'name': 'Announcement',
        'icon': '📢',
        'description': 'News, updates, or announcements',
        'prompt_modifier': 'announcement style design, bold typography emphasis, attention-grabbing layout, clean professional presentation, news-worthy visual impact, important information highlight'
    },
    'quote': {
        'name': 'Quote/Inspirational',
        'icon': '💬',
        'description': 'Inspirational quotes or text-focused posts',
        'prompt_modifier': 'inspirational quote design, elegant typography focus, motivational aesthetic, clean minimalist background, sophisticated layout, quotation marks as design element'
    },
    'product_showcase': {
        'name': 'Product Showcase',
        'icon': '🛍️',
        'description': 'Highlight products with professional styling',
        'prompt_modifier': 'product photography style, professional studio lighting, clean background, e-commerce aesthetic, high-quality product presentation, commercial photography look'
    },
    'event': {
        'name': 'Event',
        'icon': '🎉',
        'description': 'Event promotions and invitations',
        'prompt_modifier': 'event promotional design, festive celebration elements, date and time layout space, invitation style aesthetic, event marketing visual, exciting atmosphere'
    },
    'educational': {
        'name': 'Educational/Tips',
        'icon': '📚',
        'description': 'How-to content, tips, or tutorials',
        'prompt_modifier': 'educational infographic style, step-by-step visual layout, informative design elements, clean modern aesthetic, knowledge sharing visual, easy to understand presentation'
    },
    'behind_scenes': {
        'name': 'Behind the Scenes',
        'icon': '🎬',
        'description': 'Authentic, candid content',
        'prompt_modifier': 'behind the scenes aesthetic, authentic candid style, natural lighting, genuine moment capture, documentary style photography, relatable authentic feel'
    },
    'testimonial': {
        'name': 'Testimonial',
        'icon': '⭐',
        'description': 'Customer reviews and testimonials',
        'prompt_modifier': 'testimonial design style, customer review visual layout, star rating elements, trust-building aesthetic, social proof presentation, credibility focused design'
    }
}


def _build_social_media_prompt(user_prompt: str, platform_preset: str, content_type: str = None) -> str:
    """Build the final generation prompt with platform and content type modifiers."""

    platform = PLATFORM_PRESETS.get(platform_preset, {})
    platform_name = platform.get('name', 'social media')
    dimensions = platform.get('dimensions', '1080x1080')
    aspect_ratio = platform.get('aspect_ratio', '1:1')

    # Base instructions for social media optimization
    base_prompt = f"""Create a professional social media image optimized for {platform_name}.
Target dimensions: {dimensions}
Aspect ratio: {aspect_ratio}

The image should be visually striking, scroll-stopping, and optimized for social media engagement.

User's description: {user_prompt}"""

    # Add content type modifier if specified
    if content_type and content_type in CONTENT_TYPE_PRESETS:
        modifier = CONTENT_TYPE_PRESETS[content_type]['prompt_modifier']
        base_prompt += f"\n\nStyle direction: {modifier}"

    return base_prompt


@social_media_bp.route('/social-media/presets', methods=['GET'])
def get_presets():
    """
    Get available platform and content type presets.

    Response:
        {
            "success": true,
            "platforms": [...],
            "content_types": [...]
        }
    """
    platforms = [
        {
            'id': key,
            'name': value['name'],
            'platform': value['platform'],
            'icon': value['icon'],
            'aspect_ratio': value['aspect_ratio'],
            'dimensions': value['dimensions'],
            'description': value['description']
        }
        for key, value in PLATFORM_PRESETS.items()
    ]

    content_types = [
        {
            'id': key,
            'name': value['name'],
            'icon': value['icon'],
            'description': value['description']
        }
        for key, value in CONTENT_TYPE_PRESETS.items()
    ]

    return jsonify({
        'success': True,
        'platforms': platforms,
        'content_types': content_types
    })


@social_media_bp.route('/social-media/generate', methods=['POST'])
def generate_social_media_post():
    """
    Generate a social media post image.

    Request body:
        {
            "prompt": "Description of desired image" (required),
            "platform_preset": "instagram_feed" (required, key from PLATFORM_PRESETS),
            "content_type": "promotional" (optional, key from CONTENT_TYPE_PRESETS),
            "reference_image_id": "uuid" (optional, gallery image for brand/style reference),
            "model": "gemini-2.5-flash-image-preview" (optional),
            "provider": "google" (optional)
        }

    Response:
        {
            "success": true,
            "image_id": "uuid",
            "image_url": "/api/gallery/uuid/image",
            "metadata": { ... },
            "platform_preset_used": "instagram_feed",
            "content_type_used": "promotional"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        platform_preset = data.get('platform_preset')
        content_type = data.get('content_type')
        reference_image_id = data.get('reference_image_id')
        model = data.get('model', 'gemini-2.5-flash-image-preview')
        provider = data.get('provider', 'google')

        # Validate required fields
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400

        if not platform_preset:
            return jsonify({'success': False, 'error': 'Platform preset is required'}), 400

        if platform_preset not in PLATFORM_PRESETS:
            return jsonify({'success': False, 'error': f'Invalid platform preset: {platform_preset}'}), 400

        if content_type and content_type not in CONTENT_TYPE_PRESETS:
            return jsonify({'success': False, 'error': f'Invalid content type: {content_type}'}), 400

        # Get aspect ratio from platform preset
        aspect_ratio = PLATFORM_PRESETS[platform_preset]['aspect_ratio']

        # Build the final prompt
        final_prompt = _build_social_media_prompt(prompt, platform_preset, content_type)

        # Get reference image bytes if provided
        reference_images = None
        parent_id = None

        if reference_image_id:
            image_path = gallery_service.get_image_path(reference_image_id)
            if not image_path:
                return jsonify({'success': False, 'error': 'Reference image not found in gallery'}), 404

            with open(image_path, 'rb') as f:
                reference_bytes = f.read()
            reference_images = [reference_bytes]
            parent_id = reference_image_id

        # Generate image
        if reference_images:
            result = image_service.generate_with_reference(
                prompt=final_prompt,
                reference_images=reference_images,
                model=model,
                provider=provider,
                aspect_ratio=aspect_ratio
            )
        else:
            result = image_service.generate_image(
                prompt=final_prompt,
                model=model,
                provider=provider,
                aspect_ratio=aspect_ratio
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
            image_type='social_media',
            parent_id=parent_id,
            source_type='brand_reference' if reference_images else 'generated'
        )

        return jsonify({
            'success': True,
            'image_id': metadata['id'],
            'image_url': f"/api/gallery/{metadata['id']}/image",
            'metadata': metadata,
            'platform_preset_used': platform_preset,
            'content_type_used': content_type
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
