import os
import time
from SimplerLLM import ImageGenerator, ImageProvider, ImageSize

# ============================================
# CONFIGURATION
# ============================================
REFERENCE_IMAGE_PATH = "cartoon_me.png"  # Your cartoon-style full body reference image
OUTPUT_DIR = "character_shots"
MODEL = "gemini-2.5-flash-image-preview" #gemini-2.5-flash-image-preview
DELAY_BETWEEN_REQUESTS = 2  # seconds between API calls (rate limiting)

# ============================================
# TESTING OPTIONS
# ============================================
TEST_MODE = False                    # Set to False to generate all shots
TEST_CATEGORY = "gestures"             # Options: "mouth", "gestures", "faces", "poses"
TEST_LIMIT = 2                      # Number of images to generate in test mode

# ============================================
# SHOT DEFINITIONS
# ============================================

# Base prompt prefix for consistency
BASE_PROMPT = """Generate the exact same cartoon character from the reference image.
Maintain the same art style, colors, proportions, and character design exactly.
The character should be shown in full body view with a clean, simple background.

"""

# Mouth Shapes (8 shots) - for lip sync animation
MOUTH_SHAPES = [
    {
        "filename": "mouth_A.png",
        "prompt": BASE_PROMPT + """Show the character with their mouth wide open in an 'A' or 'Ah' shape.
        The jaw is dropped, mouth is fully open, teeth visible. This is the widest mouth opening.
        Expression: Speaking enthusiastically. Full body, front-facing view."""
    },
    {
        "filename": "mouth_E.png",
        "prompt": BASE_PROMPT + """Show the character with a wide horizontal smile shape for 'E' or 'Ee' sound.
        Lips stretched wide horizontally, teeth showing, slight smile shape.
        Expression: Saying 'cheese'. Full body, front-facing view."""
    },
    {
        "filename": "mouth_I.png",
        "prompt": BASE_PROMPT + """Show the character with a slight vertical mouth opening for 'I' sound.
        Mouth slightly open, more vertical than horizontal, relaxed.
        Expression: Calm speaking. Full body, front-facing view."""
    },
    {
        "filename": "mouth_O.png",
        "prompt": BASE_PROMPT + """Show the character with a round, circular open mouth for 'O' or 'Oh' sound.
        Lips formed into a perfect round circle, like saying 'oh' or 'wow'.
        Expression: Surprised or saying 'oh'. Full body, front-facing view."""
    },
    {
        "filename": "mouth_U.png",
        "prompt": BASE_PROMPT + """Show the character with pursed, forward-pushed lips for 'U' or 'Oo' sound.
        Lips pushed forward in a small circle, like blowing or saying 'ooo'.
        Expression: Thoughtful or saying 'you'. Full body, front-facing view."""
    },
    {
        "filename": "mouth_M.png",
        "prompt": BASE_PROMPT + """Show the character with closed, pressed together lips for 'M', 'B', 'P' sounds.
        Lips completely closed and slightly pressed together, neutral position.
        Expression: Humming or about to speak. Full body, front-facing view."""
    },
    {
        "filename": "mouth_F.png",
        "prompt": BASE_PROMPT + """Show the character with lower lip tucked under upper teeth for 'F' or 'V' sounds.
        Upper teeth gently biting or touching the lower lip.
        Expression: Saying 'five' or 'very'. Full body, front-facing view."""
    },
    {
        "filename": "mouth_L.png",
        "prompt": BASE_PROMPT + """Show the character with tongue slightly visible for 'L' or 'Th' sounds.
        Mouth slightly open, tongue tip touching or near upper teeth/roof of mouth.
        Expression: Saying 'la' or 'the'. Full body, front-facing view."""
    },
]

# Hand/Arm Gestures (10 shots)
GESTURES = [
    {
        "filename": "gesture_point.png",
        "prompt": BASE_PROMPT + """Show the character pointing forward with their index finger.
        One arm extended forward, index finger pointing directly at the viewer.
        Confident, engaging pose. Full body, front-facing view."""
    },
    {
        "filename": "gesture_open_palm.png",
        "prompt": BASE_PROMPT + """Show the character with an open palm facing forward.
        One hand raised with palm facing the viewer, fingers spread, presenting or explaining gesture.
        Professional, welcoming pose. Full body, front-facing view."""
    },
    {
        "filename": "gesture_shrug.png",
        "prompt": BASE_PROMPT + """Show the character doing a shrug gesture.
        Both shoulders raised, both palms facing up and out to the sides, 'I don't know' pose.
        Questioning expression. Full body, front-facing view."""
    },
    {
        "filename": "gesture_count_1.png",
        "prompt": BASE_PROMPT + """Show the character holding up 1 finger to indicate 'one' or 'first'.
        One hand raised with just the index finger extended upward, other fingers closed.
        Teaching or listing pose. Full body, front-facing view."""
    },
    {
        "filename": "gesture_count_2.png",
        "prompt": BASE_PROMPT + """Show the character holding up 2 fingers to indicate 'two' or 'second'.
        One hand raised with index and middle finger extended in a V or peace sign, other fingers closed.
        Teaching or listing pose. Full body, front-facing view."""
    },
    {
        "filename": "gesture_count_3.png",
        "prompt": BASE_PROMPT + """Show the character holding up 3 fingers to indicate 'three' or 'third'.
        One hand raised with three fingers extended (index, middle, ring), thumb and pinky closed.
        Teaching or listing pose. Full body, front-facing view."""
    },
    {
        "filename": "gesture_stop.png",
        "prompt": BASE_PROMPT + """Show the character making a 'stop' hand signal.
        One arm extended forward, palm facing out flat, fingers together in a stop gesture.
        Authoritative, clear signal. Full body, front-facing view."""
    },
    {
        "filename": "gesture_thumbs_up.png",
        "prompt": BASE_PROMPT + """Show the character giving a thumbs up.
        One hand making a fist with thumb extended upward, positive approval gesture.
        Happy, encouraging expression. Full body, front-facing view."""
    },
    {
        "filename": "gesture_wave.png",
        "prompt": BASE_PROMPT + """Show the character waving hello.
        One arm raised with hand open, mid-wave motion, friendly greeting gesture.
        Warm, welcoming expression. Full body, front-facing view."""
    },
    {
        "filename": "gesture_crossed_arms.png",
        "prompt": BASE_PROMPT + """Show the character with arms crossed over their chest.
        Both arms folded across the chest, confident or contemplative stance.
        Confident, authoritative pose. Full body, front-facing view."""
    },
]

# Facial States (6 shots)
FACIAL_STATES = [
    {
        "filename": "face_neutral.png",
        "prompt": BASE_PROMPT + """Show the character with a completely neutral facial expression.
        Relaxed face, no strong emotion, calm and composed default state.
        Mouth closed, eyes normal. Full body, front-facing view."""
    },
    {
        "filename": "face_smile.png",
        "prompt": BASE_PROMPT + """Show the character with a warm, genuine smile.
        Big happy smile, eyes slightly squinted from smiling, radiating joy.
        Friendly and approachable. Full body, front-facing view."""
    },
    {
        "filename": "face_serious.png",
        "prompt": BASE_PROMPT + """Show the character with a serious, determined expression.
        Focused eyes, slight frown, mouth set firmly, conveying importance.
        Professional and intense. Full body, front-facing view."""
    },
    {
        "filename": "face_skeptical.png",
        "prompt": BASE_PROMPT + """Show the character with a skeptical expression.
        One eyebrow raised, slight smirk or frown, questioning look.
        Doubtful or unconvinced attitude. Full body, front-facing view."""
    },
    {
        "filename": "face_surprised.png",
        "prompt": BASE_PROMPT + """Show the character with a surprised expression.
        Wide open eyes, raised eyebrows, mouth slightly open in an 'O' shape.
        Shocked or amazed reaction. Full body, front-facing view."""
    },
    {
        "filename": "face_sad.png",
        "prompt": BASE_PROMPT + """Show the character with a sad expression.
        Downturned mouth, droopy eyes, slightly lowered head, melancholy look.
        Disappointed or unhappy mood. Full body, front-facing view."""
    },
]

# Body Poses (6 shots)
BODY_POSES = [
    {
        "filename": "pose_stand.png",
        "prompt": BASE_PROMPT + """Show the character standing straight in a confident stance.
        Upright posture, shoulders back, arms naturally at sides.
        Professional, confident default standing pose. Full body, front-facing view."""
    },
    {
        "filename": "pose_lean.png",
        "prompt": BASE_PROMPT + """Show the character leaning casually to one side.
        Relaxed posture, weight shifted to one leg, maybe arms crossed or one hand on hip.
        Casual, approachable stance. Full body, front-facing view."""
    },
    {
        "filename": "pose_walk.png",
        "prompt": BASE_PROMPT + """Show the character in a mid-walk pose.
        One leg forward, one back, arms in natural walking motion, body in motion.
        Dynamic walking animation frame. Full body, slight angle view."""
    },
    {
        "filename": "pose_thinking.png",
        "prompt": BASE_PROMPT + """Show the character in a thinking pose.
        One hand on chin or touching face, slightly tilted head, contemplative expression.
        Deep in thought, pondering. Full body, front-facing view."""
    },
    {
        "filename": "pose_writing.png",
        "prompt": BASE_PROMPT + """Show the character in a writing or typing gesture.
        Hands positioned as if writing on a notepad or typing on keyboard.
        Focused, productive pose. Full body, front-facing view."""
    },
    {
        "filename": "pose_explaining.png",
        "prompt": BASE_PROMPT + """Show the character in an explaining pose.
        Both hands gesturing outward, animated body language as if presenting or teaching.
        Engaged, expressive speaking pose. Full body, front-facing view."""
    },
]


def create_output_directories():
    """Create output directory structure."""
    dirs = [
        os.path.join(OUTPUT_DIR, "mouth"),
        os.path.join(OUTPUT_DIR, "gestures"),
        os.path.join(OUTPUT_DIR, "faces"),
        os.path.join(OUTPUT_DIR, "poses"),
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Output directories created in: {OUTPUT_DIR}/")


def generate_shots(img_gen, shots, category_name, output_subdir):
    """Generate all shots in a category."""
    print(f"\n{'='*60}")
    print(f"Generating {category_name} ({len(shots)} shots)")
    print(f"{'='*60}")

    successful = 0
    failed = 0

    for i, shot in enumerate(shots, 1):
        output_path = os.path.join(OUTPUT_DIR, output_subdir, shot["filename"])
        print(f"\n[{i}/{len(shots)}] Generating: {shot['filename']}")

        try:
            result_path = img_gen.generate_image(
                prompt=shot["prompt"],
                reference_images=[REFERENCE_IMAGE_PATH],
                size=ImageSize.PORTRAIT_3_4,  # 3:4 aspect ratio for character shots
                output_format="file",
                output_path=output_path,
                model=MODEL,
            )
            print(f"    Saved: {result_path}")
            successful += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        # Rate limiting delay (skip after last item)
        if i < len(shots):
            time.sleep(DELAY_BETWEEN_REQUESTS)

    return successful, failed


def main():
    print("=" * 60)
    print("Character Shot Generator")
    print("=" * 60)
    print(f"\nReference image: {REFERENCE_IMAGE_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model: {MODEL}")

    if TEST_MODE:
        print(f"\n*** TEST MODE: {TEST_CATEGORY} category, {TEST_LIMIT} images ***")

    # Verify reference image exists
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"\nERROR: Reference image not found: {REFERENCE_IMAGE_PATH}")
        print("Please update REFERENCE_IMAGE_PATH to point to your cartoon reference image.")
        return

    # Create output directories
    create_output_directories()

    # Create image generator
    print("\nInitializing image generator...")
    img_gen = ImageGenerator.create(provider=ImageProvider.GOOGLE_GEMINI)

    # Track totals
    total_successful = 0
    total_failed = 0

    # All categories
    all_categories = {
        "mouth": (MOUTH_SHAPES, "Mouth Shapes", "mouth"),
        "gestures": (GESTURES, "Hand/Arm Gestures", "gestures"),
        "faces": (FACIAL_STATES, "Facial States", "faces"),
        "poses": (BODY_POSES, "Body Poses", "poses"),
    }

    # Select categories to run
    if TEST_MODE:
        if TEST_CATEGORY not in all_categories:
            print(f"\nERROR: Invalid TEST_CATEGORY '{TEST_CATEGORY}'")
            print(f"Valid options: {', '.join(all_categories.keys())}")
            return
        shots, name, subdir = all_categories[TEST_CATEGORY]
        categories = [(shots[:TEST_LIMIT], name, subdir)]
    else:
        categories = list(all_categories.values())

    for shots, name, subdir in categories:
        successful, failed = generate_shots(img_gen, shots, name, subdir)
        total_successful += successful
        total_failed += failed

    # Final summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total generated: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"\nOutput saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
