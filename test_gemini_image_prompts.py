"""
Comprehensive Google Gemini Image Generation Test

Tests various image generation capabilities:
- Text rendering (readable labels)
- Anatomy accuracy (hands, fingers)
- Complex scenes with details
- Transparent materials + reflections
- Style control (2D vector vs photoreal)
- Portrait realism
- Reference-based identity consistency

Requirements:
- Set GEMINI_API_KEY environment variable

Usage:
    python test_gemini_image_prompts.py              # Run all prompts
    python test_gemini_image_prompts.py 1            # Run only prompt 1
    python test_gemini_image_prompts.py 1 3 5        # Run prompts 1, 3, and 5
    python test_gemini_image_prompts.py 7-10         # Run prompts 7 through 10 (reference-based)
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from SimplerLLM.image.generation import ImageGenerator, ImageProvider, ImageSize

# Create output directory
OUTPUT_DIR = "output/gemini_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model to use
MODEL_NAME = "gemini-3-pro-image-preview"

# ============================================================================
# TEXT-TO-IMAGE PROMPTS (No Reference)
# ============================================================================

TEXT_TO_IMAGE_PROMPTS = {
    "01_product_label": {
        "name": "Photoreal product + readable label",
        "prompt": 'Ultra-realistic studio photo of a clear glass bottle with condensation on a dark slate surface. The label text must be perfectly readable: "NANO vs CHAT" (top line) and "IMAGE MODEL TEST" (second line). Softbox reflections, shallow depth of field, 50mm look, high detail, clean composition.',
    },
    "02_hands_anatomy": {
        "name": "Hands + objects (hard anatomy)",
        "prompt": "Photoreal close-up of two hands tying a shoelace on a sneaker. Natural skin texture, realistic knuckles and fingernails, correct finger count, no deformities. Morning window light, crisp focus on the knot, blurred background.",
    },
    "03_complex_scene": {
        "name": "Complex scene with many small details",
        "prompt": 'A cozy home office desk scene at night: laptop with code editor, sticky notes, a small plant, coffee mug with steam, headphones, scattered cables, and a mechanical keyboard. Cinematic lighting, realistic reflections, tiny readable sticky note text: "ship it" and "fix bugs".',
    },
    "04_transparent_glass": {
        "name": "Transparent material + reflections",
        "prompt": "A transparent glass chess set on a glossy black table, with strong reflections and refractions. Dramatic rim lighting, high contrast, razor sharp edges, physically correct refraction, no melted shapes.",
    },
    "05_vector_style": {
        "name": "Precise style control",
        "prompt": "A clean 2D flat vector illustration (not 3D) of a rocket launching from a laptop screen. Minimal shadows, simple geometric shapes, limited color palette, crisp outlines, lots of negative space, modern UI-icon style.",
    },
    "06_portrait_realism": {
        "name": "Portrait realism + hair detail",
        "prompt": "Hyperreal portrait photo of an adult man outdoors in soft overcast light. Natural skin pores, realistic hair strands, subtle imperfections, shallow depth of field, 85mm portrait look. Neutral expression, no beauty filter, true-to-life color.",
    },
}

# ============================================================================
# REFERENCE-IMAGE PROMPTS (Using user's photo)
# ============================================================================

REFERENCE_IMAGE_PROMPTS = {
    "07_wardrobe_change": {
        "name": "Same person, different wardrobe (identity consistency)",
        "prompt": "Use the reference image of the person. Keep the face identity, age, and overall likeness the same. Change outfit to: black hoodie + simple watch. Put him in a modern tech studio with a soft teal accent light behind. Photoreal, sharp eyes, natural skin texture, clean background.",
    },
    "08_whiteboard_pose": {
        "name": "Same person, strong pose + prop (composition + hands)",
        "prompt": 'Use the reference image of the person. Keep identity and facial features consistent. Make him holding a whiteboard toward camera (both hands visible, correct fingers). The whiteboard text must be clean and readable: "SEE YOU NEXT WEEK". Bright, clean studio lighting, high contrast, thumbnail framing.',
    },
    "09_pixar_style": {
        "name": "Same person, stylized but controlled (style transfer test)",
        "prompt": "Use the reference image of the person. Keep identity consistent. Render as a high-end 3D cinematic character (Pixar-level realism, not cartoony), smooth skin shading, detailed hair, soft rim light, teal + navy color mood, clean background, sharp facial features.",
    },
    "10_extreme_closeup": {
        "name": "Extreme close-up portrait (macro skin detail)",
        "prompt": "Use the reference image of the person. Extreme close-up crop (eyes + nose bridge + cheeks), preserve identity exactly. Ultra-detailed pores, peach fuzz, realistic eyelashes and catchlights, natural skin texture, no smoothing, no plastic skin. Soft diffused light, 100mm macro look, shallow depth of field, true-to-life color.",
    },
}


def parse_prompt_numbers(args):
    """Parse command line arguments to get list of prompt numbers to run.

    Supports:
    - Single numbers: 1, 2, 3
    - Ranges: 7-10 (expands to 7, 8, 9, 10)

    Returns set of prompt numbers (1-10) or None to run all.
    """
    if not args:
        return None  # Run all prompts

    selected = set()
    for arg in args:
        if '-' in arg:
            # Handle range like "7-10"
            try:
                start, end = arg.split('-')
                for num in range(int(start), int(end) + 1):
                    if 1 <= num <= 10:
                        selected.add(num)
            except ValueError:
                print(f"Warning: Invalid range '{arg}', skipping")
        else:
            # Handle single number
            try:
                num = int(arg)
                if 1 <= num <= 10:
                    selected.add(num)
                else:
                    print(f"Warning: Prompt number {num} out of range (1-10), skipping")
            except ValueError:
                print(f"Warning: Invalid argument '{arg}', skipping")

    return selected if selected else None


def generate_text_to_image(gemini, key, config):
    """Generate image from text prompt without reference."""
    print(f"\n{'=' * 60}")
    print(f"[{key}] {config['name']}")
    print("=" * 60)
    print(f"Prompt: {config['prompt'][:100]}...")

    output_path = f"{OUTPUT_DIR}/{key}.png"

    try:
        response = gemini.generate_image(
            prompt=config["prompt"],
            size=ImageSize.VERTICAL,
            output_format="file",
            output_path=output_path,
            full_response=True,
        )

        print(f"  Status: SUCCESS")
        print(f"  Time: {response.process_time:.2f}s")
        print(f"  Output: {response.output_path}")
        if response.file_size:
            print(f"  Size: {response.file_size:,} bytes")
        if response.revised_prompt:
            print(f"  Model response: {response.revised_prompt[:80]}...")

        return {"status": "success", "path": output_path, "time": response.process_time}

    except Exception as e:
        print(f"  Status: FAILED")
        print(f"  Error: {e}")
        return {"status": "failed", "error": str(e)}


def generate_with_reference(gemini, key, config, reference_path):
    """Generate image using reference image for identity consistency."""
    print(f"\n{'=' * 60}")
    print(f"[{key}] {config['name']}")
    print("=" * 60)
    print(f"Reference: {reference_path}")
    print(f"Prompt: {config['prompt'][:100]}...")

    output_path = f"{OUTPUT_DIR}/{key}.png"

    try:
        response = gemini.generate_image(
            prompt=config["prompt"],
            size=ImageSize.VERTICAL,
            reference_images=[reference_path],
            output_format="file",
            output_path=output_path,
            full_response=True,
        )

        print(f"  Status: SUCCESS")
        print(f"  Time: {response.process_time:.2f}s")
        print(f"  Output: {response.output_path}")
        if response.file_size:
            print(f"  Size: {response.file_size:,} bytes")
        if response.revised_prompt:
            print(f"  Model response: {response.revised_prompt[:80]}...")

        return {"status": "success", "path": output_path, "time": response.process_time}

    except Exception as e:
        print(f"  Status: FAILED")
        print(f"  Error: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    print("=" * 70)
    print("  GOOGLE GEMINI IMAGE GENERATION - COMPREHENSIVE TEST")
    print("=" * 70)

    # Parse command line arguments for prompt filtering
    selected_prompts = parse_prompt_numbers(sys.argv[1:])

    if selected_prompts:
        print(f"\nRunning selected prompts: {sorted(selected_prompts)}")
    else:
        print("\nRunning all prompts (1-10)")

    # Check API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("\nERROR: GEMINI_API_KEY not set!")
        print("Please set it in your .env file or environment variables.")
        return

    print(f"\nModel: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}/")

    # Create generator
    gemini = ImageGenerator.create(
        provider=ImageProvider.GOOGLE_GEMINI,
        model_name=MODEL_NAME,
        verbose=True,
    )

    results = {}

    # Filter prompts based on selection
    text_prompts_to_run = {}
    ref_prompts_to_run = {}

    for key, config in TEXT_TO_IMAGE_PROMPTS.items():
        prompt_num = int(key.split("_")[0])  # Extract number from "01_product_label"
        if selected_prompts is None or prompt_num in selected_prompts:
            text_prompts_to_run[key] = config

    for key, config in REFERENCE_IMAGE_PROMPTS.items():
        prompt_num = int(key.split("_")[0])  # Extract number from "07_wardrobe_change"
        if selected_prompts is None or prompt_num in selected_prompts:
            ref_prompts_to_run[key] = config

    # ========================================================================
    # Part 1: Text-to-Image (no reference)
    # ========================================================================
    if text_prompts_to_run:
        print("\n" + "=" * 70)
        print("  PART 1: TEXT-TO-IMAGE PROMPTS (No Reference)")
        print("=" * 70)

        for key, config in text_prompts_to_run.items():
            results[key] = generate_text_to_image(gemini, key, config)

    # ========================================================================
    # Part 2: Reference-Image Prompts
    # ========================================================================
    if ref_prompts_to_run:
        print("\n" + "=" * 70)
        print("  PART 2: REFERENCE-IMAGE PROMPTS")
        print("=" * 70)

        # Ask for reference image path
        print("\nFor identity-consistent generation, please provide a reference photo.")
        reference_path = input("Enter path to reference image (or press Enter to skip): ").strip()

        if reference_path and os.path.exists(reference_path):
            print(f"\nUsing reference image: {reference_path}")

            for key, config in ref_prompts_to_run.items():
                results[key] = generate_with_reference(gemini, key, config, reference_path)
        else:
            if reference_path:
                print(f"\nWARNING: Reference image not found: {reference_path}")
            print("Skipping reference-image prompts.")
            for key in ref_prompts_to_run:
                results[key] = {"status": "skipped", "reason": "No reference image provided"}

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_time = 0

    for key, result in results.items():
        status = result["status"]
        if status == "success":
            success_count += 1
            total_time += result.get("time", 0)
            print(f"  [OK] {key}: {result['path']} ({result['time']:.1f}s)")
        elif status == "failed":
            failed_count += 1
            print(f"  [FAIL] {key}: {result.get('error', 'Unknown error')[:50]}")
        else:
            skipped_count += 1
            print(f"  [SKIP] {key}: {result.get('reason', 'Skipped')}")

    print(f"\n  Total: {success_count} success, {failed_count} failed, {skipped_count} skipped")
    if success_count > 0:
        print(f"  Total generation time: {total_time:.1f}s")
        print(f"  Average per image: {total_time / success_count:.1f}s")

    print("\n" + "=" * 70)
    print(f"  Check the '{OUTPUT_DIR}/' folder to view generated images!")
    print("=" * 70)


if __name__ == "__main__":
    main()
