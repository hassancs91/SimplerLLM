"""
Test script to compare Gemini and Seedream 4.5 image generation and editing.

Requirements:
- Set GEMINI_API_KEY environment variable for Google Gemini
- Set ARK_API_KEY environment variable for BytePlus Seedream

Usage:
    python test_seedream_vs_gemini.py
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from SimplerLLM.image.generation import ImageGenerator, ImageProvider, ImageSize

# Create output directory
os.makedirs("output", exist_ok=True)

# Common prompts for comparison
GENERATION_PROMPT = "A majestic golden retriever sitting in a sunlit meadow with wildflowers, photorealistic style"
EDIT_PROMPT = "Change the background to a snowy winter landscape while keeping the dog the same"

# Reference image URL for editing (we'll use the generated Seedream image URL)
# For Gemini editing, we need a local file, so we'll use the saved file


def test_gemini_generation():
    """Generate image with Google Gemini"""
    print("\n" + "=" * 50)
    print("GEMINI: Generating image...")
    print("=" * 50)

    try:
        gemini = ImageGenerator.create(
            provider=ImageProvider.GOOGLE_GEMINI,
            verbose=True
        )

        response = gemini.generate_image(
            prompt=GENERATION_PROMPT,
            size=ImageSize.SQUARE,
            output_format="file",
            output_path="output/gemini_generated.png",
            full_response=True
        )

        print(f"Gemini generation completed in {response.process_time:.2f}s")
        print(f"Saved to: {response.output_path}")
        return response.output_path

    except Exception as e:
        print(f"Gemini generation failed: {e}")
        return None


def test_seedream_generation():
    """Generate image with Seedream 4.5"""
    print("\n" + "=" * 50)
    print("SEEDREAM: Generating image...")
    print("=" * 50)

    try:
        seedream = ImageGenerator.create(
            provider=ImageProvider.SEEDREAM,
            verbose=True
        )

        response = seedream.generate_image(
            prompt=GENERATION_PROMPT,
            size="2K",
            watermark=False,
            output_format="file",
            output_path="output/seedream_generated.png",
            full_response=True
        )

        print(f"Seedream generation completed in {response.process_time:.2f}s")
        print(f"Saved to: {response.output_path}")

        # Also get the URL for editing test
        url_response = seedream.generate_image(
            prompt=GENERATION_PROMPT,
            size="2K",
            watermark=False,
            output_format="url",
            full_response=True
        )

        return response.output_path, url_response.image_data  # Return both path and URL

    except Exception as e:
        print(f"Seedream generation failed: {e}")
        return None, None


def test_gemini_edit(image_path):
    """Edit image with Google Gemini"""
    print("\n" + "=" * 50)
    print("GEMINI: Editing image...")
    print("=" * 50)

    if not image_path or not os.path.exists(image_path):
        print("No source image available for Gemini editing")
        return None

    try:
        gemini = ImageGenerator.create(
            provider=ImageProvider.GOOGLE_GEMINI,
            verbose=True
        )

        response = gemini.edit_image(
            image_source=image_path,
            edit_prompt=EDIT_PROMPT,
            size=ImageSize.SQUARE,
            output_format="file",
            output_path="output/gemini_edited.png",
            full_response=True
        )

        print(f"Gemini edit completed in {response.process_time:.2f}s")
        print(f"Saved to: {response.output_path}")
        return response.output_path

    except Exception as e:
        print(f"Gemini edit failed: {e}")
        return None


def test_seedream_edit(image_url):
    """Edit image with Seedream 4.5"""
    print("\n" + "=" * 50)
    print("SEEDREAM: Editing image...")
    print("=" * 50)

    if not image_url:
        print("No source image URL available for Seedream editing")
        return None

    try:
        seedream = ImageGenerator.create(
            provider=ImageProvider.SEEDREAM,
            verbose=True
        )

        response = seedream.edit_image(
            image_source=image_url,
            edit_prompt=EDIT_PROMPT,
            size="2K",
            watermark=False,
            output_format="file",
            output_path="output/seedream_edited.png",
            full_response=True
        )

        print(f"Seedream edit completed in {response.process_time:.2f}s")
        print(f"Saved to: {response.output_path}")
        return response.output_path

    except Exception as e:
        print(f"Seedream edit failed: {e}")
        return None


def main():
    print("=" * 60)
    print("  GEMINI vs SEEDREAM 4.5 - Image Generation Comparison")
    print("=" * 60)

    print(f"\nGeneration Prompt: {GENERATION_PROMPT}")
    print(f"Edit Prompt: {EDIT_PROMPT}")

    # Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    ark_key = os.getenv("ARK_API_KEY")

    print("\n--- API Key Status ---")
    print(f"GEMINI_API_KEY: {'Set' if gemini_key else 'NOT SET'}")
    print(f"ARK_API_KEY: {'Set' if ark_key else 'NOT SET'}")

    results = {
        "gemini_generated": None,
        "seedream_generated": None,
        "gemini_edited": None,
        "seedream_edited": None,
    }

    # Test Generation
    if gemini_key:
        results["gemini_generated"] = test_gemini_generation()
    else:
        print("\nSkipping Gemini generation (GEMINI_API_KEY not set)")

    seedream_path = None
    seedream_url = None
    if ark_key:
        seedream_path, seedream_url = test_seedream_generation()
        results["seedream_generated"] = seedream_path
    else:
        print("\nSkipping Seedream generation (ARK_API_KEY not set)")

    # Test Editing
    # For Gemini edit, use the Gemini generated image (local file)
    if gemini_key and results["gemini_generated"]:
        results["gemini_edited"] = test_gemini_edit(results["gemini_generated"])

    # For Seedream edit, use the Seedream generated image URL
    if ark_key and seedream_url:
        results["seedream_edited"] = test_seedream_edit(seedream_url)

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    for name, path in results.items():
        status = f"OK - {path}" if path else "FAILED or SKIPPED"
        print(f"{name}: {status}")

    print("\n" + "=" * 60)
    print("  Check the 'output/' folder to compare the images!")
    print("=" * 60)


if __name__ == "__main__":
    main()
