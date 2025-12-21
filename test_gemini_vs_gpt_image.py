"""
Test script to compare Gemini 2.0 Flash vs OpenAI GPT-Image-1 image generation.

Requirements:
- Set GEMINI_API_KEY environment variable for Google Gemini
- Set OPENAI_API_KEY environment variable for OpenAI

Usage:
    python test_gemini_vs_gpt_image.py
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from SimplerLLM.image.generation import ImageGenerator, ImageProvider, ImageSize

# Create output directory
os.makedirs("output", exist_ok=True)

# Common prompt for comparison
GENERATION_PROMPT = "A futuristic cityscape at sunset with flying cars and neon lights, photorealistic style"


def test_gemini_generation():
    """Generate image with Google Gemini 2.0 Flash"""
    print("\n" + "=" * 50)
    print("GEMINI 2.0 FLASH: Generating image...")
    print("=" * 50)

    try:
        gemini = ImageGenerator.create(
            provider=ImageProvider.GOOGLE_GEMINI,
            model_name="gemini-3-pro-image-preview",
            verbose=True
        )

        response = gemini.generate_image(
            prompt=GENERATION_PROMPT,
            size=ImageSize.SQUARE,
            output_format="file",
            output_path="output/gemini_2_flash_generated.png",
            full_response=True
        )

        print(f"Gemini generation completed in {response.process_time:.2f}s")
        print(f"Model: {response.model}")
        print(f"Saved to: {response.output_path}")
        if response.revised_prompt:
            print(f"Revised prompt: {response.revised_prompt[:100]}...")
        return response

    except Exception as e:
        print(f"Gemini generation failed: {e}")
        return None


def test_openai_gpt_image_generation():
    """Generate image with OpenAI GPT-Image-1"""
    print("\n" + "=" * 50)
    print("OPENAI GPT-IMAGE-1: Generating image...")
    print("=" * 50)

    try:
        openai_gen = ImageGenerator.create(
            provider=ImageProvider.OPENAI_DALL_E,
            model_name="gpt-image-1.5",
            verbose=True
        )

        response = openai_gen.generate_image(
            prompt=GENERATION_PROMPT,
            size=ImageSize.SQUARE,
            quality="high",  # gpt-image-1 uses: low, medium, high, auto
            output_format="file",
            output_path="output/openai_gpt_image_generated.png",
            full_response=True
        )

        print(f"OpenAI GPT-Image-1 generation completed in {response.process_time:.2f}s")
        print(f"Model: {response.model}")
        print(f"Saved to: {response.output_path}")
        if response.revised_prompt:
            print(f"Revised prompt: {response.revised_prompt[:100]}...")
        return response

    except Exception as e:
        print(f"OpenAI GPT-Image-1 generation failed: {e}")
        return None


def main():
    print("=" * 60)
    print("  GEMINI 2.0 FLASH vs OPENAI GPT-IMAGE-1 Comparison")
    print("=" * 60)

    print(f"\nGeneration Prompt: {GENERATION_PROMPT}")

    # Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print("\n--- API Key Status ---")
    print(f"GEMINI_API_KEY: {'Set' if gemini_key else 'NOT SET'}")
    print(f"OPENAI_API_KEY: {'Set' if openai_key else 'NOT SET'}")

    results = {
        "gemini": None,
        "openai_gpt_image": None,
    }

    # Test Gemini Generation
    if gemini_key:
        results["gemini"] = test_gemini_generation()
    else:
        print("\nSkipping Gemini generation (GEMINI_API_KEY not set)")

    # Test OpenAI GPT-Image-1 Generation
    if openai_key:
        results["openai_gpt_image"] = test_openai_gpt_image_generation()
    else:
        print("\nSkipping OpenAI GPT-Image-1 generation (OPENAI_API_KEY not set)")

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    for name, response in results.items():
        if response:
            print(f"\n{name.upper()}:")
            print(f"  Status: OK")
            print(f"  Model: {response.model}")
            print(f"  Process Time: {response.process_time:.2f}s")
            print(f"  Output: {response.output_path}")
            if response.file_size:
                print(f"  File Size: {response.file_size:,} bytes")
        else:
            print(f"\n{name.upper()}: FAILED or SKIPPED")

    print("\n" + "=" * 60)
    print("  Check the 'output/' folder to compare the images!")
    print("=" * 60)


if __name__ == "__main__":
    main()
