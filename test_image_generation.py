"""
Test script for SimplerLLM Image Generation feature.
"""

from SimplerLLM import ImageGenerator, ImageProvider, ImageSize

def test_basic_usage():
    """Test basic image generation with default settings."""
    print("\n=== Test 1: Basic Image Generation (URL) ===")

    # Create image generator using factory method
    img_gen = ImageGenerator.create(
        provider=ImageProvider.OPENAI_DALL_E,
        model_name="dall-e-3",
        verbose=True
    )

    # Generate image and get URL
    prompt = "A serene mountain landscape at sunset with a calm lake reflecting the sky"
    url = img_gen.generate_image(
        prompt=prompt,
        size=ImageSize.HORIZONTAL
    )

    print(f"\nGenerated image URL: {url}")
    return url


def test_save_to_file():
    """Test saving image to file."""
    print("\n=== Test 2: Save Image to File ===")

    img_gen = ImageGenerator.create(
        provider=ImageProvider.OPENAI_DALL_E,
        verbose=True
    )

    prompt = "A futuristic city with flying cars and neon lights"
    file_path = img_gen.generate_image(
        prompt=prompt,
        size=ImageSize.SQUARE,
        quality="hd",
        style="vivid",
        output_format="file",
        output_path="output/test_city.png"
    )

    print(f"\nImage saved to: {file_path}")
    return file_path


def test_full_response():
    """Test getting full response with metadata."""
    print("\n=== Test 3: Full Response with Metadata ===")

    img_gen = ImageGenerator.create(
        provider=ImageProvider.OPENAI_DALL_E,
        verbose=True
    )

    prompt = "Abstract geometric art with vibrant colors"
    response = img_gen.generate_image(
        prompt=prompt,
        size=ImageSize.VERTICAL,
        quality="standard",
        style="natural",
        full_response=True
    )

    print(f"\n=== Response Metadata ===")
    print(f"Model: {response.model}")
    print(f"Original Prompt: {response.prompt}")
    print(f"Revised Prompt: {response.revised_prompt}")
    print(f"Size: {response.size}")
    print(f"Quality: {response.quality}")
    print(f"Style: {response.style}")
    print(f"Process Time: {response.process_time:.2f}s")
    print(f"Provider: {response.provider}")
    print(f"Image Data (URL): {response.image_data}")

    return response


def test_different_sizes():
    """Test different size options."""
    print("\n=== Test 4: Different Size Options ===")

    img_gen = ImageGenerator.create(
        provider=ImageProvider.OPENAI_DALL_E,
        verbose=False
    )

    prompt = "A minimalist logo design"

    sizes = [ImageSize.SQUARE, ImageSize.HORIZONTAL, ImageSize.VERTICAL]

    for size in sizes:
        print(f"\nGenerating {size.value} image...")
        url = img_gen.generate_image(
            prompt=prompt,
            size=size
        )
        print(f"{size.value.upper()}: {url}")


def test_direct_instantiation():
    """Test direct instantiation of OpenAIImageGenerator."""
    print("\n=== Test 5: Direct Instantiation ===")

    from SimplerLLM import OpenAIImageGenerator

    img_gen = OpenAIImageGenerator(
        provider=ImageProvider.OPENAI_DALL_E,
        model_name="dall-e-3",
        api_key=None,  # Uses env var
        verbose=True
    )

    prompt = "A cozy coffee shop interior"
    url = img_gen.generate_image(prompt=prompt)

    print(f"\nGenerated image URL: {url}")
    return url


def main():
    """Run all tests."""
    print("=" * 60)
    print("SimplerLLM Image Generation - Test Suite")
    print("=" * 60)

    try:
        # Test 1: Basic usage
        test_basic_usage()

        # Test 2: Save to file
        test_save_to_file()

        # Test 3: Full response
        test_full_response()

        # Test 4: Different sizes
        test_different_sizes()

        # Test 5: Direct instantiation
        test_direct_instantiation()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
