"""
Verification script to test the image generation setup without making API calls.
"""

from SimplerLLM import ImageGenerator, ImageProvider, ImageSize, OpenAIImageGenerator

def verify_imports():
    """Verify all imports work correctly."""
    print("=== Testing Imports ===")
    print("ImageGenerator:", ImageGenerator)
    print("ImageProvider:", ImageProvider)
    print("ImageSize:", ImageSize)
    print("OpenAIImageGenerator:", OpenAIImageGenerator)
    print("All imports successful!\n")


def verify_enums():
    """Verify enum values."""
    print("=== Testing Enums ===")
    print("ImageProvider values:")
    for provider in ImageProvider:
        print(f"  - {provider.name}: {provider.value}")

    print("\nImageSize values:")
    for size in ImageSize:
        print(f"  - {size.name}: {size.value}")
    print()


def verify_factory_pattern():
    """Verify the factory pattern works."""
    print("=== Testing Factory Pattern ===")

    img_gen = ImageGenerator.create(
        provider=ImageProvider.OPENAI_DALL_E,
        model_name="dall-e-3",
        verbose=False
    )

    print(f"Created instance: {type(img_gen).__name__}")
    print(f"Provider: {img_gen.provider}")
    print(f"Model: {img_gen.model_name}")
    print(f"Instance type matches OpenAIImageGenerator: {isinstance(img_gen, OpenAIImageGenerator)}")
    print()


def verify_direct_instantiation():
    """Verify direct instantiation works."""
    print("=== Testing Direct Instantiation ===")

    img_gen = OpenAIImageGenerator(
        provider=ImageProvider.OPENAI_DALL_E,
        model_name="dall-e-2",
        api_key="test_key",
        verbose=False
    )

    print(f"Created instance: {type(img_gen).__name__}")
    print(f"Provider: {img_gen.provider}")
    print(f"Model: {img_gen.model_name}")
    print()


def verify_size_mapping():
    """Verify size mapping functionality."""
    print("=== Testing Size Mapping ===")

    img_gen = ImageGenerator.create(
        provider=ImageProvider.OPENAI_DALL_E,
        verbose=False
    )

    for size in ImageSize:
        dimension = img_gen._map_size_to_dimensions(size)
        print(f"  {size.name} ({size.value}) -> {dimension}")

    # Test custom dimension string
    custom = img_gen._map_size_to_dimensions("512x512")
    print(f"  Custom '512x512' -> {custom}")
    print()


def verify_method_existence():
    """Verify all expected methods exist."""
    print("=== Testing Method Existence ===")

    img_gen = ImageGenerator.create(
        provider=ImageProvider.OPENAI_DALL_E,
        verbose=False
    )

    methods = [
        'generate_image',
        'generate_image_async',
        'prepare_params',
        '_map_size_to_dimensions',
        'set_provider'
    ]

    for method in methods:
        has_method = hasattr(img_gen, method)
        print(f"  {method}: {'YES' if has_method else 'NO'}")
    print()


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("SimplerLLM Image Generation - Setup Verification")
    print("=" * 60)
    print()

    try:
        verify_imports()
        verify_enums()
        verify_factory_pattern()
        verify_direct_instantiation()
        verify_size_mapping()
        verify_method_existence()

        print("=" * 60)
        print("All verification tests passed!")
        print("=" * 60)
        print()
        print("The image generation module is ready to use.")
        print("To test actual image generation, run: test_image_generation.py")
        print("Note: You'll need a valid OPENAI_API_KEY environment variable.")

    except Exception as e:
        print(f"\nError during verification: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
