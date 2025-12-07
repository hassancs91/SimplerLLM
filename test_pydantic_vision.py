"""
Test script for vision + Pydantic integration
Tests the new vision parameter support in generate_pydantic_json_model
"""
from pydantic import BaseModel, Field
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model
import os

# Define a Pydantic model for image analysis
class ImageAnalysis(BaseModel):
    description: str = Field(description="A detailed description of what's in the image")
    main_objects: list[str] = Field(description="List of main objects visible in the image")
    colors: list[str] = Field(description="Dominant colors in the image")
    scene_type: str = Field(description="Type of scene (e.g., indoor, outdoor, portrait, landscape)")

def test_pydantic_vision_openai():
    """Test vision + Pydantic with OpenAI"""
    print("=" * 60)
    print("Testing Vision + Pydantic with OpenAI")
    print("=" * 60)

    # Initialize OpenAI LLM
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-5"
    )

    # Use a local image
    image_path = "screenshot.jpg"

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return

    print(f"\nAnalyzing image: {image_path}")
    print("Using generate_pydantic_json_model with vision support...")

    # Test with vision parameters
    result = generate_pydantic_json_model(
        model_class=ImageAnalysis,
        prompt="Analyze this image and provide structured information about it.",
        llm_instance=llm,
        max_tokens=500,
        temperature=0.7,
        images=[image_path],
        detail="high"
    )

    if isinstance(result, str):
        print(f"\nError: {result}")
        return

    print("\nSUCCESS! Vision + Pydantic integration works!")
    print("\nParsed Image Analysis:")
    print(f"  Description: {result.description}")
    print(f"  Main Objects: {', '.join(result.main_objects)}")
    print(f"  Colors: {', '.join(result.colors)}")
    print(f"  Scene Type: {result.scene_type}")

    print(f"\nPydantic Model Type: {type(result)}")
    print(f"Model validation passed: {isinstance(result, ImageAnalysis)}")

    return result

def test_pydantic_vision_with_full_response():
    """Test vision + Pydantic with full_response=True"""
    print("\n" + "=" * 60)
    print("Testing Vision + Pydantic with full_response=True")
    print("=" * 60)

    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o"
    )

    image_path = "screenshot.jpg"

    print(f"\nAnalyzing image: {image_path}")
    print("Using full_response=True to get token counts...")

    result = generate_pydantic_json_model(
        model_class=ImageAnalysis,
        prompt="Analyze this image.",
        llm_instance=llm,
        max_tokens=500,
        images=[image_path],
        detail="low",  # Using low detail for faster/cheaper processing
        full_response=True
    )

    if isinstance(result, str):
        print(f"\nError: {result}")
        return

    print("\nSUCCESS!")
    print(f"  Input tokens: {result.input_token_count}")
    print(f"  Output tokens: {result.output_token_count}")
    print(f"  Model used: {result.model_name}")
    print(f"\nAnalysis: {result.model_object.description}")

    return result

if __name__ == "__main__":
    try:
        # Test 1: Basic vision + Pydantic
        test_pydantic_vision_openai()

        # Test 2: Vision + Pydantic with full response
        test_pydantic_vision_with_full_response()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
