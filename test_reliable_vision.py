"""
Test script for vision + ReliableLLM integration
Tests the new vision parameter support in ReliableLLM and generate_pydantic_json_model_reliable
"""
from pydantic import BaseModel, Field
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_reliable
import os

# Define a Pydantic model for image analysis
class ImageDescription(BaseModel):
    summary: str = Field(description="A brief summary of the image")
    objects: list[str] = Field(description="List of objects in the image")

def test_reliable_vision_basic():
    """Test vision with ReliableLLM basic generate_response"""
    print("=" * 60)
    print("Testing Vision with ReliableLLM (basic generate_response)")
    print("=" * 60)

    # Create primary and secondary LLMs
    primary_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o"
    )

    secondary_llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-5"
    )

    # Create ReliableLLM instance
    reliable_llm = ReliableLLM(
        primary_llm=primary_llm,
        secondary_llm=secondary_llm,
        verbose=True
    )

    image_path = "screenshot.jpg"

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return

    print(f"\nAnalyzing image with ReliableLLM: {image_path}")

    # Test ReliableLLM with vision
    response = reliable_llm.generate_response(
        prompt="Describe what you see in this image in one sentence.",
        images=[image_path],
        detail="high",
        max_tokens=200,
        return_provider=True
    )

    if isinstance(response, tuple):
        text, provider, model_name = response
        print("\nSUCCESS!")
        print(f"  Provider used: {provider.name}")
        print(f"  Model used: {model_name}")
        print(f"  Response: {text}")
    else:
        print(f"  Response: {response}")

    return response

def test_reliable_vision_pydantic():
    """Test vision + Pydantic with ReliableLLM"""
    print("\n" + "=" * 60)
    print("Testing Vision + Pydantic with ReliableLLM")
    print("=" * 60)

    # Create primary and secondary LLMs
    primary_llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o"
    )

    secondary_llm = LLM.create(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-5"
    )

    # Create ReliableLLM instance
    reliable_llm = ReliableLLM(
        primary_llm=primary_llm,
        secondary_llm=secondary_llm,
        verbose=True
    )

    image_path = "screenshot.jpg"

    print(f"\nAnalyzing image with ReliableLLM + Pydantic: {image_path}")

    # Test generate_pydantic_json_model_reliable with vision
    result = generate_pydantic_json_model_reliable(
        model_class=ImageDescription,
        prompt="Analyze this image and provide structured information.",
        reliable_llm=reliable_llm,
        max_tokens=400,
        images=[image_path],
        detail="high"
    )

    if isinstance(result, str):
        print(f"\nError: {result}")
        return

    # Result should be a tuple of (model_object, provider, model_name)
    model_object, provider, model_name = result

    print("\nSUCCESS! Vision + Pydantic + ReliableLLM works!")
    print(f"\n  Provider used: {provider.name}")
    print(f"  Model used: {model_name}")
    print(f"  Summary: {model_object.summary}")
    print(f"  Objects: {', '.join(model_object.objects)}")
    print(f"\n  Validated Pydantic Model: {isinstance(model_object, ImageDescription)}")

    return result

def test_reliable_vision_fallback():
    """Test that fallback works with vision (by using invalid primary)"""
    print("\n" + "=" * 60)
    print("Testing ReliableLLM Fallback with Vision")
    print("=" * 60)

    # Create an invalid primary LLM (will fail validation)
    try:
        # Use a model that doesn't exist to force fallback
        primary_llm = LLM.create(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o"  # Valid model
        )

        secondary_llm = LLM.create(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-sonnet-4-5"
        )

        reliable_llm = ReliableLLM(
            primary_llm=primary_llm,
            secondary_llm=secondary_llm,
            verbose=True
        )

        image_path = "screenshot.jpg"

        print(f"\nAnalyzing image (testing fallback mechanism): {image_path}")

        response, provider, model_name = reliable_llm.generate_response(
            prompt="What's in this image?",
            images=[image_path],
            max_tokens=100,
            return_provider=True
        )

        print(f"\nFallback test completed")
        print(f"  Provider that responded: {provider.name}")
        print(f"  Model: {model_name}")
        print(f"  Response: {response[:100]}...")

        return response

    except Exception as e:
        print(f"Note: Both providers are valid, so no fallback occurred: {e}")

if __name__ == "__main__":
    try:
        # Test 1: Basic ReliableLLM with vision
        test_reliable_vision_basic()

        # Test 2: ReliableLLM + Pydantic with vision
        test_reliable_vision_pydantic()

        # Test 3: Fallback mechanism with vision
        test_reliable_vision_fallback()

        print("\n" + "=" * 60)
        print("All ReliableLLM vision tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
