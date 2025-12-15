"""
Test Script: VisualConfig Pydantic Model Generation

This script tests SimplerLLM's generate_pydantic_json_model function
by generating VisualConfig models for Remotion video generation.

Usage:
    python test_visual_config_generation.py
"""

from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import json

from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model


# ==============================================================================
# PYDANTIC MODEL DEFINITIONS
# ==============================================================================

class AnimationType(str, Enum):
    """Available animation types for visual components."""
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SLIDE_IN = "slide_in"
    SLIDE_OUT = "slide_out"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    WIPE = "wipe"
    PARTICLE_BURST = "particle_burst"
    TEXT_REVEAL = "text_reveal"
    TYPEWRITER = "typewriter"
    MORPH = "morph"
    BOUNCE = "bounce"
    ROTATE = "rotate"
    PULSE = "pulse"


class ColorScheme(BaseModel):
    """Color scheme for the visual."""
    primary: str = Field(
        default="#4d96ff",
        description="Primary color hex code"
    )
    secondary: str = Field(
        default="#00ff88",
        description="Secondary color hex code"
    )
    background: str = Field(
        default="#0a0f1a",
        description="Background color hex code"
    )
    accent: str = Field(
        default="#ff6b6b",
        description="Accent color hex code"
    )
    text: str = Field(
        default="#ffffff",
        description="Text color hex code"
    )


class TextElement(BaseModel):
    """Configuration for a text element in the visual."""
    content: str = Field(
        ...,
        description="Text content to display"
    )
    font_size: int = Field(
        default=64,
        description="Font size in pixels"
    )
    font_weight: int = Field(
        default=700,
        description="Font weight (100-900)"
    )
    font_family: str = Field(
        default="Inter",
        description="Font family name"
    )
    position: Literal["top", "center", "bottom"] = Field(
        default="center",
        description="Vertical position"
    )
    x_offset: int = Field(
        default=0,
        description="Horizontal offset from center in pixels"
    )
    y_offset: int = Field(
        default=0,
        description="Vertical offset from position in pixels"
    )
    color: Optional[str] = Field(
        default=None,
        description="Text color (uses theme text color if not set)"
    )
    animation: AnimationType = Field(
        default=AnimationType.FADE_IN,
        description="Animation type for this text"
    )
    start_frame: int = Field(
        default=0,
        description="Frame when animation starts"
    )
    duration_frames: int = Field(
        default=30,
        description="Animation duration in frames"
    )


class ComponentConfig(BaseModel):
    """Configuration for a visual component."""
    type: str = Field(
        ...,
        description="Component type: text, shape, particle, gradient, image, chaos_lines, module_grid"
    )
    animation: AnimationType = Field(
        default=AnimationType.FADE_IN,
        description="Animation type"
    )
    start_frame: int = Field(
        default=0,
        description="Frame when animation starts"
    )
    duration_frames: int = Field(
        default=30,
        description="Animation duration in frames"
    )
    props: dict = Field(
        default_factory=dict,
        description="Component-specific properties as a JSON object (e.g., {\"color\": \"#ff0000\", \"size\": 100}). Must be an object, NOT a string."
    )


class VisualConfig(BaseModel):
    """
    Main configuration for a generated visual.

    This model defines everything needed to render an animated visual:
    - Metadata (name, description)
    - Timing (duration, fps)
    - Dimensions (width, height)
    - Styling (colors)
    - Content (text elements, components)
    """
    name: str = Field(
        ...,
        description="Name of the visual"
    )
    description: str = Field(
        ...,
        description="Description of what this visual shows"
    )
    template: Optional[str] = Field(
        default=None,
        description="Template name if using template mode"
    )

    # Timing
    duration_seconds: float = Field(
        default=5.0,
        description="Total duration in seconds"
    )
    fps: int = Field(
        default=30,
        description="Frames per second"
    )

    # Dimensions
    width: int = Field(
        default=1080,
        description="Video width in pixels"
    )
    height: int = Field(
        default=1920,
        description="Video height in pixels"
    )

    # Styling
    colors: ColorScheme = Field(
        default_factory=ColorScheme,
        description="Color scheme for the visual"
    )

    # Content
    text_elements: List[TextElement] = Field(
        default_factory=list,
        description="Text elements to display"
    )
    components: List[ComponentConfig] = Field(
        default_factory=list,
        description="Visual components"
    )

    # Generation metadata
    generation_mode: Literal["template", "ai"] = Field(
        default="template",
        description="How this config was generated"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Original user prompt"
    )

    @property
    def total_frames(self) -> int:
        """Calculate total frames based on duration and fps."""
        return int(self.duration_seconds * self.fps)

    def to_remotion_props(self) -> dict:
        """Convert to props format for Remotion."""
        return {
            "name": self.name,
            "durationInFrames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "colors": self.colors.model_dump(),
            "textElements": [t.model_dump() for t in self.text_elements],
            "components": [c.model_dump() for c in self.components],
        }


# ==============================================================================
# MAIN TEST FUNCTION
# ==============================================================================

def main():
    """Test the VisualConfig generation with SimplerLLM."""

    print("=" * 60)
    print("VisualConfig Pydantic Model Generation Test")
    print("=" * 60)

    # Create LLM instance
    llm = LLM.create(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o"
    )

    # Test prompt
    test_prompt = "Dramatic countdown timer with glowing numbers"

    print(f"\nPrompt: {test_prompt}")
    print("-" * 60)

    # System prompt for Remotion visual generation context
    system_prompt = """You are a creative visual designer for video generation.
Generate a VisualConfig that creates an engaging animated visual for Remotion.
Be creative with colors, animations, and text elements.
The visual should be eye-catching and suitable for social media.

IMPORTANT: For the 'animation' field, you MUST use ONLY one of these exact values:
fade_in, fade_out, slide_in, slide_out, scale_in, scale_out, wipe, particle_burst, text_reveal, typewriter, morph, bounce, rotate, pulse

For 'position' field in text_elements, use ONLY: top, center, bottom
For 'generation_mode' field, use ONLY: template, ai"""

    # Generate the VisualConfig
    result = generate_pydantic_json_model(
        model_class=VisualConfig,
        prompt=test_prompt,
        llm_instance=llm,
        max_retries=3,
        temperature=0.8,
        system_prompt=system_prompt,
        full_response=True
    )

    # Handle result
    if hasattr(result, 'model_object'):
        config = result.model_object

        print("\nGeneration Successful!")
        print("=" * 60)

        # Print metadata
        print(f"\nInput tokens: {result.input_token_count}")
        print(f"Output tokens: {result.output_token_count}")
        print(f"Process time: {result.process_time:.2f}s")

        # Print generated config
        print("\n" + "-" * 60)
        print("GENERATED VISUAL CONFIG:")
        print("-" * 60)
        print(f"Name: {config.name}")
        print(f"Description: {config.description}")
        print(f"Duration: {config.duration_seconds}s @ {config.fps}fps ({config.total_frames} frames)")
        print(f"Dimensions: {config.width}x{config.height}")
        print(f"Generation Mode: {config.generation_mode}")

        print(f"\nColors:")
        print(f"  Primary: {config.colors.primary}")
        print(f"  Secondary: {config.colors.secondary}")
        print(f"  Background: {config.colors.background}")
        print(f"  Accent: {config.colors.accent}")
        print(f"  Text: {config.colors.text}")

        print(f"\nText Elements ({len(config.text_elements)}):")
        for i, text in enumerate(config.text_elements, 1):
            print(f"  {i}. \"{text.content}\" - {text.animation.value} @ frame {text.start_frame}")

        print(f"\nComponents ({len(config.components)}):")
        for i, comp in enumerate(config.components, 1):
            print(f"  {i}. {comp.type} - {comp.animation.value} @ frame {comp.start_frame}")

        # Print Remotion props
        print("\n" + "-" * 60)
        print("REMOTION PROPS (JSON):")
        print("-" * 60)
        remotion_props = config.to_remotion_props()
        print(json.dumps(remotion_props, indent=2))

        # Also print full model dump
        print("\n" + "-" * 60)
        print("FULL MODEL DUMP:")
        print("-" * 60)
        print(json.dumps(config.model_dump(), indent=2, default=str))

    else:
        print(f"\nError: {result}")


if __name__ == "__main__":
    main()
