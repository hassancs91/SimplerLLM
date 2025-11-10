# SimplerLLM Image Generation Module

A unified interface for generating images from text prompts across multiple AI providers.

## Overview

The Image Generation module provides a consistent, easy-to-use interface for generating images using various AI providers (currently supporting OpenAI DALL-E, with more providers coming soon).

## Features

- **Unified Interface**: Single API that works across different providers
- **Multiple Providers**: Currently supports OpenAI DALL-E 3 & 2 (Stability AI, Google Imagen coming soon)
- **User-Friendly Sizes**: Simple size options (SQUARE, HORIZONTAL, VERTICAL) that map to provider-specific dimensions
- **Flexible Output Formats**: Get URLs, bytes, or save directly to files
- **Quality & Style Options**: Control image quality and style (for supported providers)
- **Async Support**: Built-in async methods for concurrent operations
- **Full Response Mode**: Access detailed metadata about image generation
- **Verbose Logging**: Optional detailed logging for debugging

## Installation

The image generation module is included in SimplerLLM. Make sure you have the required dependencies:

```bash
pip install openai python-dotenv requests pydantic
```

## Quick Start

```python
from SimplerLLM import ImageGenerator, ImageProvider, ImageSize

# Create image generator
img_gen = ImageGenerator.create(
    provider=ImageProvider.OPENAI_DALL_E,
    model_name="dall-e-3"
)

# Generate image
url = img_gen.generate_image(
    prompt="A serene mountain landscape at sunset",
    size=ImageSize.HORIZONTAL
)

print(f"Image URL: {url}")
```

## API Reference

### Factory Method

```python
ImageGenerator.create(
    provider=ImageProvider.OPENAI_DALL_E,
    model_name="dall-e-3",
    api_key=None,  # Uses OPENAI_API_KEY env var if not provided
    verbose=False
)
```

### Generate Image

```python
img_gen.generate_image(
    prompt: str,                      # Required: Text description
    size=ImageSize.SQUARE,            # SQUARE, HORIZONTAL, VERTICAL, or "1024x1024"
    quality: str = "standard",        # "standard" or "hd" (DALL-E 3)
    style: str = "vivid",             # "vivid" or "natural" (DALL-E 3)
    model: str = None,                # Override model
    output_format: str = "url",       # "url", "bytes", or "file"
    output_path: str = None,          # Required if output_format="file"
    full_response: bool = False,      # Get full metadata
    **kwargs                          # Provider-specific parameters
)
```

### Size Options

The `ImageSize` enum provides user-friendly options:

| ImageSize | DALL-E 3 Dimensions | DALL-E 2 Dimensions | Stability AI |
|-----------|---------------------|---------------------|--------------|
| SQUARE    | 1024x1024          | 1024x1024          | 1024x1024   |
| HORIZONTAL| 1792x1024          | 1024x1024          | 1536x1024   |
| VERTICAL  | 1024x1792          | 1024x1024          | 1024x1536   |

You can also pass custom dimensions as a string (e.g., `"512x512"`).

### Output Formats

- **url**: Returns the image URL (default)
- **bytes**: Returns image data as bytes
- **file**: Saves to disk and returns file path (requires `output_path`)

## Usage Examples

### 1. Basic Usage

```python
from SimplerLLM import ImageGenerator, ImageProvider

img_gen = ImageGenerator.create(
    provider=ImageProvider.OPENAI_DALL_E
)

url = img_gen.generate_image(
    prompt="A futuristic city at night"
)
```

### 2. Different Sizes

```python
# Square image (1024x1024)
url = img_gen.generate_image(
    prompt="A logo design",
    size=ImageSize.SQUARE
)

# Horizontal image (1792x1024)
url = img_gen.generate_image(
    prompt="A panoramic landscape",
    size=ImageSize.HORIZONTAL
)

# Vertical image (1024x1792)
url = img_gen.generate_image(
    prompt="A tall building",
    size=ImageSize.VERTICAL
)
```

### 3. Save to File

```python
file_path = img_gen.generate_image(
    prompt="A sunset over mountains",
    size=ImageSize.HORIZONTAL,
    quality="hd",
    output_format="file",
    output_path="output/sunset.png"
)
```

### 4. Get Image as Bytes

```python
image_bytes = img_gen.generate_image(
    prompt="An icon design",
    output_format="bytes"
)

# Process bytes in memory
with open("my_image.png", "wb") as f:
    f.write(image_bytes)
```

### 5. Full Response with Metadata

```python
response = img_gen.generate_image(
    prompt="Abstract art",
    size=ImageSize.SQUARE,
    quality="hd",
    style="vivid",
    full_response=True
)

print(f"Model: {response.model}")
print(f"Original Prompt: {response.prompt}")
print(f"Revised Prompt: {response.revised_prompt}")
print(f"Size: {response.size}")
print(f"Quality: {response.quality}")
print(f"Style: {response.style}")
print(f"Process Time: {response.process_time:.2f}s")
print(f"Provider: {response.provider}")
```

### 6. Async Generation

```python
import asyncio

async def generate_images():
    img_gen = ImageGenerator.create(
        provider=ImageProvider.OPENAI_DALL_E
    )

    # Generate multiple images concurrently
    prompts = [
        "A mountain",
        "A city",
        "A forest"
    ]

    tasks = [
        img_gen.generate_image_async(prompt=p)
        for p in prompts
    ]

    urls = await asyncio.gather(*tasks)
    return urls

urls = asyncio.run(generate_images())
```

### 7. Quality and Style Options (DALL-E 3)

```python
# High quality, vivid style
url = img_gen.generate_image(
    prompt="A photorealistic portrait",
    quality="hd",
    style="vivid"
)

# Standard quality, natural style
url = img_gen.generate_image(
    prompt="A realistic landscape",
    quality="standard",
    style="natural"
)
```

### 8. Verbose Mode

```python
img_gen = ImageGenerator.create(
    provider=ImageProvider.OPENAI_DALL_E,
    verbose=True  # Enable detailed logging
)

url = img_gen.generate_image(
    prompt="A beautiful scene",
    size=ImageSize.HORIZONTAL
)
```

## Architecture

The module follows a three-layer architecture:

```
SimplerLLM/image/generation/
├── base.py                    # Base classes and enums
├── wrappers/                  # Provider-specific wrappers
│   └── openai_wrapper.py     # OpenAI implementation
└── providers/                 # API implementations
    ├── openai_image.py       # OpenAI API calls
    └── image_response_models.py  # Pydantic models
```

### Layers

1. **Base Layer** (`base.py`): Defines `ImageGenerator` base class, `ImageProvider` enum, `ImageSize` enum, and factory pattern
2. **Wrapper Layer** (`wrappers/`): Provider-specific wrapper classes that handle parameters and call providers
3. **Provider Layer** (`providers/`): Pure API implementation with retry logic and error handling

## Response Models

### ImageGenerationResponse

```python
class ImageGenerationResponse(BaseModel):
    image_data: Union[bytes, str]      # Image bytes, URL, or file path
    model: str                          # Model used
    prompt: str                         # Original prompt
    revised_prompt: Optional[str]       # Enhanced prompt (if provided by model)
    size: str                           # Image dimensions
    quality: Optional[str]              # Quality setting
    style: Optional[str]                # Style setting
    process_time: float                 # Generation time in seconds
    provider: Optional[str]             # Provider name
    file_size: Optional[int]            # Image size in bytes
    output_path: Optional[str]          # File path if saved
    llm_provider_response: Optional[Any]  # Raw API response
```

## Configuration

### API Keys

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or pass it directly:

```python
img_gen = ImageGenerator.create(
    provider=ImageProvider.OPENAI_DALL_E,
    api_key="your-api-key-here"
)
```

### Retry Configuration

Configure retry behavior via environment variables:

```bash
export MAX_RETRIES=3
export RETRY_DELAY=2
```

## Provider Support

### Current Providers

- ✅ **OpenAI DALL-E 3**: Full support with quality & style options
- ✅ **OpenAI DALL-E 2**: Full support with multiple images option

### Coming Soon

- 🚧 **Stability AI**: Stable Diffusion models
- 🚧 **Google Imagen**: Google's image generation
- 🚧 **Midjourney**: Via API when available
- 🚧 **Replicate**: Various open-source models

## Adding New Providers

To add a new provider:

1. Add enum value to `ImageProvider` in `base.py`
2. Create provider implementation in `providers/`
3. Create wrapper class in `wrappers/`
4. Add case to factory method in `base.py`
5. Update size mapping in `_map_size_to_dimensions()`

## Error Handling

The module includes automatic retry logic with exponential backoff:

```python
try:
    url = img_gen.generate_image(
        prompt="A beautiful scene"
    )
except Exception as e:
    print(f"Error: {e}")
```

## Best Practices

1. **Use ImageSize enum** for consistent sizing across providers
2. **Enable verbose mode** during development for debugging
3. **Use async methods** for generating multiple images
4. **Save to file** for large images to avoid memory issues
5. **Use full_response** when you need metadata like revised prompts
6. **Set quality appropriately** - "hd" costs more but provides better quality

## Examples

See [examples/image_generation_examples.py](examples/image_generation_examples.py) for comprehensive usage examples.

## Testing

Run verification tests:

```bash
python verify_image_setup.py
```

Run full integration tests (requires API key):

```bash
python test_image_generation.py
```

## License

This module is part of SimplerLLM and follows the same license.

## Support

For issues, feature requests, or questions, please file an issue on the SimplerLLM repository.
