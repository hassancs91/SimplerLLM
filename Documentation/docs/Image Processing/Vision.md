# Vision Capabilities in SimplerLLM

SimplerLLM now supports vision capabilities, allowing you to pass images to vision-enabled language models for analysis, description, and understanding.

## Overview

Vision capabilities enable you to:
- Analyze images from URLs or local files
- Describe image content in detail
- Extract information from images
- Compare multiple images
- Combine vision with structured output (JSON mode)
- Control cost and accuracy with detail levels

## Supported Providers

Vision capabilities are available for:

### OpenAI
- **Models**: GPT-4o, GPT-4 Vision, GPT-4o-mini
- **Special Feature**: `detail` parameter for cost/quality control
- **Image Format**: Supports both URLs and base64 (data URI)

### Anthropic
- **Models**: All Claude 3, 3.5, and 4 family models
  - Claude 3: Opus, Sonnet, Haiku
  - Claude 3.5: Sonnet, Haiku
  - Claude 4: Opus, Opus 4.1, Sonnet, Sonnet 4.5 (latest)
- **Limits**: 5MB per image, 8000×8000 max pixels, up to 100 images/request
- **Token Calculation**: Approximately (width × height) ÷ 750 tokens per image
- **Best Practice**: Images placed before text for optimal performance

> **Note**: Support for more providers (Gemini, etc.) is planned for future releases.

## Quick Start

### Basic Image Analysis

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# Create an LLM instance with a vision-capable model
llm = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o"
)

# Analyze an image from a URL
response = llm.generate_response(
    prompt="What's in this image?",
    images=["https://example.com/image.jpg"],
    max_tokens=200
)

print(response)
```

### Local Image File

```python
# Analyze a local image
response = llm.generate_response(
    prompt="Describe this image in detail.",
    images=["/path/to/your/image.jpg"],
    max_tokens=300
)
```

### Anthropic Example

```python
from SimplerLLM.language.llm import LLM, LLMProvider

# Create an Anthropic LLM instance
llm = LLM.create(
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-sonnet-4-5"  # Latest Claude model
)

# Analyze an image (same interface as OpenAI)
response = llm.generate_response(
    prompt="What's in this image?",
    images=["https://example.com/image.jpg"],
    max_tokens=200
)
# Note: Anthropic doesn't have a 'detail' parameter
```

## Parameters

### `images` Parameter

The `images` parameter accepts a list of image sources. Each source can be:
- **URL string**: Direct link to an image (e.g., `"https://example.com/image.jpg"`)
- **File path string**: Path to a local image file (e.g., `"/path/to/image.png"`)

```python
# Single image
images=["https://example.com/image.jpg"]

# Multiple images
images=[
    "https://example.com/image1.jpg",
    "/path/to/local/image2.png"
]

# Mix of URLs and local files
images=[
    "https://example.com/remote.jpg",
    "./local_image.png",
    "/absolute/path/to/image.jpg"
]
```

### `detail` Parameter (OpenAI-specific)

The `detail` parameter controls the level of detail in image processing, affecting both cost and accuracy:

- **`"auto"`** (default): Let the model decide the appropriate detail level
- **`"low"`**: Faster and cheaper, suitable for simple queries
- **`"high"`**: More detailed analysis, higher cost and token usage

```python
# Low detail - cost-effective for simple questions
response = llm.generate_response(
    prompt="Is this a cat or a dog?",
    images=["image.jpg"],
    detail="low"
)

# High detail - better for complex analysis
response = llm.generate_response(
    prompt="List all objects visible in this image.",
    images=["image.jpg"],
    detail="high"
)
```

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- WebP (`.webp`)
- BMP (`.bmp`)

## Use Cases and Examples

### 1. Image Description

```python
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

response = llm.generate_response(
    prompt="Provide a detailed description of this image.",
    images=["https://example.com/landscape.jpg"],
    max_tokens=300
)
```

### 2. Object Detection

```python
response = llm.generate_response(
    prompt="List all the objects you can identify in this image.",
    images=["photo.jpg"],
    max_tokens=200
)
```

### 3. Image Comparison

```python
response = llm.generate_response(
    prompt="Compare these two images and describe the differences.",
    images=["image1.jpg", "image2.jpg"],
    max_tokens=400
)
```

### 4. OCR and Text Extraction

```python
response = llm.generate_response(
    prompt="Extract all text visible in this image.",
    images=["document_photo.jpg"],
    max_tokens=500
)
```

### 5. Structured Output with Vision

Combine vision with JSON mode for structured data extraction:

```python
response = llm.generate_response(
    prompt="""Analyze this product image and return JSON with:
    {
        "product_name": "name of the product",
        "colors": ["list of colors"],
        "category": "product category",
        "visible_text": "any text on the product"
    }""",
    images=["product.jpg"],
    json_mode=True,
    max_tokens=300
)
```

### 6. Multiple Images Analysis

```python
images = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    "https://example.com/image3.jpg"
]

response = llm.generate_response(
    prompt="Which of these images contains a cat?",
    images=images,
    max_tokens=100
)
```

### 7. Full Response Mode

Get detailed metadata about the request:

```python
response = llm.generate_response(
    prompt="What's the main subject of this image?",
    images=["image.jpg"],
    full_response=True,
    max_tokens=100
)

print(f"Response: {response.generated_text}")
print(f"Input tokens: {response.input_token_count}")
print(f"Output tokens: {response.output_token_count}")
print(f"Model: {response.model}")
print(f"Processing time: {response.process_time}s")
```

### 8. Async Vision Requests

```python
async def analyze_image():
    llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    response = await llm.generate_response_async(
        prompt="Describe this image.",
        images=["image.jpg"],
        max_tokens=200
    )

    return response

# Run the async function
import asyncio
result = asyncio.run(analyze_image())
```

## Cost Optimization

Vision requests consume more tokens than text-only requests. Here are tips for optimizing costs:

### 1. Use the `detail` Parameter

```python
# For simple yes/no questions - use low detail
response = llm.generate_response(
    prompt="Does this image contain a person?",
    images=["image.jpg"],
    detail="low",  # Cheaper
    max_tokens=10
)

# For detailed analysis - use high detail
response = llm.generate_response(
    prompt="Describe every element in this technical diagram.",
    images=["diagram.jpg"],
    detail="high",  # More accurate but expensive
    max_tokens=500
)
```

### 2. Limit Max Tokens

Only request the amount of output you need:

```python
# For brief answers
response = llm.generate_response(
    prompt="What color is the car?",
    images=["car.jpg"],
    max_tokens=10  # Only need a short answer
)
```

### 3. Use URLs When Possible

Using image URLs is generally more efficient than encoding local files to base64:

```python
# Preferred when images are available online
images=["https://example.com/image.jpg"]

# Use only when necessary
images=["/local/path/image.jpg"]  # Gets base64 encoded
```

## Token Usage

Vision requests have different token costs:
- **Low detail**: Approximately 85 tokens per image
- **High detail**: Varies based on image size (170-1,105 tokens or more)
- **Auto detail**: Model decides based on image complexity

Example token usage:

```python
response = llm.generate_response(
    prompt="Describe this image.",
    images=["image.jpg"],
    full_response=True
)

print(f"Total tokens used: {response.input_token_count + response.output_token_count}")
```

## Error Handling

### Common Errors and Solutions

#### 1. Model Not Supporting Vision

```python
# This will fail - GPT-3.5 doesn't support vision
try:
    llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-3.5-turbo")
    response = llm.generate_response(
        prompt="What's in this image?",
        images=["image.jpg"]
    )
except Exception as e:
    print(f"Error: {e}")
    # Use a vision-capable model instead
```

**Solution**: Use a vision-capable model like `gpt-4o` or `gpt-4-vision-preview`.

#### 2. File Not Found

```python
try:
    response = llm.generate_response(
        prompt="Describe this.",
        images=["/nonexistent/path/image.jpg"]
    )
except FileNotFoundError as e:
    print(f"Image file not found: {e}")
```

**Solution**: Verify the file path exists before making the request.

#### 3. Invalid Image Format

```python
try:
    response = llm.generate_response(
        prompt="Analyze this.",
        images=["file.txt"]  # Not an image
    )
except Exception as e:
    print(f"Invalid image: {e}")
```

**Solution**: Ensure files are valid image formats (JPEG, PNG, GIF, WebP, BMP).

#### 4. Invalid Detail Parameter

```python
try:
    response = llm.generate_response(
        prompt="What's this?",
        images=["image.jpg"],
        detail="medium"  # Invalid - only "low", "high", "auto" allowed
    )
except ValueError as e:
    print(f"Invalid detail level: {e}")
```

**Solution**: Use only `"low"`, `"high"`, or `"auto"`.

## Best Practices

### 1. Be Specific in Prompts

```python
# Less effective
prompt = "Tell me about this image."

# More effective
prompt = "Describe the architectural style of the building in this image, including any notable design features."
```

### 2. Choose Appropriate Detail Level

```python
# Simple classification task - use low detail
prompt = "Is this image taken indoors or outdoors?"
detail = "low"

# Detailed analysis - use high detail
prompt = "Identify all the plants in this botanical garden image."
detail = "high"
```

### 3. Limit Images for Complex Analysis

For detailed analysis, fewer images often yield better results:

```python
# Good - focused analysis
images = ["product_front.jpg", "product_back.jpg"]

# May be overwhelming
images = ["img1.jpg", "img2.jpg", ..., "img20.jpg"]  # Too many
```

### 4. Use Full Response Mode for Debugging

```python
response = llm.generate_response(
    prompt="Analyze this image.",
    images=["image.jpg"],
    full_response=True
)

# Check token usage to optimize future requests
if response.input_token_count > 1000:
    print("Consider using lower detail for this type of image")
```

### 5. Handle Errors Gracefully

```python
def analyze_image_safely(image_path, prompt):
    try:
        response = llm.generate_response(
            prompt=prompt,
            images=[image_path],
            max_tokens=200
        )
        return response
    except FileNotFoundError:
        return "Error: Image file not found"
    except Exception as e:
        return f"Error analyzing image: {str(e)}"
```

## Limitations

### Current Limitations

1. **Provider Support**: Currently only OpenAI models support vision
2. **Image Count**: OpenAI typically supports up to 10 images per request
3. **Image Size**: Very large images may be resized by the API
4. **Cost**: Vision requests are more expensive than text-only requests
5. **Messages Format**: When using the `messages` parameter directly, the `images` parameter is ignored (you need to format vision content manually)

### What Vision Models Can't Do

- Generate or edit images (for that, use image generation models)
- Process video (you can extract frames and analyze them as images)
- Guarantee 100% accuracy in OCR or object detection
- Understand content blocked by OpenAI's safety systems

## Advanced Usage

### Custom Message Formatting

For advanced use cases, you can manually format vision content:

```python
from SimplerLLM.tools.image_helpers import prepare_vision_content

# Prepare multi-part content
content = prepare_vision_content(
    text="What's in these images?",
    images=["image1.jpg", "image2.jpg"],
    detail="high"
)

# Use with custom messages
messages = [
    {"role": "system", "content": "You are an expert image analyst."},
    {"role": "user", "content": content}
]

response = llm.generate_response(messages=messages)
```

### Helper Functions

SimplerLLM provides helper functions for working with images:

```python
from SimplerLLM.tools.image_helpers import (
    is_url,
    encode_image_to_base64,
    get_image_mime_type,
    prepare_image_content,  # OpenAI format
    prepare_image_content_anthropic  # Anthropic format
)

# Check if a source is a URL
is_url("https://example.com/image.jpg")  # True
is_url("/path/to/image.jpg")  # False

# Get MIME type
mime = get_image_mime_type("photo.jpg")  # "image/jpeg"

# Encode image to base64
base64_str = encode_image_to_base64("/path/to/image.jpg")

# Prepare image content for OpenAI API
content = prepare_image_content(
    source="image.jpg",
    detail="high"
)

# Prepare image content for Anthropic API
content = prepare_image_content_anthropic(
    source="image.jpg"
)
```

## Migration Guide

### From Text-Only to Vision

If you have existing code using SimplerLLM, adding vision is straightforward:

**Before (text-only):**
```python
response = llm.generate_response(
    prompt="Describe a sunset.",
    max_tokens=200
)
```

**After (with vision):**
```python
response = llm.generate_response(
    prompt="Describe this sunset.",
    images=["sunset.jpg"],  # Just add this parameter
    max_tokens=200
)
```

All existing parameters work the same way - vision is a simple addition!

## Provider-Specific Differences

### OpenAI vs Anthropic

While the user-facing API is the same (`images` parameter), there are some differences:

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| **Detail Control** | Yes (`detail` parameter) | No |
| **File Size Limit** | No strict limit | 5MB per image |
| **Dimension Limit** | No strict limit | 8000×8000 pixels |
| **Max Images/Request** | ~10 images | Up to 100 images |
| **Token Calculation** | Varies by detail level | ~(width × height) ÷ 750 |
| **Image Ordering** | Text first, then images | Images before text (automatic) |
| **Content Structure** | `image_url` with data URI | `image` with separate base64/media_type |

**Example - OpenAI:**
```python
llm = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
response = llm.generate_response(
    prompt="What's this?",
    images=["image.jpg"],
    detail="high"  # OpenAI-specific
)
```

**Example - Anthropic:**
```python
llm = LLM.create(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-5")
response = llm.generate_response(
    prompt="What's this?",
    images=["image.jpg"]
    # No 'detail' parameter for Anthropic
)
```

## Future Enhancements

Planned features for future releases:
- ✅ ~~Support for Anthropic Claude vision models~~ (COMPLETED)
- Support for Google Gemini vision models
- Image preprocessing utilities (resize, crop, enhance)
- Vision support for other providers (Ollama with LLaVA, etc.)
- Batch image processing helpers
- Cost estimation before making requests

## Troubleshooting

### Images Not Being Analyzed

**Issue**: Model responds as if no image was provided.

**Possible causes**:
1. Using a non-vision model (e.g., GPT-3.5-turbo)
2. Image URL is broken or inaccessible
3. Local file path is incorrect

**Solution**:
- Verify you're using a vision-capable model
- Test image URLs in a browser
- Check local file paths exist

### High Token Usage

**Issue**: Vision requests consuming too many tokens.

**Solutions**:
1. Use `detail="low"` for simple questions
2. Reduce `max_tokens` for shorter responses
3. Use smaller or compressed images
4. Analyze fewer images per request

### API Errors

**Issue**: Getting errors from the OpenAI API.

**Common causes**:
- Invalid API key
- Insufficient credits
- Image violates content policy
- Unsupported image format

**Solution**: Check error message and verify API key, credits, and image content.

## Support and Resources

- **GitHub**: [SimplerLLM Repository](https://github.com/hassancs91/SimplerLLM)
- **Examples**: See `examples/vision_example.py` for complete working examples
- **Tests**: See `tests/test_vision_openai.py` for test cases

## Changelog

### Version 0.x.x (Current)
- Added vision support for OpenAI models
- Support for URL and local file images
- Detail level control (low/high/auto)
- Helper utilities for image processing
- Comprehensive documentation and examples
