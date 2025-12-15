from SimplerLLM import ImageGenerator, ImageProvider, ImageSize

# === CONFIGURE YOUR CHARACTER HERE ===
REFERENCE_IMAGE = "cartoon_me.png"  # Path to your character image
OUTPUT_FILE = "confident_smirk.png"

# === PROMPT ===
PROMPT = """Generate the exact same cartoon character from the reference image.
Maintain the same art style, colors, proportions, and character design exactly.
The character should be shown in full body view with a clean, simple background.

Show the character with a confident smirk expression.
One corner of the mouth raised in a knowing smile, eyes slightly narrowed with confidence.
Self-assured, slightly cocky attitude. Full body, front-facing view."""

# === GENERATE ===
img_gen = ImageGenerator.create(provider=ImageProvider.GOOGLE_GEMINI)
result = img_gen.generate_image(
    prompt=PROMPT,
    reference_images=[REFERENCE_IMAGE],
    size=ImageSize.PORTRAIT_3_4,
    output_format="file",
    output_path=OUTPUT_FILE,
    model="gemini-2.5-flash-image-preview",
)
print(f"Generated: {result}")
