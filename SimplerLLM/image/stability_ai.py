from pydantic import BaseModel
import base64
import requests
from typing import List
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")


class ImageData(BaseModel):
    base64_str: str
    size_kb: float
    width: int
    height: int


class ImageList(BaseModel):
    images: List[ImageData]


def generate_images(
    model_name: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    samples: int = 1,
    steps: int = 30,
    cfg_scale: int = 7,
    seed: int = 0,
    style_preset: str = None,
) -> ImageList:

    engine_id = model_name
    api_host = "https://api.stability.ai"
    api_key = STABILITY_API_KEY

    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "text_prompts": [{"text": prompt}],
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
            "seed": seed,
            "style_preset": style_preset,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    images_data = []
    for image in data["artifacts"]:
        base64_str = image["base64"]
        decoded_image = base64.b64decode(base64_str)
        size_kb = len(decoded_image) / 1024

        images_data.append(
            ImageData(
                base64_str=base64_str, size_kb=size_kb, width=width, height=height
            )
        )

    return ImageList(images=images_data)
