import base64
import os


def save_image_from_base64(
    base64_str: str, file_path: str = None, image_type: str = "png"
):
    """
    Saves an image from a base64 encoded string to a file.
    If file_path is not provided, the image is saved in the current working directory.

    :param base64_str: The base64 encoded string of the image.
    :param file_path: The path (including file name) where the image will be saved.
                      If None, saves in the current working directory with a default name.
    :param image_type: The image type/format (e.g., 'png', 'jpg'). Default is 'png'.
    """
    # Decode the base64 string
    image_data = base64.b64decode(base64_str)

    # Set the default file path if not provided
    if file_path is None:
        file_path = os.path.join(os.getcwd(), f"default_image.{image_type}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Append the file extension if not present
    if not file_path.lower().endswith(f".{image_type.lower()}"):
        file_path += f".{image_type}"

    # Write the image data to a file
    with open(file_path, "wb") as file:
        file.write(image_data)
