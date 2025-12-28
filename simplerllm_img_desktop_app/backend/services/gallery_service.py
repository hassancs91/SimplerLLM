"""
Gallery Service - Manages generated images storage and metadata
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image
import io


class GalleryService:
    """Service for managing generated images and their metadata."""

    def __init__(self, settings_service):
        self._settings_service = settings_service
        self._base_dir = self._settings_service.get_settings_dir()
        self._images_dir = self._base_dir / 'images'
        self._gallery_file = self._base_dir / 'gallery.json'

        # Ensure images directory exists
        self._images_dir.mkdir(parents=True, exist_ok=True)

        # Load gallery metadata
        self._gallery: List[Dict] = self._load_gallery()

    def _load_gallery(self) -> List[Dict]:
        """Load gallery metadata from file."""
        if self._gallery_file.exists():
            try:
                with open(self._gallery_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_gallery(self):
        """Save gallery metadata to file."""
        try:
            with open(self._gallery_file, 'w') as f:
                json.dump(self._gallery, f, indent=2)
        except IOError as e:
            print(f"Error saving gallery: {e}")

    def save_image(self, image_bytes: bytes, prompt: str, model: str) -> Dict:
        """
        Save a generated image with metadata.

        Args:
            image_bytes: Raw image bytes
            prompt: The prompt used to generate the image
            model: The model used for generation

        Returns:
            Metadata dict for the saved image
        """
        # Generate unique ID
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        filepath = self._images_dir / filename

        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        # Get image dimensions
        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
        except Exception:
            width, height = 0, 0

        # Create metadata
        metadata = {
            'id': image_id,
            'filename': filename,
            'prompt': prompt,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'size': {
                'width': width,
                'height': height
            }
        }

        # Add to gallery (newest first)
        self._gallery.insert(0, metadata)
        self._save_gallery()

        return metadata

    def get_all_images(self) -> List[Dict]:
        """Get all images metadata, sorted by newest first."""
        return self._gallery.copy()

    def get_image(self, image_id: str) -> Optional[Dict]:
        """Get metadata for a specific image."""
        for item in self._gallery:
            if item['id'] == image_id:
                return item.copy()
        return None

    def get_image_path(self, image_id: str) -> Optional[Path]:
        """Get the file path for an image."""
        metadata = self.get_image(image_id)
        if metadata:
            filepath = self._images_dir / metadata['filename']
            if filepath.exists():
                return filepath
        return None

    def delete_image(self, image_id: str) -> bool:
        """
        Delete an image and its metadata.

        Returns:
            True if deleted, False if not found
        """
        for i, item in enumerate(self._gallery):
            if item['id'] == image_id:
                # Delete the file
                filepath = self._images_dir / item['filename']
                if filepath.exists():
                    try:
                        os.remove(filepath)
                    except OSError as e:
                        print(f"Error deleting file: {e}")

                # Remove from gallery
                self._gallery.pop(i)
                self._save_gallery()
                return True
        return False

    def get_images_dir(self) -> Path:
        """Get the images directory path."""
        return self._images_dir
