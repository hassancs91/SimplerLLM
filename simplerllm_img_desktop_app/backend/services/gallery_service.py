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

    def save_image(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        image_type: str = 'generated',
        parent_id: str = None,
        source_type: str = None
    ) -> Dict:
        """
        Save a generated/edited/imported image with metadata.

        Args:
            image_bytes: Raw image bytes
            prompt: The prompt used to generate/edit the image
            model: The model used for generation/editing
            image_type: Type of image - 'generated', 'edited', or 'imported'
            parent_id: ID of parent image (for edited images)
            source_type: Source type - 'gallery' or 'local' (for edited images)

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
            },
            'type': image_type,
            'parent_id': parent_id,
            'source_type': source_type
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

    def import_image(self, file_path: str) -> Dict:
        """
        Import a local image into the gallery.

        Args:
            file_path: Path to the local image file

        Returns:
            Metadata dict for the imported image
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Read the image file
        with open(path, 'rb') as f:
            image_bytes = f.read()

        # Convert to PNG if needed and get dimensions
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Convert to PNG bytes
            png_buffer = io.BytesIO()
            img.save(png_buffer, format='PNG')
            image_bytes = png_buffer.getvalue()
            width, height = img.size
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

        # Generate unique ID
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        filepath = self._images_dir / filename

        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        # Create metadata
        metadata = {
            'id': image_id,
            'filename': filename,
            'prompt': f"Imported from: {path.name}",
            'model': None,
            'timestamp': datetime.now().isoformat(),
            'size': {
                'width': width,
                'height': height
            },
            'type': 'imported',
            'parent_id': None,
            'source_type': 'local'
        }

        # Add to gallery (newest first)
        self._gallery.insert(0, metadata)
        self._save_gallery()

        return metadata

    def get_children(self, image_id: str) -> List[Dict]:
        """
        Get all direct children of an image (images that have this as parent).

        Args:
            image_id: The parent image ID

        Returns:
            List of child image metadata
        """
        return [img for img in self._gallery if img.get('parent_id') == image_id]

    def get_root_image(self, image_id: str) -> Optional[Dict]:
        """
        Find the root ancestor of an image (the original in the version chain).

        Args:
            image_id: Any image ID in the version chain

        Returns:
            The root image metadata, or the image itself if it has no parent
        """
        current = self.get_image(image_id)
        if not current:
            return None

        # Traverse up the parent chain
        visited = set()
        while current and current.get('parent_id') and current['id'] not in visited:
            visited.add(current['id'])
            parent = self.get_image(current['parent_id'])
            if parent:
                current = parent
            else:
                break

        return current

    def get_image_family(self, image_id: str) -> List[Dict]:
        """
        Get all images in a version family (root and all descendants).

        Args:
            image_id: Any image ID in the version chain

        Returns:
            List of all related image metadata, sorted by timestamp
        """
        root = self.get_root_image(image_id)
        if not root:
            return []

        # Collect all descendants using BFS
        family = [root]
        queue = [root['id']]
        visited = {root['id']}

        while queue:
            current_id = queue.pop(0)
            children = self.get_children(current_id)
            for child in children:
                if child['id'] not in visited:
                    visited.add(child['id'])
                    family.append(child)
                    queue.append(child['id'])

        # Sort by timestamp
        family.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return family

    def get_version_tree(self, image_id: str) -> Optional[Dict]:
        """
        Build a tree structure for an image's version history.

        Args:
            image_id: Any image ID in the version chain

        Returns:
            Tree structure with nested children:
            {
                'id': '...',
                'prompt': '...',
                'timestamp': '...',
                'type': '...',
                'children': [...]
            }
        """
        root = self.get_root_image(image_id)
        if not root:
            return None

        def build_node(img: Dict) -> Dict:
            children = self.get_children(img['id'])
            # Sort children by timestamp (oldest first for tree display)
            children.sort(key=lambda x: x.get('timestamp', ''))
            return {
                'id': img['id'],
                'prompt': img.get('prompt', ''),
                'timestamp': img.get('timestamp', ''),
                'type': img.get('type', 'generated'),
                'size': img.get('size', {}),
                'children': [build_node(child) for child in children]
            }

        return build_node(root)

    def has_versions(self, image_id: str) -> bool:
        """
        Check if an image has any versions (parent or children).

        Args:
            image_id: The image ID to check

        Returns:
            True if the image has parent or children
        """
        img = self.get_image(image_id)
        if not img:
            return False

        # Has parent?
        if img.get('parent_id'):
            return True

        # Has children?
        return len(self.get_children(image_id)) > 0
