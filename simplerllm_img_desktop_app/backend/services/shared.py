"""
Shared Services - Singleton instances shared across all API routes
"""
from services.settings_service import SettingsService
from services.image_service import ImageService
from services.gallery_service import GalleryService

# Singleton instances shared across all API routes
# This ensures all routes use the same service instances,
# so changes (like saving an API key) are immediately visible everywhere
settings_service = SettingsService()
image_service = ImageService(settings_service)
gallery_service = GalleryService(settings_service)
