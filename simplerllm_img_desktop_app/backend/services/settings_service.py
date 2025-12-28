"""
Settings Service - Manages application settings and API keys
"""
import os
import json
from typing import Any, Dict, Optional
from pathlib import Path


class SettingsService:
    """Service for managing application settings and API keys."""

    def __init__(self):
        # Get the settings directory
        self._settings_dir = self._get_settings_dir()
        self._settings_file = self._settings_dir / 'settings.json'
        self._settings: Dict = self._load_settings()

    def _get_settings_dir(self) -> Path:
        """Get the directory for storing settings."""
        # Use user's app data directory
        if os.name == 'nt':  # Windows
            base_dir = Path(os.environ.get('APPDATA', Path.home()))
        else:  # macOS/Linux
            base_dir = Path.home() / '.config'

        settings_dir = base_dir / 'simplerllm-image-tools'
        settings_dir.mkdir(parents=True, exist_ok=True)
        return settings_dir

    def _load_settings(self) -> Dict:
        """Load settings from file."""
        if self._settings_file.exists():
            try:
                with open(self._settings_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Default settings
        return {
            'api_keys': {},
            'default_provider': 'google',
            'default_model': 'gemini-2.5-flash-image-preview',
            'default_settings': {
                'output_format': 'png'
            }
        }

    def _save_settings(self):
        """Save settings to file."""
        try:
            with open(self._settings_file, 'w') as f:
                json.dump(self._settings, f, indent=2)
        except IOError as e:
            print(f"Error saving settings: {e}")

    def get_settings(self) -> Dict:
        """Get all settings."""
        return self._settings.copy()

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting."""
        return self._settings.get(key, default)

    def set_setting(self, key: str, value: Any):
        """Set a specific setting."""
        self._settings[key] = value
        self._save_settings()

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self._settings.get('api_keys', {}).get(provider)

    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a provider."""
        if 'api_keys' not in self._settings:
            self._settings['api_keys'] = {}
        self._settings['api_keys'][provider] = api_key
        self._save_settings()

    def remove_api_key(self, provider: str):
        """Remove API key for a provider."""
        if 'api_keys' in self._settings and provider in self._settings['api_keys']:
            del self._settings['api_keys'][provider]
            self._save_settings()

    def has_api_key(self, provider: str) -> bool:
        """Check if an API key is set for a provider."""
        return bool(self.get_api_key(provider))

    def get_settings_dir(self) -> Path:
        """Get the settings directory path."""
        return self._settings_dir
