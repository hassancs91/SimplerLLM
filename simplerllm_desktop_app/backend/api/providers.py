"""
Providers API Endpoints
"""
from flask import Blueprint, request, jsonify
from services.llm_service import LLMService
from services.settings_service import SettingsService

providers_bp = Blueprint('providers', __name__)

# Provider configurations with available models
PROVIDERS_CONFIG = {
    'openai': {
        'id': 'openai',
        'name': 'OpenAI',
        'env_key': 'OPENAI_API_KEY',
        'models': [
            {'id': 'gpt-4o', 'name': 'GPT-4o', 'context_window': 128000},
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini', 'context_window': 128000},
            {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo', 'context_window': 128000},
            {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo', 'context_window': 16385},
            {'id': 'o1', 'name': 'o1', 'context_window': 200000},
            {'id': 'o1-mini', 'name': 'o1 Mini', 'context_window': 128000},
        ]
    },
    'anthropic': {
        'id': 'anthropic',
        'name': 'Anthropic',
        'env_key': 'ANTHROPIC_API_KEY',
        'models': [
            {'id': 'claude-3-5-sonnet-20241022', 'name': 'Claude 3.5 Sonnet', 'context_window': 200000},
            {'id': 'claude-3-opus-20240229', 'name': 'Claude 3 Opus', 'context_window': 200000},
            {'id': 'claude-3-sonnet-20240229', 'name': 'Claude 3 Sonnet', 'context_window': 200000},
            {'id': 'claude-3-haiku-20240307', 'name': 'Claude 3 Haiku', 'context_window': 200000},
        ]
    },
    'gemini': {
        'id': 'gemini',
        'name': 'Google Gemini',
        'env_key': 'GEMINI_API_KEY',
        'models': [
            {'id': 'gemini-1.5-pro', 'name': 'Gemini 1.5 Pro', 'context_window': 2097152},
            {'id': 'gemini-1.5-flash', 'name': 'Gemini 1.5 Flash', 'context_window': 1048576},
            {'id': 'gemini-2.0-flash-exp', 'name': 'Gemini 2.0 Flash', 'context_window': 1048576},
        ]
    },
    'cohere': {
        'id': 'cohere',
        'name': 'Cohere',
        'env_key': 'COHERE_API_KEY',
        'models': [
            {'id': 'command-r-plus', 'name': 'Command R+', 'context_window': 128000},
            {'id': 'command-r', 'name': 'Command R', 'context_window': 128000},
        ]
    },
    'deepseek': {
        'id': 'deepseek',
        'name': 'DeepSeek',
        'env_key': 'DEEPSEEK_API_KEY',
        'models': [
            {'id': 'deepseek-chat', 'name': 'DeepSeek Chat', 'context_window': 64000},
            {'id': 'deepseek-coder', 'name': 'DeepSeek Coder', 'context_window': 64000},
        ]
    },
    'openrouter': {
        'id': 'openrouter',
        'name': 'OpenRouter',
        'env_key': 'OPENROUTER_API_KEY',
        'models': [
            {'id': 'openai/gpt-4o', 'name': 'GPT-4o (via OpenRouter)', 'context_window': 128000},
            {'id': 'anthropic/claude-3.5-sonnet', 'name': 'Claude 3.5 Sonnet (via OpenRouter)', 'context_window': 200000},
            {'id': 'google/gemini-pro-1.5', 'name': 'Gemini 1.5 Pro (via OpenRouter)', 'context_window': 2097152},
        ]
    },
    'ollama': {
        'id': 'ollama',
        'name': 'Ollama (Local)',
        'env_key': None,
        'models': [
            {'id': 'llama2', 'name': 'Llama 2', 'context_window': 4096},
            {'id': 'llama3', 'name': 'Llama 3', 'context_window': 8192},
            {'id': 'mistral', 'name': 'Mistral', 'context_window': 8192},
            {'id': 'codellama', 'name': 'Code Llama', 'context_window': 16384},
        ]
    },
    'perplexity': {
        'id': 'perplexity',
        'name': 'Perplexity',
        'env_key': 'PERPLEXITY_API_KEY',
        'models': [
            {'id': 'llama-3.1-sonar-large-128k-online', 'name': 'Sonar Large Online', 'context_window': 128000},
            {'id': 'llama-3.1-sonar-small-128k-online', 'name': 'Sonar Small Online', 'context_window': 128000},
        ]
    }
}

settings_service = SettingsService()


@providers_bp.route('/providers', methods=['GET'])
def get_providers():
    """Get list of all available providers with their configuration status."""
    providers = []

    for provider_id, config in PROVIDERS_CONFIG.items():
        # Check if API key is configured
        configured = False
        if config.get('env_key'):
            configured = settings_service.has_api_key(provider_id)
        elif provider_id == 'ollama':
            # Ollama doesn't need an API key
            configured = True

        providers.append({
            'id': config['id'],
            'name': config['name'],
            'configured': configured,
            'requires_key': config.get('env_key') is not None,
            'models': [m['id'] for m in config['models']]
        })

    return jsonify({'providers': providers})


@providers_bp.route('/providers/<provider_id>/models', methods=['GET'])
def get_provider_models(provider_id):
    """Get available models for a specific provider."""
    if provider_id not in PROVIDERS_CONFIG:
        return jsonify({'error': f'Unknown provider: {provider_id}'}), 404

    config = PROVIDERS_CONFIG[provider_id]
    return jsonify({
        'provider': provider_id,
        'models': config['models']
    })


@providers_bp.route('/providers/<provider_id>/validate', methods=['POST'])
def validate_provider(provider_id):
    """Validate an API key for a provider."""
    if provider_id not in PROVIDERS_CONFIG:
        return jsonify({'valid': False, 'error': f'Unknown provider: {provider_id}'}), 404

    data = request.get_json() or {}
    api_key = data.get('api_key', '').strip()

    if not api_key and provider_id != 'ollama':
        return jsonify({'valid': False, 'error': 'API key is required'}), 400

    # Try to validate by making a simple request
    llm_service = LLMService()
    is_valid = llm_service.validate_api_key(provider_id, api_key)

    if is_valid:
        # Save the API key
        settings_service.set_api_key(provider_id, api_key)

    return jsonify({
        'valid': is_valid,
        'provider': provider_id,
        'message': 'API key validated successfully' if is_valid else 'Invalid API key'
    })
