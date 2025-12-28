/**
 * Main Application
 */

// Tool definitions
const TOOLS = [
    {
        id: 'chat',
        name: 'Chat',
        description: 'Have a conversation with an LLM',
        icon: '💬',
        viewId: 'chat-view'
    }
];

// Default providers data (used when backend is not available)
const DEFAULT_PROVIDERS = [
    {
        id: 'openai',
        name: 'OpenAI',
        configured: false,
        requires_key: true,
        models: [
            { id: 'gpt-4o', name: 'GPT-4o' },
            { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
            { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
            { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' }
        ]
    },
    {
        id: 'anthropic',
        name: 'Anthropic',
        configured: false,
        requires_key: true,
        models: [
            { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet' },
            { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus' },
            { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku' }
        ]
    },
    {
        id: 'gemini',
        name: 'Google Gemini',
        configured: false,
        requires_key: true,
        models: [
            { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
            { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash' },
            { id: 'gemini-2.0-flash-exp', name: 'Gemini 2.0 Flash' }
        ]
    },
    {
        id: 'ollama',
        name: 'Ollama (Local)',
        configured: true,
        requires_key: false,
        models: [
            { id: 'llama2', name: 'Llama 2' },
            { id: 'llama3', name: 'Llama 3' },
            { id: 'mistral', name: 'Mistral' }
        ]
    }
];

// Local storage keys
const STORAGE_KEYS = {
    API_KEYS: 'simplerllm_api_keys',
    SETTINGS: 'simplerllm_settings',
    SAVED_MODELS: 'simplerllm_saved_models'
};

class App {
    constructor() {
        this.providers = [];
        this.selectedProvider = null;
        this.selectedModel = null;
        this.backendReady = false;
        this.apiKeys = {};
        this.savedModels = {};

        // View routing
        this.currentView = 'tools-view';
        this.toolsView = document.getElementById('tools-view');
        this.chatView = document.getElementById('chat-view');

        this.providerSelect = document.getElementById('provider-select');
        this.modelSelect = document.getElementById('model-select');

        this.init();
    }

    async init() {
        // Load saved API keys and models from localStorage
        this.loadApiKeysFromStorage();
        this.loadSavedModelsFromStorage();

        // Initialize managers
        chatManager = new ChatManager();
        settingsManager = new SettingsManager();

        // Setup window controls
        this.setupWindowControls();

        // Render tool cards on the tools page
        this.renderToolCards();

        // Setup back button handler
        document.getElementById('btn-back-to-tools').addEventListener('click', () => {
            this.navigateTo('tools-view');
        });

        // Check backend health (with shorter timeout)
        await this.checkBackendHealth();

        // Load providers (from backend or fallback)
        await this.loadProviders();

        // Load settings
        await settingsManager.loadSettings();

        // Provider select change handler
        this.providerSelect.addEventListener('change', (e) => {
            const providerId = e.target.value;
            if (providerId) {
                this.selectProvider(providerId);
            }
        });

        // Model input change handler - save model per provider
        this.modelSelect.addEventListener('input', (e) => {
            this.selectedModel = e.target.value;
            if (this.selectedProvider && this.selectedModel) {
                this.setSavedModel(this.selectedProvider, this.selectedModel);
            }
        });
    }

    navigateTo(viewId) {
        // Hide all views
        this.toolsView.classList.add('hidden');
        this.chatView.classList.add('hidden');

        // Show target view
        document.getElementById(viewId).classList.remove('hidden');
        this.currentView = viewId;
    }

    renderToolCards() {
        const grid = document.getElementById('tools-grid');
        grid.innerHTML = '';

        TOOLS.forEach(tool => {
            const card = document.createElement('div');
            card.className = 'tool-card';
            card.innerHTML = `
                <div class="tool-card__icon">${tool.icon}</div>
                <h3 class="tool-card__name">${tool.name}</h3>
                <p class="tool-card__description">${tool.description}</p>
            `;
            card.addEventListener('click', () => this.navigateTo(tool.viewId));
            grid.appendChild(card);
        });
    }

    loadApiKeysFromStorage() {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.API_KEYS);
            if (saved) {
                this.apiKeys = JSON.parse(saved);
            }
        } catch (e) {
            console.error('Failed to load API keys from storage:', e);
            this.apiKeys = {};
        }
    }

    saveApiKeysToStorage() {
        try {
            localStorage.setItem(STORAGE_KEYS.API_KEYS, JSON.stringify(this.apiKeys));
        } catch (e) {
            console.error('Failed to save API keys to storage:', e);
        }
    }

    loadSavedModelsFromStorage() {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.SAVED_MODELS);
            if (saved) {
                this.savedModels = JSON.parse(saved);
            }
        } catch (e) {
            console.error('Failed to load saved models from storage:', e);
            this.savedModels = {};
        }
    }

    saveSavedModelsToStorage() {
        try {
            localStorage.setItem(STORAGE_KEYS.SAVED_MODELS, JSON.stringify(this.savedModels));
        } catch (e) {
            console.error('Failed to save models to storage:', e);
        }
    }

    getSavedModel(providerId) {
        return this.savedModels[providerId] || '';
    }

    setSavedModel(providerId, model) {
        this.savedModels[providerId] = model;
        this.saveSavedModelsToStorage();
    }

    setApiKey(provider, apiKey) {
        this.apiKeys[provider] = apiKey;
        this.saveApiKeysToStorage();

        // Update provider configured status
        const providerObj = this.providers.find(p => p.id === provider);
        if (providerObj) {
            providerObj.configured = true;
            this.renderProviderDropdown();
            settingsManager.setProviders(this.providers);
        }
    }

    getApiKey(provider) {
        return this.apiKeys[provider] || null;
    }

    hasApiKey(provider) {
        return !!this.apiKeys[provider];
    }

    setupWindowControls() {
        // Only if electronAPI is available (running in Electron)
        if (window.electronAPI) {
            document.getElementById('btn-minimize').addEventListener('click', () => {
                window.electronAPI.minimizeWindow();
            });

            document.getElementById('btn-maximize').addEventListener('click', () => {
                window.electronAPI.maximizeWindow();
            });

            document.getElementById('btn-close').addEventListener('click', () => {
                window.electronAPI.closeWindow();
            });
        }
    }

    async checkBackendHealth() {
        const maxRetries = 5; // Reduced retries for faster fallback
        let retries = 0;

        while (retries < maxRetries) {
            try {
                const health = await api.healthCheck();
                if (health.status === 'healthy') {
                    this.backendReady = true;
                    console.log('Backend is healthy:', health);
                    return true;
                }
            } catch (error) {
                // Backend not ready yet
            }

            retries++;
            await this.sleep(500);
        }

        console.log('Backend not available, using local mode');
        this.showToast('Running in local mode. Start backend for full functionality.', 'warning');
        return false;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async loadProviders() {
        try {
            if (this.backendReady) {
                const response = await api.getProviders();
                this.providers = response.providers || [];
            } else {
                // Use default providers with local API key status
                this.providers = DEFAULT_PROVIDERS.map(p => ({
                    ...p,
                    configured: p.requires_key ? this.hasApiKey(p.id) : true
                }));
            }

            // Pass providers to settings manager
            settingsManager.setProviders(this.providers);

            // Render provider dropdown
            this.renderProviderDropdown();

            // Auto-select OpenAI
            this.selectProvider('openai');

        } catch (error) {
            console.error('Failed to load providers:', error);
            // Fallback to defaults
            this.providers = DEFAULT_PROVIDERS.map(p => ({
                ...p,
                configured: p.requires_key ? this.hasApiKey(p.id) : true
            }));
            settingsManager.setProviders(this.providers);
            this.renderProviderDropdown();
            this.selectProvider('openai');
        }
    }

    renderProviderDropdown() {
        // Save current selection
        const currentValue = this.providerSelect.value;

        // Clear and rebuild dropdown
        this.providerSelect.innerHTML = '<option value="">Select a provider...</option>';

        this.providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;

            // Show status in the option text
            const status = provider.configured ? '' : ' (Not configured)';
            option.textContent = provider.name + status;

            this.providerSelect.appendChild(option);
        });

        // Restore selection if it still exists
        if (currentValue && this.providers.find(p => p.id === currentValue)) {
            this.providerSelect.value = currentValue;
        }
    }

    selectProvider(providerId) {
        this.selectedProvider = providerId;

        // Update dropdown
        this.providerSelect.value = providerId;

        // Restore saved model for this provider
        const savedModel = this.getSavedModel(providerId);
        this.modelSelect.value = savedModel;
        this.selectedModel = savedModel;
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');

        const toast = document.createElement('div');
        toast.className = `toast toast--${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        // Auto-remove after 4 seconds
        setTimeout(() => {
            toast.style.animation = 'slideInRight 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new App();
});
