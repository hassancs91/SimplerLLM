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
    },
    {
        id: 'brainstorm',
        name: 'Brainstorm',
        description: 'Generate and explore ideas with visual tree',
        icon: '🧠',
        viewId: 'brainstorm-view'
    },
    {
        id: 'judge',
        name: 'LLM Judge',
        description: 'Compare multiple LLMs with an AI judge',
        icon: '⚖️',
        viewId: 'judge-view'
    },
    {
        id: 'feedback',
        name: 'LLM Feedback',
        description: 'Iteratively improve responses with self-critique',
        icon: '🔄',
        viewId: 'feedback-view'
    },
    {
        id: 'retrieval',
        name: 'LLM Retrieval',
        description: 'Explore hierarchical LLM-based document retrieval',
        icon: '🔍',
        viewId: 'retrieval-view'
    },
    {
        id: 'compare',
        name: 'Compare',
        description: 'Chat with two models side by side',
        icon: '⚔️',
        viewId: 'compare-view'
    },
    {
        id: 'websearch-json',
        name: 'Web Search + JSON',
        description: 'Search the web and extract structured JSON data',
        icon: '🌐',
        viewId: 'websearch-json-view'
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
        this.brainstormView = document.getElementById('brainstorm-view');
        this.judgeView = document.getElementById('judge-view');
        this.feedbackView = document.getElementById('feedback-view');
        this.retrievalView = document.getElementById('retrieval-view');
        this.compareView = document.getElementById('compare-view');
        this.websearchJsonView = document.getElementById('websearch-json-view');

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
        brainstormManager = new BrainstormManager();
        judgeManager = new JudgeManager();
        feedbackManager = new FeedbackManager();
        retrievalManager.init();
        compareManager = new CompareManager();
        websearchJsonManager = new WebSearchJsonManager();

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

        // Load branding for status bar
        await this.loadBranding();

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
        this.brainstormView.classList.add('hidden');
        this.judgeView.classList.add('hidden');
        this.feedbackView.classList.add('hidden');
        this.retrievalView.classList.add('hidden');
        this.compareView.classList.add('hidden');
        this.websearchJsonView.classList.add('hidden');

        // Show target view
        document.getElementById(viewId).classList.remove('hidden');
        this.currentView = viewId;

        // Refresh view-specific data
        if (viewId === 'brainstorm-view' && brainstormManager) {
            brainstormManager.refresh();
        }
        if (viewId === 'judge-view' && judgeManager) {
            judgeManager.refresh();
        }
        if (viewId === 'feedback-view' && feedbackManager) {
            feedbackManager.refresh();
        }
        if (viewId === 'retrieval-view' && typeof retrievalManager !== 'undefined') {
            retrievalManager.refresh();
        }
        if (viewId === 'compare-view' && typeof compareManager !== 'undefined') {
            compareManager.refresh();
        }
        if (viewId === 'websearch-json-view' && typeof websearchJsonManager !== 'undefined') {
            websearchJsonManager.refresh();
        }
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

    // ============================================
    // Branding / Status Bar
    // ============================================

    async loadBranding() {
        try {
            const response = await fetch('js/branding.json');
            const branding = await response.json();
            this.populateStatusBar(branding);
        } catch (error) {
            console.error('Failed to load branding:', error);
        }
    }

    populateStatusBar(branding) {
        const appNameEl = document.getElementById('status-bar-app-name');
        const versionEl = document.getElementById('status-bar-version');
        const copyrightEl = document.getElementById('status-bar-copyright');
        const linksEl = document.getElementById('status-bar-links');

        if (appNameEl) appNameEl.textContent = branding.appName || '';
        if (versionEl) versionEl.textContent = `v${branding.version}` || '';
        if (copyrightEl) copyrightEl.textContent = branding.copyright || '';

        // Social links
        if (linksEl && branding.social) {
            linksEl.innerHTML = '';

            // Website link
            if (branding.website) {
                linksEl.appendChild(this.createStatusBarLink(
                    branding.website.url,
                    branding.website.label,
                    this.getWebsiteIcon()
                ));
            }

            // Social links
            Object.entries(branding.social).forEach(([key, value]) => {
                if (value.url) {
                    linksEl.appendChild(this.createStatusBarLink(
                        value.url,
                        value.label,
                        this.getSocialIcon(key)
                    ));
                }
            });
        }
    }

    createStatusBarLink(url, label, iconSvg) {
        const link = document.createElement('a');
        link.href = url;
        link.className = 'status-bar__link';
        link.title = label;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.innerHTML = iconSvg;
        return link;
    }

    getWebsiteIcon() {
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
        </svg>`;
    }

    getSocialIcon(platform) {
        const icons = {
            twitter: `<svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
            </svg>`,
            github: `<svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>`,
            youtube: `<svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
            </svg>`
        };
        return icons[platform] || this.getWebsiteIcon();
    }
}

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new App();
});
