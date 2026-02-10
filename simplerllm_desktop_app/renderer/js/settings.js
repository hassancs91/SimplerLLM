/**
 * Settings Management
 */
class SettingsManager {
    constructor() {
        this.settingsPanel = document.getElementById('settings-panel');
        this.btnSettings = document.getElementById('btn-settings');
        this.btnCloseSettings = document.getElementById('btn-close-settings');
        this.apiKeysContainer = document.getElementById('api-keys-container');

        this.providers = [];

        this.init();
    }

    init() {
        // Toggle settings panel
        this.btnSettings.addEventListener('click', () => this.openPanel());
        this.btnCloseSettings.addEventListener('click', () => this.closePanel());

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.settingsPanel.classList.contains('open')) {
                this.closePanel();
            }
        });

        // Slider value updates
        const temperatureSlider = document.getElementById('temperature-slider');
        const temperatureValue = document.getElementById('temperature-value');
        temperatureSlider.addEventListener('input', () => {
            temperatureValue.textContent = temperatureSlider.value;
        });

        const maxTokensSlider = document.getElementById('max-tokens-slider');
        const maxTokensValue = document.getElementById('max-tokens-value');
        maxTokensSlider.addEventListener('input', () => {
            maxTokensValue.textContent = maxTokensSlider.value;
        });
    }

    openPanel() {
        this.settingsPanel.classList.add('open');
        this.renderApiKeyInputs();
    }

    closePanel() {
        this.settingsPanel.classList.remove('open');
    }

    setProviders(providers) {
        this.providers = providers;
    }

    renderApiKeyInputs() {
        this.apiKeysContainer.innerHTML = '';

        // Filter providers that need API keys
        const providersWithKeys = this.providers.filter(p => p.requires_key);

        providersWithKeys.forEach(provider => {
            const hasKey = app.hasApiKey(provider.id);

            const div = document.createElement('div');
            div.className = 'api-key-input';
            div.innerHTML = `
                <div class="api-key-input__header">
                    <label class="input-label">${provider.name}</label>
                    <div class="status-dot ${hasKey ? 'status-dot--configured' : 'status-dot--unconfigured'}"
                         title="${hasKey ? 'Configured' : 'Not configured'}" id="status-${provider.id}"></div>
                </div>
                <div style="display: flex; gap: 8px;">
                    <input type="password"
                           class="input"
                           id="api-key-${provider.id}"
                           placeholder="${hasKey ? '••••••••••••••••' : 'Enter API key...'}"
                           data-provider="${provider.id}">
                    <button class="btn btn--tertiary btn--sm" id="save-btn-${provider.id}">
                        Save
                    </button>
                </div>
            `;
            this.apiKeysContainer.appendChild(div);

            // Add event listener to save button
            const saveBtn = document.getElementById(`save-btn-${provider.id}`);
            saveBtn.addEventListener('click', () => this.saveApiKey(provider.id));

            // Also save on Enter key
            const input = document.getElementById(`api-key-${provider.id}`);
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    this.saveApiKey(provider.id);
                }
            });
        });

        // Add info text
        const info = document.createElement('p');
        info.style.cssText = 'margin-top: 24px; font-size: 12px; color: var(--text-secondary);';
        info.textContent = 'API keys are stored locally in your browser and are never sent to external servers except the respective LLM providers.';
        this.apiKeysContainer.appendChild(info);
    }

    async saveApiKey(providerId) {
        const input = document.getElementById(`api-key-${providerId}`);
        const apiKey = input.value.trim();

        if (!apiKey) {
            app.showToast('Please enter an API key', 'warning');
            return;
        }

        try {
            // Save the API key locally
            app.setApiKey(providerId, apiKey);

            // If backend is available, also try to save there
            if (app.backendReady) {
                try {
                    await api.setApiKey(providerId, apiKey);
                } catch (e) {
                    console.log('Backend save failed, but local save succeeded');
                }
            }

            app.showToast(`${providerId.charAt(0).toUpperCase() + providerId.slice(1)} API key saved!`, 'success');

            // Clear input and update UI
            input.value = '';
            input.placeholder = '••••••••••••••••';

            // Update status dot
            const statusDot = document.getElementById(`status-${providerId}`);
            if (statusDot) {
                statusDot.classList.remove('status-dot--unconfigured');
                statusDot.classList.add('status-dot--configured');
                statusDot.title = 'Configured';
            }

            // Update provider in list
            const provider = this.providers.find(p => p.id === providerId);
            if (provider) {
                provider.configured = true;
            }

            // Refresh provider dropdown
            app.renderProviderDropdown();

        } catch (error) {
            console.error('Save API key error:', error);
            app.showToast(`Error: ${error.message}`, 'error');
        }
    }

    async loadSettings() {
        try {
            if (app.backendReady) {
                const settings = await api.getSettings();

                // Apply default settings
                if (settings.default_settings) {
                    const tempSlider = document.getElementById('temperature-slider');
                    const tempValue = document.getElementById('temperature-value');
                    if (settings.default_settings.temperature !== undefined) {
                        tempSlider.value = settings.default_settings.temperature;
                        tempValue.textContent = settings.default_settings.temperature;
                    }

                    const tokensSlider = document.getElementById('max-tokens-slider');
                    const tokensValue = document.getElementById('max-tokens-value');
                    if (settings.default_settings.max_tokens !== undefined) {
                        tokensSlider.value = settings.default_settings.max_tokens;
                        tokensValue.textContent = settings.default_settings.max_tokens;
                    }

                    const systemPrompt = document.getElementById('system-prompt');
                    if (settings.default_settings.system_prompt) {
                        systemPrompt.value = settings.default_settings.system_prompt;
                    }
                }

                return settings;
            }
        } catch (error) {
            console.error('Load settings error:', error);
        }

        return null;
    }
}

// Global settings manager instance (initialized in app.js)
let settingsManager;
