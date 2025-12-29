/**
 * Settings Manager - Handles settings panel and API key management
 */
class SettingsManager {
    constructor() {
        this.panel = document.getElementById('settings-panel');
        this.providers = [];
        this.init();
    }

    init() {
        // Close button
        const closeBtn = document.getElementById('btn-close-settings');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.close());
        }

        // Settings button in titlebar
        const settingsBtn = document.getElementById('btn-settings');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.toggle());
        }

        // Click outside to close
        document.addEventListener('click', (e) => {
            if (this.panel && this.panel.classList.contains('open')) {
                if (!this.panel.contains(e.target) && !e.target.closest('#btn-settings')) {
                    this.close();
                }
            }
        });

        // Event delegation for API key buttons
        const container = document.getElementById('api-keys-container');
        if (container) {
            container.addEventListener('click', (e) => {
                const btn = e.target.closest('button[data-action]');
                if (!btn) return;

                const action = btn.dataset.action;
                const providerId = btn.dataset.provider;

                if (action === 'save' && providerId) {
                    this.saveApiKey(providerId);
                } else if (action === 'remove' && providerId) {
                    this.removeApiKey(providerId);
                }
            });
        }

        // Load providers
        this.loadProviders();
    }

    async loadProviders() {
        const result = await api.getProviders();
        if (result.success) {
            this.providers = result.providers;
            this.renderApiKeyInputs();
        }
    }

    renderApiKeyInputs() {
        const container = document.getElementById('api-keys-container');
        if (!container) return;

        container.innerHTML = this.providers.map(provider => `
            <div class="api-key-input">
                <div class="api-key-input__header">
                    <span class="api-key-input__label">${provider.name}</span>
                    <span class="status-dot ${provider.configured ? 'status-dot--configured' : 'status-dot--unconfigured'}"></span>
                </div>
                <div class="input-group" style="margin-bottom: var(--spacing-sm);">
                    <input
                        type="password"
                        class="input"
                        id="api-key-${provider.id}"
                        placeholder="${provider.configured ? '••••••••••••••••' : 'Enter API key...'}"
                    >
                </div>
                <div style="display: flex; gap: var(--spacing-sm);">
                    <button class="btn btn--tertiary btn--sm" data-action="save" data-provider="${provider.id}">
                        Save
                    </button>
                    ${provider.configured ? `
                        <button class="btn btn--ghost btn--sm" data-action="remove" data-provider="${provider.id}">
                            Remove
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }

    async saveApiKey(providerId) {
        const input = document.getElementById(`api-key-${providerId}`);
        if (!input || !input.value.trim()) {
            showToast('Please enter an API key', 'warning');
            return;
        }

        const result = await api.setApiKey(providerId, input.value.trim());
        if (result.success) {
            showToast('API key saved successfully', 'success');
            input.value = '';
            await this.loadProviders();
            // Notify app of configuration change
            if (window.app) {
                window.app.checkBackendStatus();
            }
        } else {
            showToast(result.error || 'Failed to save API key', 'error');
        }
    }

    async removeApiKey(providerId) {
        const result = await api.removeApiKey(providerId);
        if (result.success) {
            showToast('API key removed', 'success');
            await this.loadProviders();
        } else {
            showToast(result.error || 'Failed to remove API key', 'error');
        }
    }

    open() {
        if (this.panel) {
            this.panel.classList.add('open');
        }
    }

    close() {
        if (this.panel) {
            this.panel.classList.remove('open');
        }
    }

    toggle() {
        if (this.panel) {
            this.panel.classList.toggle('open');
        }
    }
}

// Global settings manager instance
let settingsManager;
