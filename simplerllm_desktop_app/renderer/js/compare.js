/**
 * Compare Manager
 * Handles side-by-side model comparison with synchronized conversations
 */
class CompareManager {
    constructor() {
        // State
        this.conversationId = null;
        this.messagesLeft = [];
        this.messagesRight = [];
        this.isLoading = false;
        this.streamController = null;

        // Model configurations
        this.leftModel = { provider: null, model: null };
        this.rightModel = { provider: null, model: null };

        // DOM elements - Left column
        this.providerSelectLeft = document.getElementById('compare-provider-left');
        this.modelInputLeft = document.getElementById('compare-model-left');
        this.messagesContainerLeft = document.getElementById('compare-messages-left');
        this.emptyStateLeft = document.getElementById('compare-empty-left');

        // DOM elements - Right column
        this.providerSelectRight = document.getElementById('compare-provider-right');
        this.modelInputRight = document.getElementById('compare-model-right');
        this.messagesContainerRight = document.getElementById('compare-messages-right');
        this.emptyStateRight = document.getElementById('compare-empty-right');

        // DOM elements - Shared
        this.inputTextarea = document.getElementById('compare-input');
        this.btnSend = document.getElementById('btn-send-compare');
        this.btnClear = document.getElementById('btn-clear-compare');
        this.btnBack = document.getElementById('btn-back-from-compare');
        this.statusBadge = document.getElementById('compare-status-badge');

        // Settings elements
        this.temperatureSlider = document.getElementById('compare-temperature');
        this.temperatureValue = document.getElementById('compare-temperature-value');
        this.maxTokensSlider = document.getElementById('compare-max-tokens');
        this.maxTokensValue = document.getElementById('compare-max-tokens-value');
        this.systemPromptTextarea = document.getElementById('compare-system-prompt');

        this.init();
    }

    init() {
        this._setupActions();
        this._setupInput();
        this._setupSettings();
        this._setupProviderSelects();
    }

    _setupActions() {
        // Send button
        this.btnSend.addEventListener('click', () => this.sendMessage());

        // Clear button
        this.btnClear.addEventListener('click', () => this.clearChat());

        // Back button
        this.btnBack.addEventListener('click', () => {
            app.navigateTo('tools-view');
        });
    }

    _setupInput() {
        // Enter to send, Shift+Enter for new line
        this.inputTextarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.inputTextarea.addEventListener('input', () => {
            this.inputTextarea.style.height = 'auto';
            this.inputTextarea.style.height = Math.min(this.inputTextarea.scrollHeight, 150) + 'px';
        });
    }

    _setupSettings() {
        // Temperature slider
        this.temperatureSlider.addEventListener('input', (e) => {
            this.temperatureValue.textContent = e.target.value;
        });

        // Max tokens slider
        this.maxTokensSlider.addEventListener('input', (e) => {
            this.maxTokensValue.textContent = e.target.value;
        });
    }

    _setupProviderSelects() {
        // Left provider change
        this.providerSelectLeft.addEventListener('change', (e) => {
            this.leftModel.provider = e.target.value;
            this._restoreSavedModel('left', e.target.value);
        });

        // Right provider change
        this.providerSelectRight.addEventListener('change', (e) => {
            this.rightModel.provider = e.target.value;
            this._restoreSavedModel('right', e.target.value);
        });

        // Left model input change
        this.modelInputLeft.addEventListener('input', (e) => {
            this.leftModel.model = e.target.value;
            this._saveModel('left', this.leftModel.provider, e.target.value);
        });

        // Right model input change
        this.modelInputRight.addEventListener('input', (e) => {
            this.rightModel.model = e.target.value;
            this._saveModel('right', this.rightModel.provider, e.target.value);
        });
    }

    _restoreSavedModel(side, providerId) {
        if (!providerId) return;

        // Get saved model from app
        const savedModel = app.getSavedModel(providerId);
        const modelInput = side === 'left' ? this.modelInputLeft : this.modelInputRight;
        const modelObj = side === 'left' ? this.leftModel : this.rightModel;

        if (savedModel) {
            modelInput.value = savedModel;
            modelObj.model = savedModel;
        }
    }

    _saveModel(side, providerId, model) {
        if (providerId && model) {
            app.setSavedModel(providerId, model);
        }
    }

    refresh() {
        this._populateProviderSelects();
    }

    _populateProviderSelects() {
        const providers = app.providers || [];

        // Clear and rebuild both selects
        [this.providerSelectLeft, this.providerSelectRight].forEach(select => {
            select.innerHTML = '<option value="">Provider...</option>';

            providers.forEach(provider => {
                const option = document.createElement('option');
                option.value = provider.id;
                const status = provider.configured ? '' : ' (Not configured)';
                option.textContent = provider.name + status;
                select.appendChild(option);
            });
        });

        // Restore selections if they exist
        if (this.leftModel.provider) {
            this.providerSelectLeft.value = this.leftModel.provider;
        }
        if (this.rightModel.provider) {
            this.providerSelectRight.value = this.rightModel.provider;
        }
    }

    async sendMessage() {
        const message = this.inputTextarea.value.trim();

        if (!message || this.isLoading) return;

        // Validate left model
        if (!this.leftModel.provider || !this.modelInputLeft.value) {
            app.showToast('Please select a provider and model for Model A', 'warning');
            return;
        }

        // Validate right model
        if (!this.rightModel.provider || !this.modelInputRight.value) {
            app.showToast('Please select a provider and model for Model B', 'warning');
            return;
        }

        // Update model values from inputs
        this.leftModel.model = this.modelInputLeft.value;
        this.rightModel.model = this.modelInputRight.value;

        // Check API keys (except for Ollama)
        if (!app.hasApiKey(this.leftModel.provider) && this.leftModel.provider !== 'ollama') {
            app.showToast(`Please configure ${this.leftModel.provider} API key first`, 'warning');
            settingsManager.openPanel();
            return;
        }

        if (!app.hasApiKey(this.rightModel.provider) && this.rightModel.provider !== 'ollama') {
            app.showToast(`Please configure ${this.rightModel.provider} API key first`, 'warning');
            settingsManager.openPanel();
            return;
        }

        // Check if backend is running
        if (!app.backendReady) {
            app.showToast('Backend not running. Start the Flask server first.', 'error');
            return;
        }

        // Hide empty states
        this.emptyStateLeft.classList.add('hidden');
        this.emptyStateRight.classList.add('hidden');

        // Add user message to both columns
        this._addMessage('left', 'user', message);
        this._addMessage('right', 'user', message);

        // Clear input
        this.inputTextarea.value = '';
        this.inputTextarea.style.height = 'auto';

        // Set loading state
        this._setLoading(true);

        // Add loading indicators to both columns
        this._addLoadingIndicator('left');
        this._addLoadingIndicator('right');

        try {
            // Get settings
            const settings = {
                temperature: parseFloat(this.temperatureSlider.value),
                max_tokens: parseInt(this.maxTokensSlider.value),
                system_prompt: this.systemPromptTextarea.value
            };

            // Prepare models array
            const models = [
                { provider: this.leftModel.provider, model: this.leftModel.model },
                { provider: this.rightModel.provider, model: this.rightModel.model }
            ];

            // Stream compare
            this.streamController = api.streamCompare(
                message,
                models,
                this.conversationId,
                settings,
                (event) => this._handleStreamEvent(event),
                (error) => this._handleError(error),
                (data) => this._handleComplete(data)
            );

        } catch (error) {
            console.error('Send message error:', error);
            this._handleError(error);
        }
    }

    _handleStreamEvent(event) {
        switch (event.type) {
            case 'start':
                this.conversationId = event.conversation_id;
                break;

            case 'model_start':
                // Update status for the specific column
                this._updateColumnStatus(event.side, 'Generating...');
                break;

            case 'model_complete':
                // Remove loading indicator and add response
                this._removeLoadingIndicator(event.side);
                this._addMessage(
                    event.side,
                    'assistant',
                    event.response,
                    event.usage,
                    event.execution_time
                );
                this._updateColumnStatus(event.side, 'Done');
                break;

            case 'model_error':
                // Remove loading indicator and show error
                this._removeLoadingIndicator(event.side);
                this._addMessage(
                    event.side,
                    'assistant',
                    `Error: ${event.error}`,
                    null,
                    event.execution_time
                );
                this._updateColumnStatus(event.side, 'Error');
                break;
        }
    }

    _handleComplete(data) {
        this._setLoading(false);
        this.conversationId = data.conversation_id;
    }

    _handleError(error) {
        console.error('Compare error:', error);
        app.showToast(`Error: ${error.message}`, 'error');

        // Remove loading indicators
        this._removeLoadingIndicator('left');
        this._removeLoadingIndicator('right');

        this._setLoading(false);
    }

    _addMessage(side, role, content, usage = null, executionTime = null) {
        const container = side === 'left' ? this.messagesContainerLeft : this.messagesContainerRight;
        const messagesArray = side === 'left' ? this.messagesLeft : this.messagesRight;

        const messageEl = document.createElement('div');
        messageEl.className = `compare-message compare-message--${role}`;

        const bubbleEl = document.createElement('div');
        bubbleEl.className = 'compare-message__bubble';
        bubbleEl.textContent = content;

        messageEl.appendChild(bubbleEl);

        // Add metadata for assistant messages
        if (role === 'assistant') {
            const metaEl = document.createElement('div');
            metaEl.className = 'compare-message__meta';

            // Time
            const timeEl = document.createElement('span');
            timeEl.className = 'compare-message__time';
            timeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            metaEl.appendChild(timeEl);

            // Execution time
            if (executionTime) {
                const execEl = document.createElement('span');
                execEl.textContent = `${executionTime}s`;
                metaEl.appendChild(execEl);
            }

            // Token usage
            if (usage && (usage.input_tokens || usage.output_tokens)) {
                const tokensEl = document.createElement('span');
                tokensEl.className = 'compare-message__tokens';
                const inputTokens = usage.input_tokens || 0;
                const outputTokens = usage.output_tokens || 0;
                tokensEl.textContent = `${inputTokens}/${outputTokens} tokens`;
                metaEl.appendChild(tokensEl);
            }

            messageEl.appendChild(metaEl);
        }

        container.appendChild(messageEl);
        this._scrollToBottom(side);

        // Store message
        messagesArray.push({ role, content, timestamp: new Date().toISOString(), usage });
    }

    _addLoadingIndicator(side) {
        const container = side === 'left' ? this.messagesContainerLeft : this.messagesContainerRight;

        const loadingEl = document.createElement('div');
        loadingEl.className = 'compare-message compare-message--assistant compare-message--loading';
        loadingEl.id = `compare-loading-${side}`;
        loadingEl.innerHTML = `
            <div class="compare-message__bubble">
                <div class="compare-loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        container.appendChild(loadingEl);
        this._scrollToBottom(side);
    }

    _removeLoadingIndicator(side) {
        const loadingEl = document.getElementById(`compare-loading-${side}`);
        if (loadingEl) {
            loadingEl.remove();
        }
    }

    _updateColumnStatus(side, status) {
        // Could add per-column status indicators if needed
    }

    _scrollToBottom(side) {
        const container = side === 'left' ? this.messagesContainerLeft : this.messagesContainerRight;
        container.scrollTop = container.scrollHeight;
    }

    _setLoading(loading) {
        this.isLoading = loading;

        if (loading) {
            this.btnSend.disabled = true;
            this.statusBadge.textContent = 'Comparing...';
            this.statusBadge.className = 'badge badge--warning';
        } else {
            this.btnSend.disabled = false;
            this.statusBadge.textContent = 'Ready';
            this.statusBadge.className = 'badge badge--success';
        }
    }

    clearChat() {
        // Clear messages from both columns
        const leftMessages = this.messagesContainerLeft.querySelectorAll('.compare-message');
        const rightMessages = this.messagesContainerRight.querySelectorAll('.compare-message');
        leftMessages.forEach(msg => msg.remove());
        rightMessages.forEach(msg => msg.remove());

        // Show empty states
        this.emptyStateLeft.classList.remove('hidden');
        this.emptyStateRight.classList.remove('hidden');

        // Clear conversation on server
        if (this.conversationId) {
            api.clearCompareConversation(this.conversationId).catch(console.error);
        }

        // Reset state
        this.conversationId = null;
        this.messagesLeft = [];
        this.messagesRight = [];

        app.showToast('Comparison cleared', 'success');
    }
}

// Global compare manager instance (initialized in app.js)
let compareManager;
