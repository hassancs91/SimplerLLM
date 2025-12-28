/**
 * Chat Functionality
 */
class ChatManager {
    constructor() {
        this.conversationId = null;
        this.messages = [];
        this.isLoading = false;

        // DOM elements
        this.chatContainer = document.getElementById('chat-container');
        this.chatInput = document.getElementById('chat-input');
        this.btnSend = document.getElementById('btn-send');
        this.btnClear = document.getElementById('btn-clear');
        this.emptyState = document.getElementById('empty-state');
        this.statusBadge = document.getElementById('status-badge');

        this.init();
    }

    init() {
        // Event listeners
        this.btnSend.addEventListener('click', () => this.sendMessage());
        this.btnClear.addEventListener('click', () => this.clearChat());

        // Enter to send, Shift+Enter for new line
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.chatInput.addEventListener('input', () => {
            this.chatInput.style.height = 'auto';
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 150) + 'px';
        });
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();

        if (!message || this.isLoading) return;

        // Check if provider and model are selected
        const provider = app.selectedProvider;
        const model = app.selectedModel;

        if (!provider || !model) {
            app.showToast('Please select a provider and model first', 'warning');
            return;
        }

        // Check if API key is configured (except for Ollama)
        if (!app.hasApiKey(provider) && provider !== 'ollama') {
            app.showToast(`Please configure ${provider.charAt(0).toUpperCase() + provider.slice(1)} API key first`, 'warning');
            settingsManager.openPanel();
            return;
        }

        // Check if backend is running
        if (!app.backendReady) {
            app.showToast('Backend not running. Start the Flask server: cd backend && python app.py', 'error');
            return;
        }

        // Hide empty state
        this.emptyState.classList.add('hidden');

        // Add user message to UI
        this.addMessage('user', message);

        // Clear input
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';

        // Set loading state
        this.setLoading(true);

        try {
            // Get settings
            const settings = {
                temperature: parseFloat(document.getElementById('temperature-slider').value),
                max_tokens: parseInt(document.getElementById('max-tokens-slider').value),
                system_prompt: document.getElementById('system-prompt').value
            };

            // Send to API
            const response = await api.sendMessage(
                message,
                provider,
                model,
                this.conversationId,
                settings
            );

            if (response.success) {
                // Store conversation ID
                this.conversationId = response.conversation_id;

                // Add assistant message to UI with token usage
                this.addMessage('assistant', response.response, response.usage);
            } else {
                throw new Error(response.error || 'Failed to get response');
            }
        } catch (error) {
            console.error('Send message error:', error);
            app.showToast(`Error: ${error.message}`, 'error');

            // Add error message
            this.addMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
        } finally {
            this.setLoading(false);
        }
    }

    addMessage(role, content, usage = null) {
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message chat-message--${role}`;

        const contentEl = document.createElement('div');
        contentEl.className = 'chat-message__content';
        contentEl.textContent = content;

        const timeEl = document.createElement('div');
        timeEl.className = 'chat-message__time';
        timeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageEl.appendChild(contentEl);
        messageEl.appendChild(timeEl);

        // Add token usage for assistant messages
        if (role === 'assistant' && usage && (usage.input_tokens || usage.output_tokens)) {
            const tokensEl = document.createElement('div');
            tokensEl.className = 'chat-message__tokens';
            const inputTokens = usage.input_tokens || 0;
            const outputTokens = usage.output_tokens || 0;
            tokensEl.textContent = `${inputTokens} in · ${outputTokens} out tokens`;
            messageEl.appendChild(tokensEl);
        }

        this.chatContainer.appendChild(messageEl);

        // Scroll to bottom
        this.scrollToBottom();

        // Store message
        this.messages.push({ role, content, timestamp: new Date().toISOString(), usage });
    }

    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    setLoading(loading) {
        this.isLoading = loading;

        if (loading) {
            this.btnSend.disabled = true;
            this.statusBadge.textContent = 'Thinking...';
            this.statusBadge.className = 'badge badge--warning';

            // Add loading indicator
            const loadingEl = document.createElement('div');
            loadingEl.className = 'chat-message chat-message--assistant';
            loadingEl.id = 'loading-message';
            loadingEl.innerHTML = `
                <div class="chat-message__content">
                    <span class="animate-pulse">Thinking...</span>
                </div>
            `;
            this.chatContainer.appendChild(loadingEl);
            this.scrollToBottom();
        } else {
            this.btnSend.disabled = false;
            this.statusBadge.textContent = 'Ready';
            this.statusBadge.className = 'badge badge--success';

            // Remove loading indicator
            const loadingEl = document.getElementById('loading-message');
            if (loadingEl) loadingEl.remove();
        }
    }

    clearChat() {
        // Clear messages from UI
        const messages = this.chatContainer.querySelectorAll('.chat-message');
        messages.forEach(msg => msg.remove());

        // Show empty state
        this.emptyState.classList.remove('hidden');

        // Clear conversation on server
        if (this.conversationId) {
            api.clearConversation(this.conversationId).catch(console.error);
        }

        // Reset state
        this.conversationId = null;
        this.messages = [];

        app.showToast('Chat cleared', 'success');
    }
}

// Global chat manager instance (initialized in app.js)
let chatManager;
