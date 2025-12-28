/**
 * API Client for Flask Backend
 */
class APIClient {
    constructor() {
        this.baseURL = 'http://localhost:5123/api';
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;

        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP error ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }

    // Health check
    async healthCheck() {
        return this.request('/health');
    }

    // Chat
    async sendMessage(message, provider, model, conversationId, settings) {
        return this.request('/chat', {
            method: 'POST',
            body: JSON.stringify({
                message,
                provider,
                model,
                conversation_id: conversationId,
                settings
            })
        });
    }

    async clearConversation(conversationId) {
        return this.request(`/chat/${conversationId}`, {
            method: 'DELETE'
        });
    }

    async getHistory(conversationId) {
        return this.request(`/chat/${conversationId}/history`);
    }

    // Providers
    async getProviders() {
        return this.request('/providers');
    }

    async getProviderModels(providerId) {
        return this.request(`/providers/${providerId}/models`);
    }

    async validateProvider(providerId, apiKey) {
        return this.request(`/providers/${providerId}/validate`, {
            method: 'POST',
            body: JSON.stringify({ api_key: apiKey })
        });
    }

    // Settings
    async getSettings() {
        return this.request('/settings');
    }

    async saveSettings(settings) {
        return this.request('/settings', {
            method: 'POST',
            body: JSON.stringify(settings)
        });
    }

    async setApiKey(provider, apiKey) {
        return this.request(`/settings/api-key/${provider}`, {
            method: 'POST',
            body: JSON.stringify({ api_key: apiKey })
        });
    }

    async removeApiKey(provider) {
        return this.request(`/settings/api-key/${provider}`, {
            method: 'DELETE'
        });
    }
}

// Global API client instance
const api = new APIClient();
