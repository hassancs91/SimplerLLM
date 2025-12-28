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

    // Brainstorm
    /**
     * Start a brainstorming session with SSE streaming.
     *
     * @param {string} prompt - The brainstorming prompt
     * @param {string} provider - LLM provider (e.g., 'openai')
     * @param {string} model - Model name (e.g., 'gpt-4o')
     * @param {Object} params - Brainstorm parameters
     * @param {Function} onEvent - Callback for each SSE event
     * @param {Function} onError - Callback for errors
     * @param {Function} onComplete - Callback when complete
     * @returns {Object} Controller with abort() method
     */
    streamBrainstorm(prompt, provider, model, params, onEvent, onError, onComplete) {
        const controller = { aborted: false };

        // Start the streaming request
        fetch(`${this.baseURL}/brainstorm/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt,
                provider,
                model,
                params
            })
        }).then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            function processStream() {
                reader.read().then(({ done, value }) => {
                    if (done || controller.aborted) {
                        return;
                    }

                    buffer += decoder.decode(value, { stream: true });

                    // Process complete SSE messages
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Keep incomplete line in buffer

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));

                                if (data.type === 'complete') {
                                    onComplete(data);
                                } else if (data.type === 'error') {
                                    onError(new Error(data.error));
                                } else {
                                    onEvent(data);
                                }
                            } catch (e) {
                                console.error('Failed to parse SSE data:', e);
                            }
                        }
                    }

                    processStream();
                }).catch(error => {
                    if (!controller.aborted) {
                        onError(error);
                    }
                });
            }

            processStream();
        }).catch(error => {
            onError(error);
        });

        // Return controller with abort method
        controller.abort = () => {
            controller.aborted = true;
        };

        return controller;
    }

    /**
     * Run brainstorm without streaming (fallback).
     */
    async runBrainstorm(prompt, provider, model, params) {
        return this.request('/brainstorm/run', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                provider,
                model,
                params
            })
        });
    }

    // Judge
    /**
     * Start an LLM Judge evaluation with SSE streaming.
     *
     * @param {string} prompt - The prompt to evaluate
     * @param {Array} contestants - Array of {provider, model} objects
     * @param {Object} judgeConfig - {provider, model} for the judge LLM
     * @param {string} mode - Evaluation mode: 'select_best', 'synthesize', or 'compare'
     * @param {Array} criteria - Evaluation criteria array
     * @param {Function} onEvent - Callback for each SSE event
     * @param {Function} onError - Callback for errors
     * @param {Function} onComplete - Callback when complete
     * @returns {Object} Controller with abort() method
     */
    streamJudge(prompt, contestants, judgeConfig, mode, criteria, onEvent, onError, onComplete) {
        const controller = { aborted: false };

        fetch(`${this.baseURL}/judge/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt,
                contestants,
                judge: judgeConfig,
                mode,
                criteria
            })
        }).then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            function processStream() {
                reader.read().then(({ done, value }) => {
                    if (done || controller.aborted) {
                        return;
                    }

                    buffer += decoder.decode(value, { stream: true });

                    // Process complete SSE messages
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));

                                if (data.type === 'complete') {
                                    onComplete(data);
                                } else if (data.type === 'error') {
                                    onError(new Error(data.error));
                                } else {
                                    onEvent(data);
                                }
                            } catch (e) {
                                console.error('Failed to parse SSE data:', e);
                            }
                        }
                    }

                    processStream();
                }).catch(error => {
                    if (!controller.aborted) {
                        onError(error);
                    }
                });
            }

            processStream();
        }).catch(error => {
            onError(error);
        });

        controller.abort = () => {
            controller.aborted = true;
        };

        return controller;
    }

    /**
     * Run judge without streaming (fallback).
     */
    async runJudge(prompt, contestants, judgeConfig, mode, criteria) {
        return this.request('/judge/run', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                contestants,
                judge: judgeConfig,
                mode,
                criteria
            })
        });
    }
}

// Global API client instance
const api = new APIClient();
