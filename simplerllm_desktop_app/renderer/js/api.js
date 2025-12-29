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

    // Feedback
    /**
     * Start an LLM Feedback loop with SSE streaming.
     *
     * @param {string} prompt - The prompt to improve an answer for
     * @param {Object} config - Configuration object:
     *   - architecture: 'single', 'dual', or 'multi'
     *   - generator: {provider, model} for generator LLM
     *   - critic: {provider, model} for critic LLM (dual mode)
     *   - providers: [{provider, model}, ...] for multi mode
     *   - max_iterations: number (1-10)
     *   - criteria: array of criteria strings
     *   - initial_answer: optional starting answer
     *   - convergence_threshold: number (0-0.5)
     *   - quality_threshold: optional number (1-10)
     * @param {Function} onEvent - Callback for each SSE event
     * @param {Function} onError - Callback for errors
     * @param {Function} onComplete - Callback when complete
     * @returns {Object} Controller with abort() method
     */
    streamFeedback(prompt, config, onEvent, onError, onComplete) {
        const controller = { aborted: false };

        fetch(`${this.baseURL}/feedback/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt,
                architecture: config.architecture,
                generator: config.generator,
                critic: config.critic,
                providers: config.providers,
                max_iterations: config.max_iterations,
                criteria: config.criteria,
                initial_answer: config.initial_answer,
                convergence_threshold: config.convergence_threshold,
                quality_threshold: config.quality_threshold
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
     * Run feedback without streaming (fallback).
     */
    async runFeedback(prompt, config) {
        return this.request('/feedback/run', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                architecture: config.architecture,
                generator: config.generator,
                critic: config.critic,
                providers: config.providers,
                max_iterations: config.max_iterations,
                criteria: config.criteria,
                initial_answer: config.initial_answer,
                convergence_threshold: config.convergence_threshold,
                quality_threshold: config.quality_threshold
            })
        });
    }

    // Retrieval
    /**
     * Get available sample datasets.
     */
    async getRetrievalSamples() {
        return this.request('/retrieval/samples');
    }

    /**
     * Get current index status.
     */
    async getRetrievalStatus() {
        return this.request('/retrieval/status');
    }

    /**
     * Build a retrieval index with SSE streaming.
     *
     * @param {string} source - 'text' or 'sample'
     * @param {string} text - Text to index (if source='text')
     * @param {string} sampleId - Sample dataset ID (if source='sample')
     * @param {string} provider - LLM provider
     * @param {string} model - Model name
     * @param {Function} onEvent - Callback for each SSE event
     * @param {Function} onError - Callback for errors
     * @param {Function} onComplete - Callback when complete
     * @returns {Object} Controller with abort() method
     */
    streamBuildIndex(source, text, sampleId, provider, model, onEvent, onError, onComplete) {
        const controller = { aborted: false };

        fetch(`${this.baseURL}/retrieval/build-index/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                source,
                text,
                sample_id: sampleId,
                provider,
                model
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
     * Build index without streaming (fallback).
     */
    async buildIndex(source, text, sampleId, provider, model) {
        return this.request('/retrieval/build-index', {
            method: 'POST',
            body: JSON.stringify({
                source,
                text,
                sample_id: sampleId,
                provider,
                model
            })
        });
    }

    /**
     * Query the retrieval index with SSE streaming.
     *
     * @param {string} query - Search query
     * @param {number} topK - Number of results to return
     * @param {string} provider - LLM provider
     * @param {string} model - Model name
     * @param {Function} onEvent - Callback for each SSE event
     * @param {Function} onError - Callback for errors
     * @param {Function} onComplete - Callback when complete
     * @returns {Object} Controller with abort() method
     */
    streamRetrieve(query, topK, provider, model, onEvent, onError, onComplete) {
        const controller = { aborted: false };

        fetch(`${this.baseURL}/retrieval/query/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                top_k: topK,
                provider,
                model
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
     * Query without streaming (fallback).
     */
    async retrieve(query, topK, provider, model) {
        return this.request('/retrieval/query', {
            method: 'POST',
            body: JSON.stringify({
                query,
                top_k: topK,
                provider,
                model
            })
        });
    }

    /**
     * Clear the retrieval index.
     */
    async clearRetrievalIndex() {
        return this.request('/retrieval/clear', {
            method: 'POST'
        });
    }

    // Compare
    /**
     * Send a message to two models for comparison with SSE streaming.
     *
     * @param {string} message - The message to send
     * @param {Array} models - Array of {provider, model} for both models (exactly 2)
     * @param {string} conversationId - Optional conversation ID for continuing chat
     * @param {Object} settings - Settings (temperature, max_tokens, system_prompt)
     * @param {Function} onEvent - Callback for each SSE event
     * @param {Function} onError - Callback for errors
     * @param {Function} onComplete - Callback when complete
     * @returns {Object} Controller with abort() method
     */
    streamCompare(message, models, conversationId, settings, onEvent, onError, onComplete) {
        const controller = { aborted: false };

        fetch(`${this.baseURL}/compare/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                models,
                conversation_id: conversationId,
                settings
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
     * Run compare without streaming (fallback).
     */
    async runCompare(message, models, conversationId, settings) {
        return this.request('/compare/run', {
            method: 'POST',
            body: JSON.stringify({
                message,
                models,
                conversation_id: conversationId,
                settings
            })
        });
    }

    /**
     * Clear a compare conversation.
     */
    async clearCompareConversation(conversationId) {
        return this.request(`/compare/${conversationId}`, {
            method: 'DELETE'
        });
    }

    /**
     * Get compare conversation history.
     */
    async getCompareHistory(conversationId) {
        return this.request(`/compare/${conversationId}/history`);
    }

    // Web Search + JSON
    /**
     * Get available web search providers.
     */
    async getWebSearchProviders() {
        return this.request('/websearch-json/providers');
    }

    /**
     * Generate structured JSON from web search.
     *
     * @param {string} prompt - The search query/prompt
     * @param {string} provider - LLM provider ('openai' or 'perplexity')
     * @param {string} model - Model name
     * @param {Object} schema - Schema with 'fields' array (for form mode)
     * @param {Object} settings - Optional settings (temperature, max_tokens)
     * @param {string} schemaMode - 'form' or 'code'
     * @param {string} schemaCode - Pydantic model code (for code mode)
     * @returns {Promise} Response with structured data and sources
     */
    async generateWebSearchJson(prompt, provider, model, schema, settings = {}, schemaMode = 'form', schemaCode = null) {
        return this.request('/websearch-json/generate', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                provider,
                model,
                schema,
                settings,
                schema_mode: schemaMode,
                schema_code: schemaCode
            })
        });
    }

    /**
     * Get Python code example for the current configuration.
     *
     * @param {string} prompt - The search query/prompt
     * @param {string} provider - LLM provider
     * @param {string} model - Model name
     * @param {Object} schema - Schema definition
     * @returns {Promise} Response with code string
     */
    async getWebSearchCodeExample(prompt, provider, model, schema) {
        return this.request('/websearch-json/code-example', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                provider,
                model,
                schema
            })
        });
    }
}

// Global API client instance
const api = new APIClient();
