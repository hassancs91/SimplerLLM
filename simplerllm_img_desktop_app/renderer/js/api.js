/**
 * API Client - Handles communication with the Flask backend
 */
class API {
    constructor() {
        this.baseUrl = 'http://localhost:5124/api';
    }

    /**
     * Make an HTTP request
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
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
            return data;
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            return { success: false, error: error.message };
        }
    }

    /**
     * GET request
     */
    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    /**
     * POST request
     */
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE request
     */
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }

    // ============================================
    // Health Check
    // ============================================

    async checkHealth() {
        return this.get('/health');
    }

    // ============================================
    // Providers
    // ============================================

    async getProviders() {
        return this.get('/providers');
    }

    async getProviderModels(providerId) {
        return this.get(`/providers/${providerId}/models`);
    }

    // ============================================
    // Settings
    // ============================================

    async getSettings() {
        return this.get('/settings');
    }

    async saveSettings(settings) {
        return this.post('/settings', settings);
    }

    async setApiKey(provider, apiKey) {
        return this.post(`/settings/api-key/${provider}`, { api_key: apiKey });
    }

    async removeApiKey(provider) {
        return this.delete(`/settings/api-key/${provider}`);
    }

    // ============================================
    // Image Generation
    // ============================================

    async generateImage(prompt, model = 'gemini-2.5-flash-image-preview', provider = 'google', aspectRatio = '1:1') {
        return this.post('/generate', { prompt, model, provider, aspect_ratio: aspectRatio });
    }

    // ============================================
    // Gallery
    // ============================================

    async getGallery() {
        return this.get('/gallery');
    }

    async getImageMetadata(imageId) {
        return this.get(`/gallery/${imageId}`);
    }

    getImageUrl(imageId) {
        return `${this.baseUrl}/gallery/${imageId}/image`;
    }

    async deleteImage(imageId) {
        return this.delete(`/gallery/${imageId}`);
    }
}

// Global API instance
const api = new API();
