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
    // Image Editing
    // ============================================

    /**
     * Edit an image using AI with text instructions
     * @param {string} sourceImageId - Gallery image ID to edit (use this OR sourcePath)
     * @param {string} sourcePath - Local file path to edit (use this OR sourceImageId)
     * @param {string} prompt - Edit instructions
     * @param {string} model - Model to use
     * @param {string} provider - Provider ID
     * @param {string} aspectRatio - Output aspect ratio
     */
    async editImage(sourceImageId, sourcePath, prompt, model = 'gemini-2.5-flash-image-preview', provider = 'google', aspectRatio = '1:1') {
        return this.post('/edit', {
            source_image_id: sourceImageId,
            source_path: sourcePath,
            prompt,
            model,
            provider,
            aspect_ratio: aspectRatio
        });
    }

    /**
     * Import a local image into the gallery
     * @param {string} filePath - Path to the local image file
     */
    async importImage(filePath) {
        return this.post('/import', { file_path: filePath });
    }

    /**
     * Get the version tree for an image
     * @param {string} imageId - Any image ID in the version chain
     */
    async getVersionTree(imageId) {
        return this.get(`/gallery/${imageId}/versions`);
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

    async getImagePath(imageId) {
        return this.get(`/gallery/${imageId}/path`);
    }

    async deleteImage(imageId) {
        return this.delete(`/gallery/${imageId}`);
    }

    // ============================================
    // Sketch to Image
    // ============================================

    /**
     * Generate an image from a sketch and prompt
     * @param {string} sketchData - Base64 encoded PNG sketch data
     * @param {string} prompt - Text description of desired output
     * @param {string} model - Model to use
     * @param {string} provider - Provider ID
     * @param {string} aspectRatio - Output aspect ratio
     */
    async sketchToImage(sketchData, prompt, model = 'gemini-2.5-flash-image-preview', provider = 'google', aspectRatio = '1:1') {
        return this.post('/sketch-to-image', {
            sketch_data: sketchData,
            prompt,
            model,
            provider,
            aspect_ratio: aspectRatio
        });
    }

    // ============================================
    // Product Enhancement
    // ============================================

    /**
     * Get available enhancement presets
     */
    async getEnhancePresets() {
        return this.get('/enhance/presets');
    }

    /**
     * Enhance a product image
     * @param {string} sourceImageId - Gallery image ID (use this OR sourcePath)
     * @param {string} sourcePath - Local file path (use this OR sourceImageId)
     * @param {string} preset - Preset ID (e.g., 'ecommerce_white')
     * @param {string} customPrompt - Custom enhancement instructions
     * @param {string} model - Model to use
     * @param {string} provider - Provider ID
     * @param {string} aspectRatio - Output aspect ratio
     */
    async enhanceProduct(sourceImageId, sourcePath, preset, customPrompt, model = 'gemini-2.5-flash-image-preview', provider = 'google', aspectRatio = '1:1') {
        return this.post('/enhance', {
            source_image_id: sourceImageId,
            source_path: sourcePath,
            preset,
            custom_prompt: customPrompt,
            model,
            provider,
            aspect_ratio: aspectRatio
        });
    }

    // ============================================
    // YouTube Thumbnail Generator
    // ============================================

    /**
     * Get available thumbnail style presets
     */
    async getThumbnailPresets() {
        return this.get('/thumbnail/presets');
    }

    /**
     * Generate a YouTube thumbnail with face reference
     * @param {string} faceImageId - Gallery image ID for face reference (use this OR facePath)
     * @param {string} facePath - Local file path for face reference (use this OR faceImageId)
     * @param {string} prompt - Full thumbnail description
     * @param {string} preset - Preset ID (e.g., 'reaction', 'tutorial')
     * @param {string} thumbnailText - Text to render on the thumbnail
     * @param {string} model - Model to use
     * @param {string} provider - Provider ID
     */
    async generateThumbnail(faceImageId, facePath, prompt, preset, thumbnailText, model = 'gemini-2.5-flash-image-preview', provider = 'google') {
        return this.post('/thumbnail/generate', {
            face_image_id: faceImageId,
            face_image_path: facePath,
            prompt,
            preset,
            thumbnail_text: thumbnailText,
            model,
            provider
        });
    }

    // ============================================
    // Portrait Studio
    // ============================================

    /**
     * Get available portrait transformation presets
     */
    async getPortraitPresets() {
        return this.get('/portrait/presets');
    }

    /**
     * Generate an image with face reference for character consistency
     * @param {string} faceImageId - Gallery image ID for face reference (use this OR facePath)
     * @param {string} facePath - Local file path for face reference (use this OR faceImageId)
     * @param {string} prompt - Image description (optional if preset provided)
     * @param {string} preset - Preset ID (e.g., 'elderly', 'pixar', 'zombie')
     * @param {string} aspectRatio - Aspect ratio (1:1, 16:9, 9:16)
     * @param {string} model - Model to use
     */
    async generatePortrait(faceImageId, facePath, prompt, preset, aspectRatio = '1:1', model = 'gemini-2.5-flash-image-preview') {
        return this.post('/portrait/generate', {
            face_image_id: faceImageId,
            face_image_path: facePath,
            prompt,
            preset,
            aspect_ratio: aspectRatio,
            model
        });
    }

    // ============================================
    // Character Generator
    // ============================================

    /**
     * Get available character style and pose presets
     */
    async getCharacterPresets() {
        return this.get('/character/presets');
    }

    /**
     * Generate an initial character with style preset
     * @param {string} prompt - Character description
     * @param {string} style - Style preset ID (anime, realistic, cartoon, pixel_art, fantasy)
     * @param {string} aspectRatio - Aspect ratio
     * @param {string} model - Model to use
     */
    async generateCharacter(prompt, style, aspectRatio = '1:1', model = 'gemini-2.5-flash-image-preview') {
        return this.post('/character/generate', {
            prompt,
            style,
            aspect_ratio: aspectRatio,
            model
        });
    }

    /**
     * Generate a pose variation of an existing character
     * @param {string} referenceImageId - Gallery image ID of original character
     * @param {string} pose - Pose preset ID
     * @param {string} customPrompt - Additional instructions
     * @param {string} aspectRatio - Aspect ratio
     * @param {string} model - Model to use
     */
    async generateCharacterVariation(referenceImageId, pose, customPrompt = '', aspectRatio = '1:1', model = 'gemini-2.5-flash-image-preview') {
        return this.post('/character/variation', {
            reference_image_id: referenceImageId,
            pose,
            custom_prompt: customPrompt,
            aspect_ratio: aspectRatio,
            model
        });
    }

    // ============================================
    // Image Fusion
    // ============================================

    /**
     * Generate a fused image from multiple reference images
     * @param {Array} imageSources - Array of { type: 'gallery'|'local', id?: string, path?: string }
     * @param {string} prompt - Description of desired output
     * @param {string} aspectRatio - Aspect ratio (1:1, 16:9, 9:16)
     * @param {string} model - Model to use
     */
    async fuseImages(imageSources, prompt, aspectRatio = '1:1', model = 'gemini-2.5-flash-image-preview') {
        return this.post('/fusion/generate', {
            image_sources: imageSources,
            prompt,
            aspect_ratio: aspectRatio,
            model
        });
    }
}

// Global API instance
const api = new API();
