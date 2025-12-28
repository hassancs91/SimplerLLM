/**
 * Image Generation Manager - Handles image generation UI and logic
 */
class ImageGenerationManager {
    constructor() {
        this.isGenerating = false;
        this.selectedAspectRatio = '1:1';
        this.promptInput = document.getElementById('prompt-input');
        this.modelSelect = document.getElementById('model-select');
        this.generateBtn = document.getElementById('btn-generate');
        this.statusSection = document.getElementById('generation-status');
        this.previewSection = document.getElementById('preview-section');
        this.aspectRatioSelector = document.getElementById('aspect-ratio-selector');
        this.init();
    }

    init() {
        // Generate button
        if (this.generateBtn) {
            this.generateBtn.addEventListener('click', () => this.generate());
        }

        // Enter key in prompt (with Shift+Enter for new line)
        if (this.promptInput) {
            this.promptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.generate();
                }
            });
        }

        // Aspect ratio selector
        if (this.aspectRatioSelector) {
            this.aspectRatioSelector.addEventListener('click', (e) => {
                const btn = e.target.closest('.aspect-btn');
                if (btn) {
                    this.selectAspectRatio(btn);
                }
            });
        }

        // Load models
        this.loadModels();
    }

    selectAspectRatio(btn) {
        // Remove active class from all buttons
        this.aspectRatioSelector.querySelectorAll('.aspect-btn').forEach(b => b.classList.remove('active'));
        // Add active class to clicked button
        btn.classList.add('active');
        // Store selected ratio
        this.selectedAspectRatio = btn.dataset.ratio;
    }

    async loadModels() {
        const result = await api.getProviders();
        if (result.success && result.providers.length > 0) {
            const provider = result.providers[0]; // Currently only Google
            if (this.modelSelect && provider.models) {
                this.modelSelect.innerHTML = provider.models.map(model => `
                    <option value="${model.id}">${model.name}</option>
                `).join('');
            }
        }
    }

    async generate() {
        if (this.isGenerating) return;

        const prompt = this.promptInput?.value.trim();
        if (!prompt) {
            showToast('Please enter a prompt', 'warning');
            this.promptInput?.focus();
            return;
        }

        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';
        const aspectRatio = this.selectedAspectRatio || '1:1';

        // Start generating
        this.isGenerating = true;
        this.showGeneratingState();

        try {
            const result = await api.generateImage(prompt, model, 'google', aspectRatio);

            if (result.success) {
                showToast('Image generated successfully!', 'success');
                this.showPreview(result);

                // Add to gallery instantly
                if (window.galleryManager) {
                    galleryManager.addImageToTop({
                        id: result.image_id,
                        prompt: prompt,
                        model: model,
                        timestamp: result.metadata?.timestamp || new Date().toISOString()
                    });
                }

                // Clear prompt
                if (this.promptInput) {
                    this.promptInput.value = '';
                }
            } else {
                showToast(result.error || 'Failed to generate image', 'error');
            }
        } catch (error) {
            console.error('Generation error:', error);
            showToast('An error occurred during generation', 'error');
        } finally {
            this.isGenerating = false;
            this.hideGeneratingState();
        }
    }

    showGeneratingState() {
        // Disable button
        if (this.generateBtn) {
            this.generateBtn.disabled = true;
            this.generateBtn.innerHTML = `
                <span class="spinner" style="width: 20px; height: 20px;"></span>
                Generating...
            `;
        }

        // Show status
        if (this.statusSection) {
            this.statusSection.classList.remove('hidden');
            this.statusSection.innerHTML = `
                <div class="generation-status__header">
                    <span class="badge badge--warning">Generating</span>
                </div>
                <div class="progress-bar" style="margin-top: var(--spacing-sm);">
                    <div class="progress-bar__fill" style="width: 50%; animation: pulse 1s infinite;"></div>
                </div>
                <p style="font-size: var(--font-size-sm); color: var(--text-secondary); margin-top: var(--spacing-sm);">
                    This may take a few seconds...
                </p>
            `;
        }
    }

    hideGeneratingState() {
        // Enable button
        if (this.generateBtn) {
            this.generateBtn.disabled = false;
            this.generateBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 5v14M5 12h14"></path>
                </svg>
                Generate
            `;
        }

        // Hide status
        if (this.statusSection) {
            this.statusSection.classList.add('hidden');
        }
    }

    showPreview(result) {
        if (!this.previewSection) {
            console.error('Preview section not found');
            return;
        }

        console.log('Showing preview for image:', result.image_id);
        console.log('Image URL:', api.getImageUrl(result.image_id));

        this.previewSection.classList.remove('hidden');
        this.previewSection.innerHTML = `
            <h3 class="section-title">Latest Generation</h3>
            <div class="preview-image-container">
                <img class="preview-image" src="${api.getImageUrl(result.image_id)}" alt="Generated image"
                     onerror="console.error('Failed to load image:', this.src)">
                <div class="preview-prompt">${this.escapeHtml(result.metadata?.prompt || 'No prompt')}</div>
            </div>
        `;

        // Scroll the preview section into view
        this.previewSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global image generation manager instance
let imageGenerationManager;
