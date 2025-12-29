/**
 * Product Enhancer Manager - Handles product image enhancement UI and logic
 */
class ProductEnhancerManager {
    constructor() {
        this.isEnhancing = false;
        this.sourceImage = null;  // { id: string, type: 'gallery' | 'local', path?: string }
        this.selectedPreset = null;
        this.presets = [];

        // DOM elements
        this.sourcePreview = document.getElementById('enhance-source-preview');
        this.resultPreview = document.getElementById('enhance-result-preview');
        this.promptInput = document.getElementById('enhance-custom-prompt');
        this.modelSelect = document.getElementById('enhance-model-select');
        this.enhanceBtn = document.getElementById('btn-enhance');
        this.statusSection = document.getElementById('enhance-status');
        this.presetsContainer = document.getElementById('enhance-presets');
        this.browseBtn = document.getElementById('btn-browse-enhance-image');
        this.galleryBtn = document.getElementById('btn-select-enhance-gallery');
        this.galleryPickerModal = document.getElementById('enhance-gallery-picker-modal');
        this.galleryPickerGrid = document.getElementById('enhance-gallery-picker-grid');
        this.closePickerBtn = document.getElementById('btn-close-enhance-gallery-picker');
        this.configureApiKeyBtn = document.getElementById('btn-configure-api-key-enhance');

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadModels();
        await this.loadPresets();
    }

    setupEventListeners() {
        // Enhance button
        if (this.enhanceBtn) {
            this.enhanceBtn.addEventListener('click', () => this.enhance());
        }

        // Browse local button
        if (this.browseBtn) {
            this.browseBtn.addEventListener('click', () => this.browseLocalImage());
        }

        // Select from gallery button
        if (this.galleryBtn) {
            this.galleryBtn.addEventListener('click', () => this.openGalleryPicker());
        }

        // Close gallery picker
        if (this.closePickerBtn) {
            this.closePickerBtn.addEventListener('click', () => this.closeGalleryPicker());
        }

        // Click outside to close picker
        if (this.galleryPickerModal) {
            this.galleryPickerModal.addEventListener('click', (e) => {
                if (e.target === this.galleryPickerModal) {
                    this.closeGalleryPicker();
                }
            });
        }

        // Configure API key button
        if (this.configureApiKeyBtn) {
            this.configureApiKeyBtn.addEventListener('click', () => {
                if (window.settingsManager) {
                    settingsManager.open();
                }
            });
        }

        // Preset selection (delegated)
        if (this.presetsContainer) {
            this.presetsContainer.addEventListener('click', (e) => {
                const presetCard = e.target.closest('.preset-card');
                if (presetCard) {
                    this.selectPreset(presetCard.dataset.preset);
                }
            });
        }

        // Update enhance button state when prompt changes
        if (this.promptInput) {
            this.promptInput.addEventListener('input', () => this.updateEnhanceButtonState());
        }
    }

    async loadModels() {
        const result = await api.getProviders();
        if (result.success && result.providers.length > 0) {
            const provider = result.providers[0];
            if (this.modelSelect && provider.models) {
                this.modelSelect.innerHTML = provider.models.map(model => `
                    <option value="${model.id}">${model.name}</option>
                `).join('');
            }
        }
    }

    async loadPresets() {
        const result = await api.getEnhancePresets();
        if (result.success && result.presets) {
            this.presets = result.presets;
            this.renderPresets();
        }
    }

    renderPresets() {
        if (!this.presetsContainer) return;

        const presetIcons = {
            'full_enhancement': '✨',
            'ecommerce_white': '🛒',
            'social_media_lifestyle': '📱',
            'studio_professional': '📸'
        };

        this.presetsContainer.innerHTML = this.presets.map(preset => `
            <div class="preset-card ${this.selectedPreset === preset.id ? 'preset-card--active' : ''}"
                 data-preset="${preset.id}">
                <div class="preset-card__icon">${presetIcons[preset.id] || '🎨'}</div>
                <div class="preset-card__info">
                    <div class="preset-card__name">${preset.name}</div>
                    <div class="preset-card__description">${preset.description}</div>
                </div>
            </div>
        `).join('');
    }

    selectPreset(presetId) {
        this.selectedPreset = this.selectedPreset === presetId ? null : presetId;
        this.renderPresets();
        this.updateEnhanceButtonState();
    }

    async setSourceFromGallery(imageId) {
        this.sourceImage = {
            id: imageId,
            type: 'gallery'
        };
        this.renderSourcePreview();
        this.updateEnhanceButtonState();
    }

    async setSourceFromLocal(filePath) {
        try {
            const result = await api.importImage(filePath);

            if (result.success) {
                this.sourceImage = {
                    id: result.image_id,
                    type: 'local',
                    path: filePath
                };
                this.renderSourcePreview();

                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }

                showToast('Image imported successfully!', 'success');
            } else {
                showToast(result.error || 'Failed to import image', 'error');
            }
        } catch (error) {
            console.error('Import error:', error);
            showToast('Failed to import image', 'error');
        }

        this.updateEnhanceButtonState();
    }

    async browseLocalImage() {
        if (!window.electronAPI?.openFileDialog) {
            showToast('File dialog not available', 'error');
            return;
        }

        try {
            const filePath = await window.electronAPI.openFileDialog({
                title: 'Select Product Image to Enhance',
                filters: [
                    { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp', 'gif'] }
                ]
            });

            if (filePath) {
                await this.setSourceFromLocal(filePath);
            }
        } catch (error) {
            console.error('File dialog error:', error);
            showToast('Failed to open file dialog', 'error');
        }
    }

    async openGalleryPicker() {
        if (!this.galleryPickerModal || !this.galleryPickerGrid) return;

        const result = await api.getGallery();
        if (!result.success) {
            showToast('Failed to load gallery', 'error');
            return;
        }

        this.galleryPickerGrid.innerHTML = result.images.map(img => `
            <div class="gallery-picker__item" data-id="${img.id}">
                <img src="${api.getImageUrl(img.id)}" alt="${this.escapeHtml(img.prompt || 'Image')}">
                ${img.type === 'enhanced' ? '<span class="image-badge image-badge--enhanced">Enhanced</span>' : ''}
                ${img.type === 'edited' ? '<span class="image-badge image-badge--edited">Edited</span>' : ''}
                ${img.type === 'imported' ? '<span class="image-badge image-badge--imported">Imported</span>' : ''}
            </div>
        `).join('');

        this.galleryPickerGrid.querySelectorAll('.gallery-picker__item').forEach(item => {
            item.addEventListener('click', () => {
                this.setSourceFromGallery(item.dataset.id);
                this.closeGalleryPicker();
            });
        });

        this.galleryPickerModal.classList.remove('hidden');
    }

    closeGalleryPicker() {
        if (this.galleryPickerModal) {
            this.galleryPickerModal.classList.add('hidden');
        }
    }

    clearSource() {
        this.sourceImage = null;

        if (this.sourcePreview) {
            this.sourcePreview.innerHTML = `
                <div class="source-preview__empty">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                    <p>Select a product image</p>
                </div>
            `;
        }

        if (this.resultPreview) {
            this.resultPreview.classList.add('hidden');
        }

        this.updateEnhanceButtonState();
    }

    renderSourcePreview() {
        if (!this.sourcePreview || !this.sourceImage) return;

        this.sourcePreview.innerHTML = `
            <div class="source-preview__image-container">
                <img class="source-preview__image" src="${api.getImageUrl(this.sourceImage.id)}" alt="Source image">
                <button class="source-preview__clear" title="Clear selection">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
        `;

        const clearBtn = this.sourcePreview.querySelector('.source-preview__clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearSource());
        }
    }

    updateEnhanceButtonState() {
        if (this.enhanceBtn) {
            const hasSource = !!this.sourceImage;
            const hasPresetOrPrompt = this.selectedPreset || this.promptInput?.value.trim();
            this.enhanceBtn.disabled = !hasSource || !hasPresetOrPrompt;
        }
    }

    async enhance() {
        if (this.isEnhancing || !this.sourceImage) return;

        const customPrompt = this.promptInput?.value.trim();

        if (!this.selectedPreset && !customPrompt) {
            showToast('Please select a preset or enter custom instructions', 'warning');
            return;
        }

        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';

        this.isEnhancing = true;
        this.showEnhancingState();

        try {
            const result = await api.enhanceProduct(
                this.sourceImage.id,
                null,
                this.selectedPreset,
                customPrompt,
                model
            );

            if (result.success) {
                showToast('Product image enhanced successfully!', 'success');
                this.showResult(result);

                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }
            } else {
                showToast(result.error || 'Failed to enhance image', 'error');
            }
        } catch (error) {
            console.error('Enhancement error:', error);
            showToast('An error occurred during enhancement', 'error');
        } finally {
            this.isEnhancing = false;
            this.hideEnhancingState();
        }
    }

    showEnhancingState() {
        if (this.enhanceBtn) {
            this.enhanceBtn.disabled = true;
            this.enhanceBtn.innerHTML = `
                <span class="spinner" style="width: 20px; height: 20px;"></span>
                Enhancing...
            `;
        }

        if (this.statusSection) {
            this.statusSection.classList.remove('hidden');
            this.statusSection.innerHTML = `
                <div class="generation-status__header">
                    <span class="badge badge--warning">Enhancing Product</span>
                </div>
                <div class="progress-bar" style="margin-top: var(--spacing-sm);">
                    <div class="progress-bar__fill" style="width: 50%; animation: pulse 1s infinite;"></div>
                </div>
                <p style="font-size: var(--font-size-sm); color: var(--text-secondary); margin-top: var(--spacing-sm);">
                    AI is enhancing your product image...
                </p>
            `;
        }
    }

    hideEnhancingState() {
        if (this.enhanceBtn) {
            this.enhanceBtn.disabled = !this.sourceImage || (!this.selectedPreset && !this.promptInput?.value.trim());
            this.enhanceBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707"></path>
                </svg>
                Enhance Product
            `;
        }

        if (this.statusSection) {
            this.statusSection.classList.add('hidden');
        }
    }

    showResult(result) {
        if (!this.resultPreview) return;

        this.resultPreview.classList.remove('hidden');
        this.resultPreview.innerHTML = `
            <h3 class="section-title">Enhanced Result</h3>
            <div class="enhance-comparison">
                <div class="enhance-comparison__item">
                    <span class="enhance-comparison__label">Original</span>
                    <img class="enhance-comparison__image" src="${api.getImageUrl(this.sourceImage.id)}" alt="Original">
                </div>
                <div class="enhance-comparison__arrow">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                </div>
                <div class="enhance-comparison__item">
                    <span class="enhance-comparison__label">Enhanced</span>
                    <img class="enhance-comparison__image" src="${api.getImageUrl(result.image_id)}" alt="Enhanced">
                </div>
            </div>
            <div class="enhance-actions">
                <button class="btn btn--secondary btn--sm" id="btn-enhance-again">
                    Enhance Again
                </button>
                <button class="btn btn--primary btn--sm" id="btn-use-enhanced">
                    Use as New Source
                </button>
            </div>
        `;

        // Enhance again button
        const enhanceAgainBtn = this.resultPreview.querySelector('#btn-enhance-again');
        if (enhanceAgainBtn) {
            enhanceAgainBtn.addEventListener('click', () => {
                this.resultPreview.classList.add('hidden');
            });
        }

        // Use enhanced as new source
        const useEnhancedBtn = this.resultPreview.querySelector('#btn-use-enhanced');
        if (useEnhancedBtn) {
            useEnhancedBtn.addEventListener('click', () => {
                this.setSourceFromGallery(result.image_id);
                this.resultPreview.classList.add('hidden');
            });
        }

        this.resultPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Navigate to enhance view with a specific image
    async enhanceFromGallery(imageId) {
        if (window.app) {
            app.showView('enhance');
        }
        await this.setSourceFromGallery(imageId);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global product enhancer manager instance
let productEnhancerManager;
