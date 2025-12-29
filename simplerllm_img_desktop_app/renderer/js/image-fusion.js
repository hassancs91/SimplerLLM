/**
 * Image Fusion Manager - Handles multi-image fusion UI and logic
 */
class ImageFusionManager {
    constructor() {
        this.isGenerating = false;
        this.selectedImages = [];  // Array of { id: string, type: 'gallery' | 'local' }
        this.maxImages = 5;
        this.selectedAspectRatio = '1:1';

        // DOM elements
        this.selectedGrid = document.getElementById('fusion-selected-grid');
        this.imagesCount = document.getElementById('fusion-images-count');
        this.resultPreview = document.getElementById('fusion-result-preview');
        this.promptInput = document.getElementById('fusion-prompt-input');
        this.modelSelect = document.getElementById('fusion-model-select');
        this.generateBtn = document.getElementById('btn-generate-fusion');
        this.statusSection = document.getElementById('fusion-status');
        this.aspectRatioSelector = document.getElementById('fusion-aspect-ratio-selector');
        this.browseBtn = document.getElementById('btn-browse-fusion-image');
        this.galleryBtn = document.getElementById('btn-select-fusion-gallery');
        this.galleryPickerModal = document.getElementById('fusion-gallery-picker-modal');
        this.galleryPickerGrid = document.getElementById('fusion-gallery-picker-grid');
        this.closePickerBtn = document.getElementById('btn-close-fusion-gallery-picker');
        this.configureApiKeyBtn = document.getElementById('btn-configure-api-key-fusion');

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadModels();
        this.renderSelectedImages();
    }

    setupEventListeners() {
        // Generate button
        if (this.generateBtn) {
            this.generateBtn.addEventListener('click', () => this.generate());
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

        // Aspect ratio selection
        if (this.aspectRatioSelector) {
            this.aspectRatioSelector.querySelectorAll('.aspect-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.selectAspectRatio(btn.dataset.ratio);
                });
            });
        }

        // Update generate button state when prompt changes
        if (this.promptInput) {
            this.promptInput.addEventListener('input', () => this.updateGenerateButtonState());
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

    selectAspectRatio(ratio) {
        this.selectedAspectRatio = ratio;

        // Update UI
        if (this.aspectRatioSelector) {
            this.aspectRatioSelector.querySelectorAll('.aspect-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.ratio === ratio);
            });
        }
    }

    async addImageFromGallery(imageId) {
        if (this.selectedImages.length >= this.maxImages) {
            showToast(`Maximum ${this.maxImages} images allowed`, 'warning');
            return;
        }

        // Check if already added
        if (this.selectedImages.some(img => img.id === imageId)) {
            showToast('Image already selected', 'warning');
            return;
        }

        this.selectedImages.push({
            id: imageId,
            type: 'gallery'
        });

        this.renderSelectedImages();
        this.updateGenerateButtonState();
    }

    async addImageFromLocal(filePath) {
        if (this.selectedImages.length >= this.maxImages) {
            showToast(`Maximum ${this.maxImages} images allowed`, 'warning');
            return;
        }

        try {
            const result = await api.importImage(filePath);

            if (result.success) {
                this.selectedImages.push({
                    id: result.image_id,
                    type: 'local'
                });

                this.renderSelectedImages();

                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }

                showToast('Image added successfully!', 'success');
            } else {
                showToast(result.error || 'Failed to import image', 'error');
            }
        } catch (error) {
            console.error('Import error:', error);
            showToast('Failed to import image', 'error');
        }

        this.updateGenerateButtonState();
    }

    removeImage(index) {
        this.selectedImages.splice(index, 1);
        this.renderSelectedImages();
        this.updateGenerateButtonState();
    }

    clearAllImages() {
        this.selectedImages = [];
        this.renderSelectedImages();
        this.updateGenerateButtonState();

        if (this.resultPreview) {
            this.resultPreview.classList.add('hidden');
        }
    }

    renderSelectedImages() {
        if (!this.selectedGrid) return;

        // Update count
        if (this.imagesCount) {
            this.imagesCount.textContent = `${this.selectedImages.length}/${this.maxImages}`;
        }

        // Render slots
        let html = '';
        for (let i = 0; i < this.maxImages; i++) {
            const image = this.selectedImages[i];
            if (image) {
                html += `
                    <div class="fusion-image-slot fusion-image-slot--filled" data-index="${i}">
                        <img class="fusion-image-slot__image" src="${api.getImageUrl(image.id)}" alt="Reference ${i + 1}">
                        <button class="fusion-image-slot__remove" title="Remove image">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                        <span class="fusion-image-slot__number">${i + 1}</span>
                    </div>
                `;
            } else {
                html += `
                    <div class="fusion-image-slot fusion-image-slot--empty">
                        <span>${i + 1}</span>
                    </div>
                `;
            }
        }
        this.selectedGrid.innerHTML = html;

        // Add remove button handlers
        this.selectedGrid.querySelectorAll('.fusion-image-slot__remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const slot = btn.closest('.fusion-image-slot');
                const index = parseInt(slot.dataset.index);
                this.removeImage(index);
            });
        });
    }

    async browseLocalImage() {
        if (!window.electronAPI?.openFileDialog) {
            showToast('File dialog not available', 'error');
            return;
        }

        if (this.selectedImages.length >= this.maxImages) {
            showToast(`Maximum ${this.maxImages} images allowed`, 'warning');
            return;
        }

        try {
            const filePath = await window.electronAPI.openFileDialog({
                title: 'Select Image for Fusion',
                filters: [
                    { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp'] }
                ]
            });

            if (filePath) {
                await this.addImageFromLocal(filePath);
            }
        } catch (error) {
            console.error('File dialog error:', error);
            showToast('Failed to open file dialog', 'error');
        }
    }

    async openGalleryPicker() {
        if (!this.galleryPickerModal || !this.galleryPickerGrid) return;

        if (this.selectedImages.length >= this.maxImages) {
            showToast(`Maximum ${this.maxImages} images already selected`, 'warning');
            return;
        }

        const result = await api.getGallery();
        if (!result.success) {
            showToast('Failed to load gallery', 'error');
            return;
        }

        // Filter out already selected images
        const selectedIds = new Set(this.selectedImages.map(img => img.id));

        this.galleryPickerGrid.innerHTML = result.images.map(img => `
            <div class="gallery-picker__item ${selectedIds.has(img.id) ? 'gallery-picker__item--selected' : ''}"
                 data-id="${img.id}"
                 ${selectedIds.has(img.id) ? 'style="opacity: 0.5; pointer-events: none;"' : ''}>
                <img src="${api.getImageUrl(img.id)}" alt="${this.escapeHtml(img.prompt || 'Image')}">
                ${img.type === 'fusion' ? '<span class="image-badge image-badge--fusion">Fusion</span>' : ''}
                ${img.type === 'imported' ? '<span class="image-badge image-badge--imported">Imported</span>' : ''}
            </div>
        `).join('');

        this.galleryPickerGrid.querySelectorAll('.gallery-picker__item:not([style*="pointer-events"])').forEach(item => {
            item.addEventListener('click', () => {
                this.addImageFromGallery(item.dataset.id);
                // Don't close picker - allow selecting multiple
                // Update the item to show it's selected
                item.style.opacity = '0.5';
                item.style.pointerEvents = 'none';
            });
        });

        this.galleryPickerModal.classList.remove('hidden');
    }

    closeGalleryPicker() {
        if (this.galleryPickerModal) {
            this.galleryPickerModal.classList.add('hidden');
        }
    }

    updateGenerateButtonState() {
        if (this.generateBtn) {
            const hasImages = this.selectedImages.length > 0;
            const hasPrompt = this.promptInput?.value.trim();
            this.generateBtn.disabled = !hasImages || !hasPrompt;
        }
    }

    async generate() {
        if (this.isGenerating) return;

        const prompt = this.promptInput?.value.trim();

        if (this.selectedImages.length === 0) {
            showToast('Please select at least one image', 'warning');
            return;
        }

        if (!prompt) {
            showToast('Please enter a prompt', 'warning');
            return;
        }

        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';

        // Build image sources for API
        const imageSources = this.selectedImages.map(img => ({
            type: 'gallery',  // All images are in gallery after import
            id: img.id
        }));

        this.isGenerating = true;
        this.showGeneratingState();

        try {
            const result = await api.fuseImages(
                imageSources,
                prompt,
                this.selectedAspectRatio,
                model
            );

            if (result.success) {
                showToast('Image fusion complete!', 'success');
                this.showResult(result);

                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }
            } else {
                showToast(result.error || 'Failed to generate fused image', 'error');
            }
        } catch (error) {
            console.error('Fusion error:', error);
            showToast('An error occurred during fusion', 'error');
        } finally {
            this.isGenerating = false;
            this.hideGeneratingState();
        }
    }

    showGeneratingState() {
        if (this.generateBtn) {
            this.generateBtn.disabled = true;
            this.generateBtn.innerHTML = `
                <span class="spinner" style="width: 20px; height: 20px;"></span>
                Generating...
            `;
        }

        if (this.statusSection) {
            this.statusSection.classList.remove('hidden');
            this.statusSection.innerHTML = `
                <div class="generation-status__header">
                    <span class="badge badge--warning">Fusing Images</span>
                </div>
                <div class="progress-bar" style="margin-top: var(--spacing-sm);">
                    <div class="progress-bar__fill" style="width: 50%; animation: pulse 1s infinite;"></div>
                </div>
                <p style="font-size: var(--font-size-sm); color: var(--text-secondary); margin-top: var(--spacing-sm);">
                    AI is combining ${this.selectedImages.length} image${this.selectedImages.length > 1 ? 's' : ''}...
                </p>
            `;
        }
    }

    hideGeneratingState() {
        const hasImages = this.selectedImages.length > 0;
        const hasPrompt = this.promptInput?.value.trim();

        if (this.generateBtn) {
            this.generateBtn.disabled = !hasImages || !hasPrompt;
            this.generateBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                    <path d="M2 17l10 5 10-5"></path>
                    <path d="M2 12l10 5 10-5"></path>
                </svg>
                Fuse Images
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
            <h3 class="section-title">Fused Result</h3>
            <div class="fusion-result">
                <img class="fusion-result__image" src="${api.getImageUrl(result.image_id)}" alt="Fused image">
                <div class="fusion-result__info">
                    <span class="badge badge--success">${result.source_count} images fused</span>
                    <span class="badge badge--secondary">${this.selectedAspectRatio}</span>
                </div>
            </div>
            <div class="fusion-actions">
                <button class="btn btn--secondary btn--sm" id="btn-fusion-generate-another">
                    Generate Another
                </button>
                <button class="btn btn--primary btn--sm" id="btn-fusion-use-result">
                    Use as Reference
                </button>
            </div>
        `;

        // Generate another button
        const generateAnotherBtn = this.resultPreview.querySelector('#btn-fusion-generate-another');
        if (generateAnotherBtn) {
            generateAnotherBtn.addEventListener('click', () => {
                this.resultPreview.classList.add('hidden');
            });
        }

        // Use as reference button - adds result to selected images
        const useResultBtn = this.resultPreview.querySelector('#btn-fusion-use-result');
        if (useResultBtn) {
            useResultBtn.addEventListener('click', () => {
                this.clearAllImages();
                this.addImageFromGallery(result.image_id);
                this.resultPreview.classList.add('hidden');
            });
        }

        this.resultPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Navigate to fusion view with a specific image
    async fuseFromGallery(imageId) {
        if (window.app) {
            app.showView('fusion');
        }
        this.clearAllImages();
        await this.addImageFromGallery(imageId);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global Image Fusion manager instance
let imageFusionManager;
