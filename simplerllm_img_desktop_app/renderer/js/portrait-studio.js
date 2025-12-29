/**
 * Portrait Studio Manager - Handles face reference image generation UI and logic
 */
class PortraitStudioManager {
    constructor() {
        this.isGenerating = false;
        this.faceImage = null;  // { id: string, type: 'gallery' | 'local', path?: string }
        this.selectedAspectRatio = '1:1';
        this.selectedPreset = null;
        this.presets = [];

        // DOM elements
        this.facePreview = document.getElementById('portrait-face-preview');
        this.resultPreview = document.getElementById('portrait-result-preview');
        this.promptInput = document.getElementById('portrait-prompt-input');
        this.modelSelect = document.getElementById('portrait-model-select');
        this.generateBtn = document.getElementById('btn-generate-portrait');
        this.statusSection = document.getElementById('portrait-status');
        this.aspectRatioSelector = document.getElementById('portrait-aspect-ratio-selector');
        this.browseBtn = document.getElementById('btn-browse-portrait-face');
        this.galleryBtn = document.getElementById('btn-select-portrait-gallery');
        this.galleryPickerModal = document.getElementById('portrait-gallery-picker-modal');
        this.galleryPickerGrid = document.getElementById('portrait-gallery-picker-grid');
        this.closePickerBtn = document.getElementById('btn-close-portrait-gallery-picker');
        this.configureApiKeyBtn = document.getElementById('btn-configure-api-key-portrait');
        this.presetsContainer = document.getElementById('portrait-presets');

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadModels();
        await this.loadPresets();
    }

    setupEventListeners() {
        // Generate button
        if (this.generateBtn) {
            this.generateBtn.addEventListener('click', () => this.generate());
        }

        // Browse local button for face image
        if (this.browseBtn) {
            this.browseBtn.addEventListener('click', () => this.browseFaceImage());
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

        // Preset selection via event delegation
        if (this.presetsContainer) {
            this.presetsContainer.addEventListener('click', (e) => {
                const presetCard = e.target.closest('.preset-card');
                if (presetCard) {
                    this.selectPreset(presetCard.dataset.preset);
                }
            });
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
        const result = await api.getPortraitPresets();
        if (result.success && result.presets) {
            this.presets = result.presets;
            this.renderPresets();
        }
    }

    renderPresets() {
        if (!this.presetsContainer) return;

        this.presetsContainer.innerHTML = this.presets.map(preset => `
            <div class="preset-card ${this.selectedPreset === preset.id ? 'preset-card--active' : ''}"
                 data-preset="${preset.id}">
                <div class="preset-card__icon">${preset.icon}</div>
                <div class="preset-card__info">
                    <div class="preset-card__name">${preset.name}</div>
                    <div class="preset-card__description">${preset.description}</div>
                </div>
            </div>
        `).join('');
    }

    selectPreset(presetId) {
        // Toggle: clicking same preset deselects it
        this.selectedPreset = this.selectedPreset === presetId ? null : presetId;
        this.renderPresets();
        this.updateGenerateButtonState();
    }

    getPresetName(presetId) {
        const preset = this.presets.find(p => p.id === presetId);
        return preset ? preset.name : presetId;
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

    async setFaceFromGallery(imageId) {
        this.faceImage = {
            id: imageId,
            type: 'gallery'
        };
        this.renderFacePreview();
        this.updateGenerateButtonState();
    }

    async setFaceFromLocal(filePath) {
        try {
            const result = await api.importImage(filePath);

            if (result.success) {
                this.faceImage = {
                    id: result.image_id,
                    type: 'local',
                    path: filePath
                };
                this.renderFacePreview();

                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }

                showToast('Face image imported successfully!', 'success');
            } else {
                showToast(result.error || 'Failed to import image', 'error');
            }
        } catch (error) {
            console.error('Import error:', error);
            showToast('Failed to import face image', 'error');
        }

        this.updateGenerateButtonState();
    }

    async browseFaceImage() {
        if (!window.electronAPI?.openFileDialog) {
            showToast('File dialog not available', 'error');
            return;
        }

        try {
            const filePath = await window.electronAPI.openFileDialog({
                title: 'Select Face Reference Image',
                filters: [
                    { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp'] }
                ]
            });

            if (filePath) {
                await this.setFaceFromLocal(filePath);
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
                ${img.type === 'portrait' ? '<span class="image-badge image-badge--portrait">Portrait</span>' : ''}
                ${img.type === 'imported' ? '<span class="image-badge image-badge--imported">Imported</span>' : ''}
            </div>
        `).join('');

        this.galleryPickerGrid.querySelectorAll('.gallery-picker__item').forEach(item => {
            item.addEventListener('click', () => {
                this.setFaceFromGallery(item.dataset.id);
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

    clearFace() {
        this.faceImage = null;

        if (this.facePreview) {
            this.facePreview.innerHTML = `
                <div class="face-preview__empty">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="8" r="4"></circle>
                        <path d="M4 20c0-4 4-6 8-6s8 2 8 6"></path>
                    </svg>
                    <p>Select a face reference image</p>
                </div>
            `;
        }

        if (this.resultPreview) {
            this.resultPreview.classList.add('hidden');
        }

        this.updateGenerateButtonState();
    }

    renderFacePreview() {
        if (!this.facePreview || !this.faceImage) return;

        this.facePreview.innerHTML = `
            <div class="face-preview__image-container">
                <img class="face-preview__image" src="${api.getImageUrl(this.faceImage.id)}" alt="Face reference">
                <button class="face-preview__clear" title="Clear selection">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
        `;

        const clearBtn = this.facePreview.querySelector('.face-preview__clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearFace());
        }
    }

    updateGenerateButtonState() {
        if (this.generateBtn) {
            // Either prompt OR preset is required
            const hasPrompt = this.promptInput?.value.trim();
            const hasPreset = !!this.selectedPreset;
            this.generateBtn.disabled = !hasPrompt && !hasPreset;
        }
    }

    async generate() {
        if (this.isGenerating) return;

        const prompt = this.promptInput?.value.trim();
        const preset = this.selectedPreset;

        if (!prompt && !preset) {
            showToast('Please select a preset or enter a prompt', 'warning');
            return;
        }

        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';

        this.isGenerating = true;
        this.showGeneratingState();

        try {
            const result = await api.generatePortrait(
                this.faceImage?.id || null,
                null,  // We always use gallery ID after import
                prompt,
                preset,
                this.selectedAspectRatio,
                model
            );

            if (result.success) {
                showToast('Portrait generated successfully!', 'success');
                this.showResult(result);

                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }
            } else {
                showToast(result.error || 'Failed to generate portrait', 'error');
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
                    <span class="badge badge--warning">Creating Portrait</span>
                </div>
                <div class="progress-bar" style="margin-top: var(--spacing-sm);">
                    <div class="progress-bar__fill" style="width: 50%; animation: pulse 1s infinite;"></div>
                </div>
                <p style="font-size: var(--font-size-sm); color: var(--text-secondary); margin-top: var(--spacing-sm);">
                    AI is generating your image...
                </p>
            `;
        }
    }

    hideGeneratingState() {
        const hasPrompt = this.promptInput?.value.trim();
        const hasPreset = !!this.selectedPreset;

        if (this.generateBtn) {
            this.generateBtn.disabled = !hasPrompt && !hasPreset;
            this.generateBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="8" r="4"></circle>
                    <path d="M4 20c0-4 4-6 8-6s8 2 8 6"></path>
                </svg>
                Generate Portrait
            `;
        }

        if (this.statusSection) {
            this.statusSection.classList.add('hidden');
        }
    }

    showResult(result) {
        if (!this.resultPreview) return;

        const presetBadge = result.preset_used
            ? `<span class="badge badge--secondary">${this.getPresetName(result.preset_used)}</span>`
            : '';

        this.resultPreview.classList.remove('hidden');
        this.resultPreview.innerHTML = `
            <h3 class="section-title">Generated Portrait</h3>
            <div class="portrait-result">
                <img class="portrait-result__image" src="${api.getImageUrl(result.image_id)}" alt="Generated portrait">
                <div class="portrait-result__info">
                    ${presetBadge}
                    <span class="badge badge--success">${this.selectedAspectRatio}</span>
                </div>
            </div>
            <div class="portrait-actions">
                <button class="btn btn--secondary btn--sm" id="btn-generate-another-portrait">
                    Generate Another
                </button>
                <button class="btn btn--primary btn--sm" id="btn-use-as-face-portrait">
                    Use as Face Reference
                </button>
            </div>
        `;

        // Generate another button
        const generateAnotherBtn = this.resultPreview.querySelector('#btn-generate-another-portrait');
        if (generateAnotherBtn) {
            generateAnotherBtn.addEventListener('click', () => {
                this.resultPreview.classList.add('hidden');
            });
        }

        // Use as face reference button
        const useAsFaceBtn = this.resultPreview.querySelector('#btn-use-as-face-portrait');
        if (useAsFaceBtn) {
            useAsFaceBtn.addEventListener('click', () => {
                this.setFaceFromGallery(result.image_id);
                this.resultPreview.classList.add('hidden');
            });
        }

        this.resultPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Navigate to portrait view with a specific face image
    async generateFromGallery(imageId) {
        if (window.app) {
            app.showView('portrait');
        }
        await this.setFaceFromGallery(imageId);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global Portrait Studio manager instance
let portraitStudioManager;
