/**
 * Social Media Post Generator Manager - Handles social media image generation UI and logic
 */
class SocialMediaManager {
    constructor() {
        this.isGenerating = false;
        this.referenceImage = null;  // { id: string, type: 'gallery' }
        this.selectedPlatform = null;
        this.selectedContentType = null;
        this.platforms = [];
        this.contentTypes = [];

        // DOM elements
        this.referencePreview = document.getElementById('social-reference-preview');
        this.resultPreview = document.getElementById('social-result-preview');
        this.promptInput = document.getElementById('social-prompt-input');
        this.modelSelect = document.getElementById('social-model-select');
        this.generateBtn = document.getElementById('btn-generate-social');
        this.statusSection = document.getElementById('social-status');
        this.platformsContainer = document.getElementById('social-platform-presets');
        this.contentTypesContainer = document.getElementById('social-content-presets');
        this.galleryBtn = document.getElementById('btn-select-social-gallery');
        this.galleryPickerModal = document.getElementById('social-gallery-picker-modal');
        this.galleryPickerGrid = document.getElementById('social-gallery-picker-grid');
        this.closePickerBtn = document.getElementById('btn-close-social-gallery-picker');
        this.configureApiKeyBtn = document.getElementById('btn-configure-api-key-social');

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

        // Platform selection (delegated)
        if (this.platformsContainer) {
            this.platformsContainer.addEventListener('click', (e) => {
                const platformCard = e.target.closest('.platform-preset-card');
                if (platformCard) {
                    this.selectPlatform(platformCard.dataset.platform);
                }
            });
        }

        // Content type selection (delegated with toggle)
        if (this.contentTypesContainer) {
            this.contentTypesContainer.addEventListener('click', (e) => {
                const contentCard = e.target.closest('.content-preset-card');
                if (contentCard) {
                    this.selectContentType(contentCard.dataset.content);
                }
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

    async loadPresets() {
        const result = await api.getSocialMediaPresets();
        if (result.success) {
            this.platforms = result.platforms || [];
            this.contentTypes = result.content_types || [];
            this.renderPlatformPresets();
            this.renderContentTypePresets();
        }
    }

    renderPlatformPresets() {
        if (!this.platformsContainer) return;

        // Group platforms by network
        const groups = {};
        const platformOrder = ['instagram', 'facebook', 'twitter', 'linkedin'];
        const platformNames = {
            'instagram': 'Instagram',
            'facebook': 'Facebook',
            'twitter': 'X / Twitter',
            'linkedin': 'LinkedIn'
        };
        const platformIcons = {
            'instagram': '📷',
            'facebook': '📘',
            'twitter': '🐦',
            'linkedin': '💼'
        };

        this.platforms.forEach(platform => {
            const group = platform.platform;
            if (!groups[group]) {
                groups[group] = [];
            }
            groups[group].push(platform);
        });

        let html = '';
        platformOrder.forEach(groupKey => {
            if (groups[groupKey]) {
                html += `
                    <div class="platform-group">
                        <div class="platform-group__header">
                            <span class="platform-group__header-icon">${platformIcons[groupKey] || ''}</span>
                            ${platformNames[groupKey] || groupKey}
                        </div>
                        <div class="platform-group__options">
                            ${groups[groupKey].map(platform => `
                                <div class="platform-preset-card ${this.selectedPlatform === platform.id ? 'platform-preset-card--active' : ''}"
                                     data-platform="${platform.id}">
                                    <div class="platform-preset-card__name">${platform.name.replace(platformNames[groupKey] + ' ', '')}</div>
                                    <div class="platform-preset-card__ratio">${platform.aspect_ratio}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }
        });

        this.platformsContainer.innerHTML = html;
    }

    renderContentTypePresets() {
        if (!this.contentTypesContainer) return;

        this.contentTypesContainer.innerHTML = this.contentTypes.map(content => `
            <div class="content-preset-card ${this.selectedContentType === content.id ? 'content-preset-card--active' : ''}"
                 data-content="${content.id}"
                 title="${content.description}">
                <div class="content-preset-card__icon">${content.icon}</div>
                <div class="content-preset-card__name">${content.name}</div>
            </div>
        `).join('');
    }

    selectPlatform(platformId) {
        this.selectedPlatform = platformId;
        this.renderPlatformPresets();
        this.updateGenerateButtonState();
    }

    selectContentType(contentId) {
        // Toggle behavior - click again to deselect
        this.selectedContentType = this.selectedContentType === contentId ? null : contentId;
        this.renderContentTypePresets();
    }

    async setReferenceFromGallery(imageId) {
        this.referenceImage = {
            id: imageId,
            type: 'gallery'
        };
        this.renderReferencePreview();
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
                ${img.type === 'social_media' ? '<span class="image-badge image-badge--social_media">Social</span>' : ''}
                ${img.type === 'imported' ? '<span class="image-badge image-badge--imported">Imported</span>' : ''}
            </div>
        `).join('');

        this.galleryPickerGrid.querySelectorAll('.gallery-picker__item').forEach(item => {
            item.addEventListener('click', () => {
                this.setReferenceFromGallery(item.dataset.id);
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

    clearReference() {
        this.referenceImage = null;

        if (this.referencePreview) {
            this.referencePreview.innerHTML = `
                <div class="social-reference-preview__empty">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                    <p>Select a reference image for brand consistency</p>
                </div>
            `;
        }
    }

    renderReferencePreview() {
        if (!this.referencePreview || !this.referenceImage) return;

        this.referencePreview.innerHTML = `
            <div class="social-reference-preview__image-container">
                <img class="social-reference-preview__image" src="${api.getImageUrl(this.referenceImage.id)}" alt="Brand reference">
                <button class="social-reference-preview__clear" title="Clear selection">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
        `;

        const clearBtn = this.referencePreview.querySelector('.social-reference-preview__clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearReference());
        }
    }

    updateGenerateButtonState() {
        if (this.generateBtn) {
            // Platform is required, prompt is required
            const hasPrompt = this.promptInput?.value.trim();
            const hasPlatform = !!this.selectedPlatform;
            this.generateBtn.disabled = !hasPrompt || !hasPlatform;
        }
    }

    async generate() {
        if (this.isGenerating) return;

        const prompt = this.promptInput?.value.trim();

        if (!prompt) {
            showToast('Please enter a post description', 'warning');
            return;
        }

        if (!this.selectedPlatform) {
            showToast('Please select a platform and size', 'warning');
            return;
        }

        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';

        this.isGenerating = true;
        this.showGeneratingState();

        try {
            const result = await api.generateSocialMediaPost(
                prompt,
                this.selectedPlatform,
                this.selectedContentType,
                this.referenceImage?.id || null,
                model
            );

            if (result.success) {
                showToast('Social media post generated successfully!', 'success');
                this.showResult(result);

                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }
            } else {
                showToast(result.error || 'Failed to generate post', 'error');
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
                    <span class="badge badge--warning">Creating Post</span>
                </div>
                <div class="progress-bar" style="margin-top: var(--spacing-sm);">
                    <div class="progress-bar__fill" style="width: 50%; animation: pulse 1s infinite;"></div>
                </div>
                <p style="font-size: var(--font-size-sm); color: var(--text-secondary); margin-top: var(--spacing-sm);">
                    AI is generating your social media post image...
                </p>
            `;
        }
    }

    hideGeneratingState() {
        const hasPrompt = this.promptInput?.value.trim();
        const hasPlatform = !!this.selectedPlatform;

        if (this.generateBtn) {
            this.generateBtn.disabled = !hasPrompt || !hasPlatform;
            this.generateBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect>
                    <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path>
                    <line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line>
                </svg>
                Generate Post
            `;
        }

        if (this.statusSection) {
            this.statusSection.classList.add('hidden');
        }
    }

    showResult(result) {
        if (!this.resultPreview) return;

        const platform = this.platforms.find(p => p.id === result.platform_preset_used);
        const contentType = this.contentTypes.find(c => c.id === result.content_type_used);

        this.resultPreview.classList.remove('hidden');
        this.resultPreview.innerHTML = `
            <h3 class="section-title">Generated Post</h3>
            <div class="social-result">
                <img class="social-result__image" src="${api.getImageUrl(result.image_id)}" alt="Generated social media post">
                <div class="social-result__info">
                    ${platform ? `<span class="social-result__badge">${platform.icon} ${platform.name}</span>` : ''}
                    ${contentType ? `<span class="social-result__badge">${contentType.icon} ${contentType.name}</span>` : ''}
                </div>
            </div>
            <div class="social-actions">
                <button class="btn btn--secondary btn--sm" id="btn-generate-another-social">
                    Generate Another
                </button>
                <button class="btn btn--primary btn--sm" id="btn-use-as-reference">
                    Use as Brand Reference
                </button>
            </div>
        `;

        // Generate another button
        const generateAnotherBtn = this.resultPreview.querySelector('#btn-generate-another-social');
        if (generateAnotherBtn) {
            generateAnotherBtn.addEventListener('click', () => {
                this.resultPreview.classList.add('hidden');
            });
        }

        // Use as reference button
        const useAsReferenceBtn = this.resultPreview.querySelector('#btn-use-as-reference');
        if (useAsReferenceBtn) {
            useAsReferenceBtn.addEventListener('click', () => {
                this.setReferenceFromGallery(result.image_id);
                this.resultPreview.classList.add('hidden');
            });
        }

        this.resultPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global social media manager instance
let socialMediaManager;
