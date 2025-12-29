/**
 * Character Generator Manager - Two-phase character creation with style and pose presets
 */
class CharacterGeneratorManager {
    constructor() {
        this.isGenerating = false;
        this.phase = 'create'; // 'create' or 'variation'
        this.generatedCharacter = null; // { id: string, prompt: string }
        this.selectedStyle = 'anime';
        this.selectedPose = 'front_view';
        this.selectedAspectRatio = '1:1';
        this.styles = [];
        this.poses = [];

        // DOM elements - Create Phase
        this.createPhaseSection = document.getElementById('character-create-phase');
        this.promptInput = document.getElementById('character-prompt-input');
        this.stylePresets = document.getElementById('character-style-presets');
        this.aspectRatioSelector = document.getElementById('character-aspect-ratio-selector');
        this.modelSelect = document.getElementById('character-model-select');
        this.generateBtn = document.getElementById('btn-generate-character');
        this.statusSection = document.getElementById('character-status');

        // DOM elements - Variation Phase
        this.variationPhaseSection = document.getElementById('character-variation-phase');
        this.characterPreview = document.getElementById('character-reference-preview');
        this.posePresets = document.getElementById('character-pose-presets');
        this.variationPromptInput = document.getElementById('character-variation-prompt');
        this.variationAspectSelector = document.getElementById('character-variation-aspect-selector');
        this.generateVariationBtn = document.getElementById('btn-generate-variation');
        this.variationStatus = document.getElementById('character-variation-status');
        this.backToCreateBtn = document.getElementById('btn-back-to-create');

        // DOM elements - Result
        this.resultSection = document.getElementById('character-result-section');

        // DOM elements - Gallery picker
        this.galleryPickerModal = document.getElementById('character-gallery-picker-modal');
        this.galleryPickerGrid = document.getElementById('character-gallery-picker-grid');
        this.closePickerBtn = document.getElementById('btn-close-character-gallery-picker');
        this.selectFromGalleryBtn = document.getElementById('btn-select-character-gallery');

        // DOM elements - Controls containers
        this.createControls = document.getElementById('character-create-controls');
        this.variationControls = document.getElementById('character-variation-controls');

        // API key button
        this.configureApiKeyBtn = document.getElementById('btn-configure-api-key-character');

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadModels();
        await this.loadPresets();
    }

    setupEventListeners() {
        // Generate character button
        if (this.generateBtn) {
            this.generateBtn.addEventListener('click', () => this.generateCharacter());
        }

        // Generate variation button
        if (this.generateVariationBtn) {
            this.generateVariationBtn.addEventListener('click', () => this.generateVariation());
        }

        // Back to create button
        if (this.backToCreateBtn) {
            this.backToCreateBtn.addEventListener('click', () => this.showCreatePhase());
        }

        // Select from gallery button
        if (this.selectFromGalleryBtn) {
            this.selectFromGalleryBtn.addEventListener('click', () => this.openGalleryPicker());
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

        // Aspect ratio selection (create phase)
        if (this.aspectRatioSelector) {
            this.aspectRatioSelector.querySelectorAll('.aspect-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.selectAspectRatio(btn.dataset.ratio, 'create');
                });
            });
        }

        // Aspect ratio selection (variation phase)
        if (this.variationAspectSelector) {
            this.variationAspectSelector.querySelectorAll('.aspect-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.selectAspectRatio(btn.dataset.ratio, 'variation');
                });
            });
        }

        // Style preset selection (delegated)
        if (this.stylePresets) {
            this.stylePresets.addEventListener('click', (e) => {
                const card = e.target.closest('.style-preset-card');
                if (card) {
                    this.selectStyle(card.dataset.style);
                }
            });
        }

        // Pose preset selection (delegated)
        if (this.posePresets) {
            this.posePresets.addEventListener('click', (e) => {
                const card = e.target.closest('.pose-preset-card');
                if (card) {
                    this.selectPose(card.dataset.pose);
                }
            });
        }

        // Update button states when prompts change
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
        const result = await api.getCharacterPresets();
        if (result.success) {
            this.styles = result.styles;
            this.poses = result.poses;
            this.renderStylePresets();
            this.renderPosePresets();
        }
    }

    renderStylePresets() {
        if (!this.stylePresets) return;

        const styleIcons = {
            'anime': '&#127884;',
            'realistic': '&#128247;',
            'cartoon': '&#127912;',
            'pixel_art': '&#128126;',
            'fantasy': '&#128481;'
        };

        this.stylePresets.innerHTML = this.styles.map(style => `
            <div class="style-preset-card ${this.selectedStyle === style.id ? 'style-preset-card--active' : ''}"
                 data-style="${style.id}">
                <div class="style-preset-card__icon">${styleIcons[style.id] || '&#127912;'}</div>
                <div class="style-preset-card__info">
                    <div class="style-preset-card__name">${style.name}</div>
                    <div class="style-preset-card__description">${style.description}</div>
                </div>
            </div>
        `).join('');
    }

    renderPosePresets() {
        if (!this.posePresets) return;

        const poseIcons = {
            'front_view': '&#129485;',
            'side_profile': '&#128100;',
            'three_quarter': '&#128260;',
            'back_view': '&#128694;',
            'action_pose': '&#9889;',
            'sitting': '&#129681;',
            'full_body': '&#129489;',
            'portrait_closeup': '&#128522;'
        };

        this.posePresets.innerHTML = this.poses.map(pose => `
            <div class="pose-preset-card ${this.selectedPose === pose.id ? 'pose-preset-card--active' : ''}"
                 data-pose="${pose.id}">
                <div class="pose-preset-card__icon">${poseIcons[pose.id] || '&#127917;'}</div>
                <div class="pose-preset-card__info">
                    <div class="pose-preset-card__name">${pose.name}</div>
                    <div class="pose-preset-card__description">${pose.description}</div>
                </div>
            </div>
        `).join('');
    }

    selectStyle(styleId) {
        this.selectedStyle = styleId;
        this.renderStylePresets();
    }

    selectPose(poseId) {
        this.selectedPose = poseId;
        this.renderPosePresets();
    }

    selectAspectRatio(ratio, phase) {
        this.selectedAspectRatio = ratio;

        const selector = phase === 'create' ? this.aspectRatioSelector : this.variationAspectSelector;
        if (selector) {
            selector.querySelectorAll('.aspect-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.ratio === ratio);
            });
        }
    }

    updateGenerateButtonState() {
        if (this.generateBtn) {
            const hasPrompt = this.promptInput?.value.trim();
            this.generateBtn.disabled = !hasPrompt;
        }
    }

    showCreatePhase() {
        this.phase = 'create';
        if (this.createPhaseSection) this.createPhaseSection.classList.remove('hidden');
        if (this.variationPhaseSection) this.variationPhaseSection.classList.add('hidden');
        if (this.resultSection) this.resultSection.classList.add('hidden');
        if (this.createControls) this.createControls.classList.remove('hidden');
        if (this.variationControls) this.variationControls.classList.add('hidden');
    }

    showVariationPhase() {
        this.phase = 'variation';
        if (this.createPhaseSection) this.createPhaseSection.classList.add('hidden');
        if (this.variationPhaseSection) this.variationPhaseSection.classList.remove('hidden');
        if (this.createControls) this.createControls.classList.add('hidden');
        if (this.variationControls) this.variationControls.classList.remove('hidden');
        this.renderCharacterPreview();
    }

    renderCharacterPreview() {
        if (!this.characterPreview || !this.generatedCharacter) return;

        this.characterPreview.innerHTML = `
            <div class="character-preview__image-container">
                <img class="character-preview__image" src="${api.getImageUrl(this.generatedCharacter.id)}" alt="Generated character">
                <button class="character-preview__change" title="Change reference" id="btn-change-character">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21.5 2v6h-6M2.5 22v-6h6M2 12a10 10 0 0 1 16.1-7.9M22 12a10 10 0 0 1-16.1 7.9"></path>
                    </svg>
                </button>
            </div>
            <p class="character-preview__prompt">${this.escapeHtml(this.generatedCharacter.prompt || 'Character')}</p>
        `;

        const changeBtn = this.characterPreview.querySelector('#btn-change-character');
        if (changeBtn) {
            changeBtn.addEventListener('click', () => this.openGalleryPicker());
        }
    }

    async generateCharacter() {
        if (this.isGenerating) return;

        const prompt = this.promptInput?.value.trim();
        if (!prompt) {
            showToast('Please enter a character description', 'warning');
            return;
        }

        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';

        this.isGenerating = true;
        this.showGeneratingState('create');

        try {
            const result = await api.generateCharacter(
                prompt,
                this.selectedStyle,
                this.selectedAspectRatio,
                model
            );

            if (result.success) {
                showToast('Character generated successfully!', 'success');

                this.generatedCharacter = {
                    id: result.image_id,
                    prompt: prompt
                };

                // Add to gallery
                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }

                // Show result and transition to variation phase
                this.showCharacterResult(result);
            } else {
                showToast(result.error || 'Failed to generate character', 'error');
            }
        } catch (error) {
            console.error('Generation error:', error);
            showToast('An error occurred during generation', 'error');
        } finally {
            this.isGenerating = false;
            this.hideGeneratingState('create');
        }
    }

    async generateVariation() {
        if (this.isGenerating || !this.generatedCharacter) return;

        const customPrompt = this.variationPromptInput?.value.trim();
        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';

        this.isGenerating = true;
        this.showGeneratingState('variation');

        try {
            const result = await api.generateCharacterVariation(
                this.generatedCharacter.id,
                this.selectedPose,
                customPrompt,
                this.selectedAspectRatio,
                model
            );

            if (result.success) {
                showToast('Variation generated successfully!', 'success');
                this.showVariationResult(result);

                // Add to gallery
                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }
            } else {
                showToast(result.error || 'Failed to generate variation', 'error');
            }
        } catch (error) {
            console.error('Variation error:', error);
            showToast('An error occurred during generation', 'error');
        } finally {
            this.isGenerating = false;
            this.hideGeneratingState('variation');
        }
    }

    showGeneratingState(phase) {
        const btn = phase === 'create' ? this.generateBtn : this.generateVariationBtn;
        const status = phase === 'create' ? this.statusSection : this.variationStatus;

        if (btn) {
            btn.disabled = true;
            btn.innerHTML = `
                <span class="spinner" style="width: 20px; height: 20px;"></span>
                Generating...
            `;
        }

        if (status) {
            status.classList.remove('hidden');
            status.innerHTML = `
                <div class="generation-status__header">
                    <span class="badge badge--warning">${phase === 'create' ? 'Creating Character' : 'Creating Variation'}</span>
                </div>
                <div class="progress-bar" style="margin-top: var(--spacing-sm);">
                    <div class="progress-bar__fill" style="width: 50%; animation: pulse 1s infinite;"></div>
                </div>
                <p style="font-size: var(--font-size-sm); color: var(--text-secondary); margin-top: var(--spacing-sm);">
                    AI is generating your ${phase === 'create' ? 'character' : 'variation'}...
                </p>
            `;
        }
    }

    hideGeneratingState(phase) {
        const btn = phase === 'create' ? this.generateBtn : this.generateVariationBtn;
        const status = phase === 'create' ? this.statusSection : this.variationStatus;
        const hasPrompt = this.promptInput?.value.trim();

        if (btn) {
            btn.disabled = phase === 'create' ? !hasPrompt : false;
            btn.innerHTML = phase === 'create'
                ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                       <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                       <circle cx="12" cy="7" r="4"></circle>
                   </svg>
                   Generate Character`
                : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                       <polyline points="17 1 21 5 17 9"></polyline>
                       <path d="M3 11V9a4 4 0 0 1 4-4h14"></path>
                       <polyline points="7 23 3 19 7 15"></polyline>
                       <path d="M21 13v2a4 4 0 0 1-4 4H3"></path>
                   </svg>
                   Generate Variation`;
        }

        if (status) {
            status.classList.add('hidden');
        }
    }

    showCharacterResult(result) {
        if (!this.resultSection) return;

        // Hide the create phase when showing result so it takes the full space
        if (this.createPhaseSection) {
            this.createPhaseSection.classList.add('hidden');
        }

        const styleName = this.styles.find(s => s.id === result.style_used)?.name || 'Custom';

        this.resultSection.classList.remove('hidden');
        this.resultSection.innerHTML = `
            <h3 class="section-title">Generated Character</h3>
            <div class="character-result">
                <img class="character-result__image" src="${api.getImageUrl(result.image_id)}" alt="Generated character">
                <div class="character-result__info">
                    <span class="badge badge--success">${styleName} Style</span>
                </div>
            </div>
            <div class="character-result__actions">
                <button class="btn btn--secondary btn--sm" id="btn-create-new-character">
                    Create New Character
                </button>
                <button class="btn btn--primary btn--sm" id="btn-create-variations">
                    Create Variations
                </button>
            </div>
        `;

        // Create new character button
        const newCharBtn = this.resultSection.querySelector('#btn-create-new-character');
        if (newCharBtn) {
            newCharBtn.addEventListener('click', () => {
                this.resultSection.classList.add('hidden');
                if (this.createPhaseSection) {
                    this.createPhaseSection.classList.remove('hidden');
                }
                this.promptInput.value = '';
                this.updateGenerateButtonState();
            });
        }

        // Create variations button
        const variationsBtn = this.resultSection.querySelector('#btn-create-variations');
        if (variationsBtn) {
            variationsBtn.addEventListener('click', () => {
                this.resultSection.classList.add('hidden');
                this.showVariationPhase();
            });
        }

        this.resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    showVariationResult(result) {
        if (!this.resultSection) return;

        const poseName = this.poses.find(p => p.id === result.pose_used)?.name || result.pose_used;

        this.resultSection.classList.remove('hidden');
        this.resultSection.innerHTML = `
            <h3 class="section-title">Generated Variation</h3>
            <div class="variation-comparison">
                <div class="variation-comparison__item">
                    <span class="variation-comparison__label">Original</span>
                    <img class="variation-comparison__image" src="${api.getImageUrl(this.generatedCharacter.id)}" alt="Original">
                </div>
                <div class="variation-comparison__arrow">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                </div>
                <div class="variation-comparison__item">
                    <span class="variation-comparison__label">${poseName}</span>
                    <img class="variation-comparison__image" src="${api.getImageUrl(result.image_id)}" alt="Variation">
                </div>
            </div>
            <div class="character-result__actions">
                <button class="btn btn--secondary btn--sm" id="btn-another-variation">
                    Generate Another Variation
                </button>
                <button class="btn btn--primary btn--sm" id="btn-use-as-reference">
                    Use as New Reference
                </button>
            </div>
        `;

        // Another variation button
        const anotherBtn = this.resultSection.querySelector('#btn-another-variation');
        if (anotherBtn) {
            anotherBtn.addEventListener('click', () => {
                this.resultSection.classList.add('hidden');
            });
        }

        // Use as reference button
        const useAsRefBtn = this.resultSection.querySelector('#btn-use-as-reference');
        if (useAsRefBtn) {
            useAsRefBtn.addEventListener('click', () => {
                this.generatedCharacter = {
                    id: result.image_id,
                    prompt: this.generatedCharacter.prompt
                };
                this.renderCharacterPreview();
                this.resultSection.classList.add('hidden');
            });
        }

        this.resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
                ${img.type === 'character' ? '<span class="image-badge image-badge--character">Character</span>' : ''}
                ${img.type === 'character_variation' ? '<span class="image-badge image-badge--variation">Variation</span>' : ''}
            </div>
        `).join('');

        this.galleryPickerGrid.querySelectorAll('.gallery-picker__item').forEach(item => {
            item.addEventListener('click', () => {
                this.setCharacterFromGallery(item.dataset.id);
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

    async setCharacterFromGallery(imageId) {
        const metadata = await api.getImageMetadata(imageId);

        this.generatedCharacter = {
            id: imageId,
            prompt: metadata.success ? metadata.metadata?.prompt : 'Selected character'
        };

        if (this.phase === 'create') {
            this.showVariationPhase();
        } else {
            this.renderCharacterPreview();
        }
    }

    // Navigate to character generator with an existing character
    async useCharacterFromGallery(imageId) {
        if (window.app) {
            app.showView('character');
        }
        await this.setCharacterFromGallery(imageId);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global Character Generator manager instance
let characterGeneratorManager;
