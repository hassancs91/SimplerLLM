/**
 * Image Editing Manager - Handles image editing UI and logic
 */
class ImageEditingManager {
    constructor() {
        this.isEditing = false;
        this.sourceImage = null;  // { id: string, type: 'gallery' | 'local', path?: string }
        this.versionTree = null;
        this.currentVersion = null;

        // DOM elements
        this.sourcePreview = document.getElementById('source-preview');
        this.versionSection = document.getElementById('version-section');
        this.versionTreeContainer = document.getElementById('version-tree');
        this.promptInput = document.getElementById('edit-prompt-input');
        this.modelSelect = document.getElementById('edit-model-select');
        this.editBtn = document.getElementById('btn-apply-edit');
        this.statusSection = document.getElementById('edit-status');
        this.previewSection = document.getElementById('edit-preview-section');
        this.browseBtn = document.getElementById('btn-browse-image');
        this.galleryBtn = document.getElementById('btn-select-gallery');
        this.galleryPickerModal = document.getElementById('gallery-picker-modal');
        this.galleryPickerGrid = document.getElementById('gallery-picker-grid');
        this.closePickerBtn = document.getElementById('btn-close-gallery-picker');
        this.configureApiKeyBtn = document.getElementById('btn-configure-api-key-edit');

        this.init();
    }

    init() {
        // Edit button
        if (this.editBtn) {
            this.editBtn.addEventListener('click', () => this.applyEdit());
        }

        // Enter key in prompt (with Shift+Enter for new line)
        if (this.promptInput) {
            this.promptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey && this.sourceImage) {
                    e.preventDefault();
                    this.applyEdit();
                }
            });
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

        // Version tree click delegation
        if (this.versionTreeContainer) {
            this.versionTreeContainer.addEventListener('click', (e) => {
                const node = e.target.closest('.version-node');
                const deleteBtn = e.target.closest('.version-node__delete');

                if (deleteBtn) {
                    e.stopPropagation();
                    const nodeElement = deleteBtn.closest('.version-node');
                    if (nodeElement) {
                        this.deleteVersion(nodeElement.dataset.id);
                    }
                } else if (node) {
                    this.selectVersion(node.dataset.id);
                }
            });
        }

        // Load models
        this.loadModels();
    }

    /**
     * Load available models from the API
     */
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

    /**
     * Set source image from gallery
     */
    async setSourceFromGallery(imageId) {
        this.sourceImage = {
            id: imageId,
            type: 'gallery'
        };

        // Render preview
        this.renderSourcePreview();

        // Load version tree
        await this.loadVersionTree(imageId);

        // Enable edit button
        this.updateEditButtonState();
    }

    /**
     * Set source image from local file
     */
    async setSourceFromLocal(filePath) {
        try {
            // Import the image to gallery first
            const result = await api.importImage(filePath);

            if (result.success) {
                this.sourceImage = {
                    id: result.image_id,
                    type: 'local',
                    path: filePath
                };

                // Render preview
                this.renderSourcePreview();

                // No version tree for newly imported images (they are the root)
                this.versionTree = null;
                this.hideVersionSection();

                // Update gallery if visible
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

        // Enable edit button
        this.updateEditButtonState();
    }

    /**
     * Browse for a local image file
     */
    async browseLocalImage() {
        if (!window.electronAPI?.openFileDialog) {
            showToast('File dialog not available', 'error');
            return;
        }

        try {
            const filePath = await window.electronAPI.openFileDialog({
                title: 'Select Image to Edit',
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

    /**
     * Open gallery picker modal
     */
    async openGalleryPicker() {
        if (!this.galleryPickerModal || !this.galleryPickerGrid) return;

        // Load gallery images
        const result = await api.getGallery();
        if (!result.success) {
            showToast('Failed to load gallery', 'error');
            return;
        }

        // Render gallery grid
        this.galleryPickerGrid.innerHTML = result.images.map(img => `
            <div class="gallery-picker__item" data-id="${img.id}">
                <img src="${api.getImageUrl(img.id)}" alt="${this.escapeHtml(img.prompt || 'Image')}">
                ${img.type === 'edited' ? '<span class="image-badge image-badge--edited">Edited</span>' : ''}
                ${img.type === 'imported' ? '<span class="image-badge image-badge--imported">Imported</span>' : ''}
                ${img.type === 'sketch' ? '<span class="image-badge image-badge--sketch">Sketch</span>' : ''}
            </div>
        `).join('');

        // Add click handlers
        this.galleryPickerGrid.querySelectorAll('.gallery-picker__item').forEach(item => {
            item.addEventListener('click', () => {
                this.setSourceFromGallery(item.dataset.id);
                this.closeGalleryPicker();
            });
        });

        // Show modal
        this.galleryPickerModal.classList.remove('hidden');
    }

    /**
     * Close gallery picker modal
     */
    closeGalleryPicker() {
        if (this.galleryPickerModal) {
            this.galleryPickerModal.classList.add('hidden');
        }
    }

    /**
     * Load version tree for an image
     */
    async loadVersionTree(imageId) {
        try {
            const result = await api.getVersionTree(imageId);

            if (result.success && result.has_versions) {
                this.versionTree = result.tree;
                this.currentVersion = imageId;
                this.showVersionSection();
                this.renderVersionTree();
            } else {
                this.versionTree = null;
                this.hideVersionSection();
            }
        } catch (error) {
            console.error('Error loading version tree:', error);
            this.hideVersionSection();
        }
    }

    /**
     * Select a version from the tree
     */
    selectVersion(imageId) {
        this.currentVersion = imageId;
        this.sourceImage = {
            id: imageId,
            type: 'gallery'
        };

        // Update preview
        this.renderSourcePreview();

        // Update tree highlighting
        this.renderVersionTree();

        // Enable edit button
        this.updateEditButtonState();
    }

    /**
     * Delete a version
     */
    async deleteVersion(imageId) {
        if (!confirm('Are you sure you want to delete this version?')) {
            return;
        }

        try {
            const result = await api.deleteImage(imageId);

            if (result.success) {
                showToast('Version deleted', 'success');

                // If we deleted the current source, clear it
                if (this.sourceImage?.id === imageId) {
                    this.clearSource();
                } else {
                    // Reload version tree
                    if (this.sourceImage?.id) {
                        await this.loadVersionTree(this.sourceImage.id);
                    }
                }

                // Update gallery
                if (window.galleryManager) {
                    galleryManager.removeImage(imageId);
                }
            } else {
                showToast(result.error || 'Failed to delete version', 'error');
            }
        } catch (error) {
            console.error('Delete error:', error);
            showToast('Failed to delete version', 'error');
        }
    }

    /**
     * Clear the source image
     */
    clearSource() {
        this.sourceImage = null;
        this.versionTree = null;
        this.currentVersion = null;

        // Reset preview
        if (this.sourcePreview) {
            this.sourcePreview.innerHTML = `
                <div class="source-preview__empty">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                    <p>Select an image to edit</p>
                </div>
            `;
        }

        this.hideVersionSection();
        this.updateEditButtonState();
    }

    /**
     * Apply the edit to the source image
     */
    async applyEdit() {
        if (this.isEditing || !this.sourceImage) return;

        const prompt = this.promptInput?.value.trim();
        if (!prompt) {
            showToast('Please enter an edit prompt', 'warning');
            this.promptInput?.focus();
            return;
        }

        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';

        // Start editing
        this.isEditing = true;
        this.showEditingState();

        try {
            const result = await api.editImage(
                this.sourceImage.id,
                null,  // no local path since we always import first
                prompt,
                model
            );

            if (result.success) {
                showToast('Image edited successfully!', 'success');
                this.showPreview(result);

                // Update source to the new edited image
                this.sourceImage = {
                    id: result.image_id,
                    type: 'gallery'
                };
                this.currentVersion = result.image_id;

                // Reload version tree
                await this.loadVersionTree(result.image_id);

                // Update preview
                this.renderSourcePreview();

                // Add to gallery
                if (window.galleryManager) {
                    galleryManager.addImageToTop(result.metadata);
                }

                // Clear prompt
                if (this.promptInput) {
                    this.promptInput.value = '';
                }
            } else {
                showToast(result.error || 'Failed to edit image', 'error');
            }
        } catch (error) {
            console.error('Edit error:', error);
            showToast('An error occurred during editing', 'error');
        } finally {
            this.isEditing = false;
            this.hideEditingState();
        }
    }

    /**
     * Render the source image preview
     */
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

        // Add clear button handler
        const clearBtn = this.sourcePreview.querySelector('.source-preview__clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearSource());
        }
    }

    /**
     * Render the version tree
     */
    renderVersionTree() {
        if (!this.versionTreeContainer || !this.versionTree) return;

        const renderNode = (node, depth = 0) => {
            const isActive = node.id === this.currentVersion;
            const typeIcon = node.type === 'generated' ? '🎨' :
                            node.type === 'edited' ? '✨' :
                            node.type === 'imported' ? '📁' : '📷';

            return `
                <div class="version-node ${isActive ? 'version-node--active' : ''}"
                     data-id="${node.id}" style="--depth: ${depth}">
                    <div class="version-node__connector"></div>
                    <div class="version-node__content">
                        <img class="version-node__thumb" src="${api.getImageUrl(node.id)}" alt="Version">
                        <div class="version-node__info">
                            <span class="version-node__type">${typeIcon}</span>
                            <span class="version-node__prompt">${this.truncate(node.prompt, 40)}</span>
                            <span class="version-node__meta">${this.formatDate(node.timestamp)}</span>
                        </div>
                        <button class="version-node__delete" title="Delete version">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                ${node.children && node.children.length > 0 ?
                    node.children.map(child => renderNode(child, depth + 1)).join('') : ''}
            `;
        };

        this.versionTreeContainer.innerHTML = renderNode(this.versionTree);
    }

    showVersionSection() {
        if (this.versionSection) {
            this.versionSection.classList.remove('hidden');
        }
    }

    hideVersionSection() {
        if (this.versionSection) {
            this.versionSection.classList.add('hidden');
        }
    }

    updateEditButtonState() {
        if (this.editBtn) {
            this.editBtn.disabled = !this.sourceImage;
        }
    }

    showEditingState() {
        // Disable button
        if (this.editBtn) {
            this.editBtn.disabled = true;
            this.editBtn.innerHTML = `
                <span class="spinner" style="width: 20px; height: 20px;"></span>
                Editing...
            `;
        }

        // Show status
        if (this.statusSection) {
            this.statusSection.classList.remove('hidden');
            this.statusSection.innerHTML = `
                <div class="generation-status__header">
                    <span class="badge badge--warning">Editing</span>
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

    hideEditingState() {
        // Enable button
        if (this.editBtn) {
            this.editBtn.disabled = !this.sourceImage;
            this.editBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                </svg>
                Apply Edit
            `;
        }

        // Hide status
        if (this.statusSection) {
            this.statusSection.classList.add('hidden');
        }
    }

    showPreview(result) {
        if (!this.previewSection) return;

        this.previewSection.classList.remove('hidden');
        this.previewSection.innerHTML = `
            <h3 class="section-title">Edit Result</h3>
            <div class="preview-image-container">
                <img class="preview-image" src="${api.getImageUrl(result.image_id)}" alt="Edited image">
                <div class="preview-prompt">${this.escapeHtml(result.metadata?.prompt || 'No prompt')}</div>
            </div>
            <div class="preview-actions" style="margin-top: var(--spacing-sm);">
                <button class="btn btn--secondary btn--sm" id="btn-continue-editing">
                    Continue Editing
                </button>
            </div>
        `;

        // Continue editing button
        const continueBtn = this.previewSection.querySelector('#btn-continue-editing');
        if (continueBtn) {
            continueBtn.addEventListener('click', () => {
                this.previewSection.classList.add('hidden');
                this.promptInput?.focus();
            });
        }

        // Scroll into view
        this.previewSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    /**
     * Navigate to edit view with a specific image
     */
    async editFromGallery(imageId) {
        // Switch to edit view
        if (window.app) {
            app.showView('edit');
        }

        // Set source image
        await this.setSourceFromGallery(imageId);
    }

    truncate(text, maxLength) {
        if (!text) return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }

    formatDate(timestamp) {
        if (!timestamp) return '';
        try {
            const date = new Date(timestamp);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch {
            return '';
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global image editing manager instance
let imageEditingManager;
