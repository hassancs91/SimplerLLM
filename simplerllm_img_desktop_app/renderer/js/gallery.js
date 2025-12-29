/**
 * Gallery Manager - Handles image gallery display and interactions
 */
class GalleryManager {
    constructor() {
        this.images = [];
        this.currentImage = null;
        this.gridContainer = document.getElementById('gallery-grid');
        this.lightbox = document.getElementById('lightbox');
        this.init();
    }

    init() {
        // Lightbox close button
        const closeBtn = document.querySelector('.lightbox__close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeLightbox());
        }

        // Close lightbox on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.lightbox && !this.lightbox.classList.contains('hidden')) {
                this.closeLightbox();
            }
        });

        // Close lightbox on background click
        if (this.lightbox) {
            this.lightbox.addEventListener('click', (e) => {
                if (e.target === this.lightbox) {
                    this.closeLightbox();
                }
            });
        }

        // Lightbox action buttons
        const lightboxOpenFileBtn = document.getElementById('lightbox-open-file');
        if (lightboxOpenFileBtn) {
            lightboxOpenFileBtn.addEventListener('click', () => {
                if (this.currentImage) {
                    this.openFileLocation(this.currentImage.id);
                }
            });
        }

        const lightboxDeleteBtn = document.getElementById('lightbox-delete');
        if (lightboxDeleteBtn) {
            lightboxDeleteBtn.addEventListener('click', () => {
                if (this.currentImage) {
                    this.deleteImage(this.currentImage.id);
                }
            });
        }

        // Gallery grid event delegation
        if (this.gridContainer) {
            this.gridContainer.addEventListener('click', (e) => {
                const card = e.target.closest('.image-card');
                if (!card) return;

                const imageId = card.dataset.id;

                // Check if clicked on action button
                const actionBtn = e.target.closest('.image-card__action-btn');
                if (actionBtn) {
                    e.stopPropagation();
                    const action = actionBtn.dataset.action;
                    if (action === 'open-file') {
                        this.openFileLocation(imageId);
                    } else if (action === 'delete') {
                        this.deleteImage(imageId);
                    } else if (action === 'edit') {
                        this.editImage(imageId);
                    }
                    return;
                }

                // Otherwise open lightbox
                this.openLightbox(imageId);
            });
        }
    }

    async loadGallery() {
        console.log('Loading gallery...');
        const result = await api.getGallery();
        console.log('Gallery API result:', result);
        if (result.success) {
            this.images = result.images || [];
            console.log('Gallery loaded:', this.images.length, 'images');
            this.render();
        } else {
            console.error('Failed to load gallery:', result.error);
        }
    }

    render() {
        if (!this.gridContainer) return;

        if (this.images.length === 0) {
            this.gridContainer.innerHTML = `
                <div class="gallery-empty">
                    <div class="gallery-empty__icon">🎨</div>
                    <div class="gallery-empty__text">No images yet</div>
                    <div class="gallery-empty__subtext">Generate your first image using the panel on the right</div>
                </div>
            `;
            return;
        }

        this.gridContainer.innerHTML = this.images.map(image => `
            <div class="image-card" data-id="${image.id}">
                ${image.type === 'edited' ? '<span class="image-card__badge image-card__badge--edited">Edited</span>' : ''}
                ${image.type === 'imported' ? '<span class="image-card__badge image-card__badge--imported">Imported</span>' : ''}
                ${image.type === 'sketch' ? '<span class="image-card__badge image-card__badge--sketch">Sketch</span>' : ''}
                <img class="image-card__img" src="${api.getImageUrl(image.id)}" alt="${this.escapeHtml(image.prompt)}" loading="lazy">
                <div class="image-card__overlay">
                    <div class="image-card__prompt">${this.escapeHtml(image.prompt)}</div>
                </div>
                <div class="image-card__actions">
                    <button class="image-card__action-btn" data-action="edit" title="Edit this image">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                        </svg>
                    </button>
                    <button class="image-card__action-btn" data-action="open-file" title="Open file location">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                        </svg>
                    </button>
                    <button class="image-card__action-btn" data-action="delete" title="Delete">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `).join('');

        // Update count
        const countEl = document.getElementById('gallery-count');
        if (countEl) {
            countEl.textContent = `${this.images.length} image${this.images.length !== 1 ? 's' : ''}`;
        }
    }

    openLightbox(imageId) {
        const image = this.images.find(img => img.id === imageId);
        if (!image || !this.lightbox) return;

        this.currentImage = image;

        // Set image
        const imgEl = document.getElementById('lightbox-image');
        if (imgEl) {
            imgEl.src = api.getImageUrl(image.id);
        }

        // Set info
        const promptEl = document.getElementById('lightbox-prompt');
        if (promptEl) {
            promptEl.textContent = image.prompt;
        }

        const metaEl = document.getElementById('lightbox-meta');
        if (metaEl) {
            const date = new Date(image.timestamp).toLocaleString();
            metaEl.textContent = `${image.model} • ${date}`;
        }

        // Show lightbox
        this.lightbox.classList.remove('hidden');
    }

    closeLightbox() {
        if (this.lightbox) {
            this.lightbox.classList.add('hidden');
            this.currentImage = null;
        }
    }

    async deleteImage(imageId) {
        if (!confirm('Are you sure you want to delete this image?')) {
            return;
        }

        const result = await api.deleteImage(imageId);
        if (result.success) {
            showToast('Image deleted', 'success');
            await this.loadGallery();

            // Close lightbox if viewing deleted image
            if (this.currentImage && this.currentImage.id === imageId) {
                this.closeLightbox();
            }
        } else {
            showToast(result.error || 'Failed to delete image', 'error');
        }
    }

    addImageToTop(imageData) {
        // Add new image to the beginning of the array
        this.images.unshift(imageData);
        // Re-render the gallery
        this.render();

        // Highlight the new image briefly
        setTimeout(() => {
            const newCard = this.gridContainer?.querySelector(`[data-id="${imageData.id}"]`);
            if (newCard) {
                newCard.classList.add('image-card--new');
                setTimeout(() => newCard.classList.remove('image-card--new'), 1000);
            }
        }, 50);
    }

    async openFileLocation(imageId) {
        if (!imageId) return;

        try {
            // Get the file path from the backend
            const result = await api.getImagePath(imageId);
            if (result.success && result.path) {
                // Use Electron to show the file in folder
                await window.electronAPI.showItemInFolder(result.path);
            } else {
                showToast('Could not find file location', 'error');
            }
        } catch (error) {
            console.error('Open file location error:', error);
            showToast('Failed to open file location', 'error');
        }
    }

    /**
     * Navigate to edit view with the selected image
     */
    editImage(imageId) {
        if (window.imageEditingManager) {
            imageEditingManager.editFromGallery(imageId);
        }
    }

    /**
     * Remove an image from the gallery array without reloading
     */
    removeImage(imageId) {
        const index = this.images.findIndex(img => img.id === imageId);
        if (index !== -1) {
            this.images.splice(index, 1);
            this.render();
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global gallery manager instance
let galleryManager;
