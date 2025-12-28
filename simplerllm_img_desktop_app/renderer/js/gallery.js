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
            <div class="image-card" data-id="${image.id}" onclick="galleryManager.openLightbox('${image.id}')">
                <img class="image-card__img" src="${api.getImageUrl(image.id)}" alt="${this.escapeHtml(image.prompt)}" loading="lazy">
                <div class="image-card__overlay">
                    <div class="image-card__prompt">${this.escapeHtml(image.prompt)}</div>
                </div>
                <div class="image-card__actions">
                    <button class="image-card__action-btn" onclick="event.stopPropagation(); galleryManager.downloadImage('${image.id}')" title="Download">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                    </button>
                    <button class="image-card__action-btn" onclick="event.stopPropagation(); galleryManager.deleteImage('${image.id}')" title="Delete">
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

    async downloadImage(imageId) {
        if (!imageId) return;

        const image = this.images.find(img => img.id === imageId);
        const filename = image?.prompt
            ? `simplerllm-${image.prompt.slice(0, 30).replace(/[^a-z0-9]/gi, '_')}.png`
            : `simplerllm-${imageId}.png`;

        try {
            showToast('Preparing download...', 'info');

            // Fetch the image as a blob
            const response = await fetch(api.getImageUrl(imageId));
            const blob = await response.blob();

            // Create object URL and download
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Clean up the object URL
            URL.revokeObjectURL(url);

            showToast('Download started!', 'success');
        } catch (error) {
            console.error('Download error:', error);
            showToast('Failed to download image', 'error');
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
