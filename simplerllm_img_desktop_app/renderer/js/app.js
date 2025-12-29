/**
 * Main Application Controller
 */
class App {
    constructor() {
        this.currentView = 'tools';
        this.backendStatus = false;
        this.init();
    }

    async init() {
        // Initialize window controls
        this.initWindowControls();

        // Initialize managers
        settingsManager = new SettingsManager();
        galleryManager = new GalleryManager();
        imageGenerationManager = new ImageGenerationManager();
        imageEditingManager = new ImageEditingManager();
        sketchManager = new SketchManager();
        productEnhancerManager = new ProductEnhancerManager();
        youTubeThumbnailManager = new YouTubeThumbnailManager();
        portraitStudioManager = new PortraitStudioManager();
        characterGeneratorManager = new CharacterGeneratorManager();
        imageFusionManager = new ImageFusionManager();

        // Check backend status
        await this.checkBackendStatus();

        // Load gallery
        await galleryManager.loadGallery();

        // Set up navigation
        this.initNavigation();

        console.log('App initialized');
    }

    initWindowControls() {
        // Window control buttons
        const minimizeBtn = document.getElementById('btn-minimize');
        const maximizeBtn = document.getElementById('btn-maximize');
        const closeBtn = document.getElementById('btn-close');

        if (minimizeBtn) {
            minimizeBtn.addEventListener('click', () => {
                window.electronAPI?.minimizeWindow();
            });
        }

        if (maximizeBtn) {
            maximizeBtn.addEventListener('click', () => {
                window.electronAPI?.maximizeWindow();
            });
        }

        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                window.electronAPI?.closeWindow();
            });
        }
    }

    initNavigation() {
        // Tool card clicks
        document.querySelectorAll('.tool-card').forEach(card => {
            card.addEventListener('click', () => {
                const tool = card.dataset.tool;
                if (tool) {
                    this.showView(tool);
                }
            });
        });

        // Back buttons
        document.querySelectorAll('[data-action="back"]').forEach(btn => {
            btn.addEventListener('click', () => {
                this.showView('tools');
            });
        });

        // Configure API Key button
        const configureApiKeyBtn = document.getElementById('btn-configure-api-key');
        if (configureApiKeyBtn) {
            configureApiKeyBtn.addEventListener('click', () => {
                settingsManager.open();
            });
        }
    }

    showView(viewName) {
        // Hide all views
        document.querySelectorAll('.tools-view, .generate-view, .edit-view, .sketch-view, .enhance-view, .thumbnail-view, .portrait-view, .character-view, .fusion-view').forEach(view => {
            view.classList.add('hidden');
        });

        // Show target view
        const targetView = document.getElementById(`${viewName}-view`);
        if (targetView) {
            targetView.classList.remove('hidden');
            this.currentView = viewName;
        }
    }

    async checkBackendStatus() {
        try {
            const result = await api.checkHealth();
            this.backendStatus = result.status === 'ok';

            // Update status badge if exists
            const badge = document.getElementById('status-badge');
            if (badge) {
                if (this.backendStatus) {
                    badge.className = 'badge badge--success';
                    badge.textContent = 'Ready';
                } else {
                    badge.className = 'badge badge--error';
                    badge.textContent = 'Offline';
                }
            }
        } catch (error) {
            this.backendStatus = false;
            console.error('Backend status check failed:', error);
        }
    }
}

// ============================================
// Toast Notifications
// ============================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Remove after delay
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================
// Initialize App
// ============================================

let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new App();
    window.app = app;
});
