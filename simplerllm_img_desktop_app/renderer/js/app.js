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
        socialMediaManager = new SocialMediaManager();

        // Check backend status
        await this.checkBackendStatus();

        // Load gallery
        await galleryManager.loadGallery();

        // Set up navigation
        this.initNavigation();

        // Load branding and populate status bar
        await this.loadBranding();

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
        document.querySelectorAll('.tools-view, .generate-view, .edit-view, .sketch-view, .enhance-view, .thumbnail-view, .portrait-view, .character-view, .fusion-view, .social-view').forEach(view => {
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

    async loadBranding() {
        try {
            const response = await fetch('js/branding.json');
            const branding = await response.json();
            this.populateStatusBar(branding);
        } catch (error) {
            console.error('Failed to load branding:', error);
        }
    }

    populateStatusBar(branding) {
        // App name and version
        const appNameEl = document.getElementById('status-bar-app-name');
        const versionEl = document.getElementById('status-bar-version');
        const copyrightEl = document.getElementById('status-bar-copyright');
        const linksEl = document.getElementById('status-bar-links');

        if (appNameEl) appNameEl.textContent = branding.appName || '';
        if (versionEl) versionEl.textContent = `v${branding.version}` || '';
        if (copyrightEl) copyrightEl.textContent = branding.copyright || '';

        // Social links
        if (linksEl && branding.social) {
            linksEl.innerHTML = '';

            // Website link
            if (branding.website) {
                linksEl.appendChild(this.createStatusBarLink(
                    branding.website.url,
                    branding.website.label,
                    this.getWebsiteIcon()
                ));
            }

            // Social links
            Object.entries(branding.social).forEach(([key, value]) => {
                if (value.url) {
                    linksEl.appendChild(this.createStatusBarLink(
                        value.url,
                        value.label,
                        this.getSocialIcon(key)
                    ));
                }
            });
        }
    }

    createStatusBarLink(url, label, iconSvg) {
        const link = document.createElement('a');
        link.href = url;
        link.className = 'status-bar__link';
        link.title = label;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.innerHTML = iconSvg;
        return link;
    }

    getWebsiteIcon() {
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
        </svg>`;
    }

    getSocialIcon(platform) {
        const icons = {
            twitter: `<svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
            </svg>`,
            github: `<svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>`,
            youtube: `<svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
            </svg>`
        };
        return icons[platform] || this.getWebsiteIcon();
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
