/**
 * Sketch Manager - Handles sketch canvas and sketch-to-image generation
 */
class SketchManager {
    constructor() {
        this.isGenerating = false;
        this.selectedAspectRatio = '1:1';

        // Canvas state
        this.canvas = null;
        this.ctx = null;
        this.isDrawing = false;
        this.brushSize = 5;
        this.brushColor = '#000000';
        this.lastX = 0;
        this.lastY = 0;

        // DOM elements
        this.canvasContainer = document.getElementById('sketch-canvas-container');
        this.promptInput = document.getElementById('sketch-prompt-input');
        this.modelSelect = document.getElementById('sketch-model-select');
        this.generateBtn = document.getElementById('btn-sketch-generate');
        this.clearBtn = document.getElementById('btn-sketch-clear');
        this.statusSection = document.getElementById('sketch-status');
        this.previewSection = document.getElementById('sketch-preview-section');
        this.aspectRatioSelector = document.getElementById('sketch-aspect-ratio-selector');
        this.brushSizeSlider = document.getElementById('brush-size-slider');
        this.brushSizeDisplay = document.getElementById('brush-size-display');
        this.colorPicker = document.getElementById('brush-color-picker');
        this.colorPresets = document.getElementById('color-presets');
        this.configureApiKeyBtn = document.getElementById('btn-configure-api-key-sketch');

        this.init();
    }

    init() {
        this.setupCanvas();
        this.setupEventListeners();
        this.loadModels();
        this.setDefaultPrompt();
    }

    /**
     * Set default prompt text
     */
    setDefaultPrompt() {
        if (this.promptInput) {
            this.promptInput.value = 'Transform this sketch into a realistic detailed image';
        }
    }

    /**
     * Setup canvas and create initial canvas element
     */
    setupCanvas() {
        this.createCanvas();
    }

    /**
     * Create canvas element based on current aspect ratio
     */
    createCanvas() {
        if (!this.canvasContainer) return;

        // Calculate dimensions based on aspect ratio
        const dimensions = this.getCanvasDimensions(this.selectedAspectRatio);

        // Clear existing canvas
        this.canvasContainer.innerHTML = '';

        // Create new canvas
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'sketch-canvas';
        this.canvas.width = dimensions.width;
        this.canvas.height = dimensions.height;
        this.canvas.classList.add('sketch-canvas');

        this.canvasContainer.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');

        // Set white background
        this.clearCanvas();

        // Setup drawing events
        this.setupCanvasEvents();
    }

    /**
     * Get canvas dimensions based on aspect ratio
     */
    getCanvasDimensions(ratio) {
        const maxWidth = 500;
        const maxHeight = 500;

        switch (ratio) {
            case '16:9':
                return { width: maxWidth, height: Math.round(maxWidth * 9 / 16) };
            case '9:16':
                return { width: Math.round(maxHeight * 9 / 16), height: maxHeight };
            case '1:1':
            default:
                return { width: maxWidth, height: maxWidth };
        }
    }

    /**
     * Setup canvas drawing events
     */
    setupCanvasEvents() {
        if (!this.canvas) return;

        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());

        // Touch events for tablet/touch support
        this.canvas.addEventListener('touchstart', (e) => this.startDrawing(e));
        this.canvas.addEventListener('touchmove', (e) => this.draw(e));
        this.canvas.addEventListener('touchend', () => this.stopDrawing());
    }

    /**
     * Setup event listeners for UI controls
     */
    setupEventListeners() {
        // Generate button
        if (this.generateBtn) {
            this.generateBtn.addEventListener('click', () => this.generate());
        }

        // Clear button
        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => this.clearCanvas());
        }

        // Enter key in prompt (with Shift+Enter for new line)
        if (this.promptInput) {
            this.promptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.generate();
                }
            });
        }

        // Aspect ratio selector
        if (this.aspectRatioSelector) {
            this.aspectRatioSelector.addEventListener('click', (e) => {
                const btn = e.target.closest('.aspect-btn');
                if (btn) {
                    this.selectAspectRatio(btn);
                }
            });
        }

        // Brush size slider
        if (this.brushSizeSlider) {
            this.brushSizeSlider.addEventListener('input', (e) => {
                this.brushSize = parseInt(e.target.value);
                if (this.brushSizeDisplay) {
                    this.brushSizeDisplay.textContent = `${this.brushSize}px`;
                }
            });
        }

        // Color picker
        if (this.colorPicker) {
            this.colorPicker.addEventListener('input', (e) => {
                this.brushColor = e.target.value;
                this.updateColorPresetSelection();
            });
        }

        // Color presets
        if (this.colorPresets) {
            this.colorPresets.addEventListener('click', (e) => {
                const preset = e.target.closest('.color-preset');
                if (preset) {
                    this.brushColor = preset.dataset.color;
                    if (this.colorPicker) {
                        this.colorPicker.value = this.brushColor;
                    }
                    this.updateColorPresetSelection();
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
    }

    // ============================================
    // Drawing Methods
    // ============================================

    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getPosition(e);
        this.lastX = pos.x;
        this.lastY = pos.y;
    }

    draw(e) {
        if (!this.isDrawing) return;
        e.preventDefault();

        const pos = this.getPosition(e);

        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.strokeStyle = this.brushColor;
        this.ctx.lineWidth = this.brushSize;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.stroke();

        this.lastX = pos.x;
        this.lastY = pos.y;
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    getPosition(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    clearCanvas() {
        if (!this.ctx || !this.canvas) return;
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    // ============================================
    // UI Control Methods
    // ============================================

    selectAspectRatio(btn) {
        if (!this.aspectRatioSelector) return;
        this.aspectRatioSelector.querySelectorAll('.aspect-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.selectedAspectRatio = btn.dataset.ratio;
        // Recreate canvas with new dimensions
        this.createCanvas();
    }

    updateColorPresetSelection() {
        if (!this.colorPresets) return;
        this.colorPresets.querySelectorAll('.color-preset').forEach(preset => {
            preset.classList.toggle('active', preset.dataset.color.toLowerCase() === this.brushColor.toLowerCase());
        });
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

    // ============================================
    // Canvas Data Methods
    // ============================================

    getCanvasDataUrl() {
        if (!this.canvas) return null;
        return this.canvas.toDataURL('image/png');
    }

    getCanvasBase64() {
        const dataUrl = this.getCanvasDataUrl();
        if (!dataUrl) return null;
        // Remove "data:image/png;base64," prefix
        return dataUrl.split(',')[1];
    }

    isCanvasEmpty() {
        if (!this.canvas || !this.ctx) return true;

        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;

        // Check if all pixels are white (255, 255, 255)
        for (let i = 0; i < data.length; i += 4) {
            if (data[i] !== 255 || data[i + 1] !== 255 || data[i + 2] !== 255) {
                return false;
            }
        }
        return true;
    }

    // ============================================
    // Generation Methods
    // ============================================

    async generate() {
        if (this.isGenerating) return;

        // Validate sketch
        if (this.isCanvasEmpty()) {
            showToast('Please draw something on the canvas', 'warning');
            return;
        }

        const prompt = this.promptInput?.value.trim();
        if (!prompt) {
            showToast('Please enter a prompt describing the desired output', 'warning');
            this.promptInput?.focus();
            return;
        }

        const model = this.modelSelect?.value || 'gemini-2.5-flash-image-preview';
        const aspectRatio = this.selectedAspectRatio || '1:1';
        const sketchData = this.getCanvasBase64();

        if (!sketchData) {
            showToast('Failed to capture sketch', 'error');
            return;
        }

        // Start generating
        this.isGenerating = true;
        this.showGeneratingState();

        try {
            const result = await api.sketchToImage(sketchData, prompt, model, 'google', aspectRatio);

            if (result.success) {
                showToast('Image generated from sketch!', 'success');
                this.showPreview(result);

                // Refresh gallery to show new images
                if (window.galleryManager) {
                    await galleryManager.loadGallery();
                }

                // Don't clear prompt - user might want to iterate
            } else {
                showToast(result.error || 'Failed to generate image from sketch', 'error');
            }
        } catch (error) {
            console.error('Sketch generation error:', error);
            showToast('An error occurred during generation', 'error');
        } finally {
            this.isGenerating = false;
            this.hideGeneratingState();
        }
    }

    // ============================================
    // UI State Methods
    // ============================================

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
                    <span class="badge badge--warning">Generating from Sketch</span>
                </div>
                <div class="progress-bar" style="margin-top: var(--spacing-sm);">
                    <div class="progress-bar__fill" style="width: 50%; animation: pulse 1s infinite;"></div>
                </div>
                <p style="font-size: var(--font-size-sm); color: var(--text-secondary); margin-top: var(--spacing-sm);">
                    Transforming your sketch...
                </p>
            `;
        }
    }

    hideGeneratingState() {
        if (this.generateBtn) {
            this.generateBtn.disabled = false;
            this.generateBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 19l7-7 3 3-7 7-3-3z"></path>
                    <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path>
                    <path d="M2 2l7.586 7.586"></path>
                    <circle cx="11" cy="11" r="2"></circle>
                </svg>
                Generate from Sketch
            `;
        }

        if (this.statusSection) {
            this.statusSection.classList.add('hidden');
        }
    }

    showPreview(result) {
        if (!this.previewSection) return;

        this.previewSection.classList.remove('hidden');
        this.previewSection.innerHTML = `
            <h3 class="section-title">Generated Image</h3>
            <div class="preview-comparison">
                <div class="preview-comparison__item">
                    <span class="preview-comparison__label">Your Sketch</span>
                    <img class="preview-image preview-image--sketch" src="${api.getImageUrl(result.sketch_id)}" alt="Original sketch">
                </div>
                <div class="preview-comparison__arrow">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                </div>
                <div class="preview-comparison__item">
                    <span class="preview-comparison__label">Generated Result</span>
                    <img class="preview-image" src="${api.getImageUrl(result.image_id)}" alt="Generated image">
                </div>
            </div>
        `;

        this.previewSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global sketch manager instance
let sketchManager;
