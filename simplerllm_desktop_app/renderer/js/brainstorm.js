/**
 * Brainstorm Manager
 * Handles brainstorming sessions with real-time tree visualization
 */
class BrainstormManager {
    constructor() {
        // DOM elements - Input
        this.promptInput = document.getElementById('brainstorm-prompt');
        this.providerSelect = document.getElementById('brainstorm-provider');
        this.modelInput = document.getElementById('brainstorm-model');

        // DOM elements - Sliders
        this.maxDepthSlider = document.getElementById('max-depth-slider');
        this.maxDepthValue = document.getElementById('max-depth-value');
        this.ideasPerLevelSlider = document.getElementById('ideas-per-level-slider');
        this.ideasPerLevelValue = document.getElementById('ideas-per-level-value');
        this.topNSlider = document.getElementById('top-n-slider');
        this.topNValue = document.getElementById('top-n-value');
        this.minQualitySlider = document.getElementById('min-quality-slider');
        this.minQualityValue = document.getElementById('min-quality-value');

        // DOM elements - Buttons
        this.btnStart = document.getElementById('btn-start-brainstorm');
        this.btnClear = document.getElementById('btn-clear-brainstorm');
        this.btnCancel = document.getElementById('btn-cancel-brainstorm');
        this.btnExport = document.getElementById('btn-export-results');
        this.btnBack = document.getElementById('btn-back-from-brainstorm');

        // DOM elements - Zoom controls
        this.btnZoomIn = document.getElementById('btn-zoom-in');
        this.btnZoomOut = document.getElementById('btn-zoom-out');
        this.btnZoomReset = document.getElementById('btn-zoom-reset');

        // DOM elements - Status
        this.statusSection = document.getElementById('brainstorm-status');
        this.statusBadge = document.getElementById('brainstorm-status-badge');
        this.progressBar = document.getElementById('brainstorm-progress');
        this.currentDepthEl = document.getElementById('current-depth');
        this.ideasCountEl = document.getElementById('ideas-count');
        this.elapsedTimeEl = document.getElementById('elapsed-time');

        // DOM elements - Stats
        this.statsSection = document.getElementById('brainstorm-stats');
        this.statTotalIdeas = document.getElementById('stat-total-ideas');
        this.statBestScore = document.getElementById('stat-best-score');
        this.statAvgScore = document.getElementById('stat-avg-score');
        this.statExecTime = document.getElementById('stat-exec-time');

        // DOM elements - Idea detail
        this.ideaDetail = document.getElementById('brainstorm-idea-detail');
        this.detailScore = document.getElementById('detail-score');
        this.detailText = document.getElementById('detail-text');
        this.detailReasoning = document.getElementById('detail-reasoning');
        this.detailDepth = document.getElementById('detail-depth');
        this.detailId = document.getElementById('detail-id');

        // DOM elements - Code section
        this.codeSection = document.getElementById('brainstorm-code-section');
        this.codeSectionHeader = document.getElementById('toggle-code-section');
        this.codeSectionContent = document.getElementById('code-section-content');
        this.codeToggleIcon = document.getElementById('code-toggle-icon');
        this.codeSnippet = document.getElementById('code-snippet');
        this.btnCopyCode = document.getElementById('btn-copy-code');

        // State
        this.streamController = null;
        this.isRunning = false;
        this.startTime = null;
        this.timerInterval = null;
        this.result = null;
        this.ideasCount = 0;
        this.currentDepth = 0;

        // Tree visualizer
        this.treeVisualizer = null;

        this.init();
    }

    init() {
        // Initialize tree visualizer
        this.treeVisualizer = new TreeVisualizer('brainstorm-tree-container', 'brainstorm-svg');
        this.treeVisualizer.onNodeClick = (nodeId, data) => this.showIdeaDetail(data);

        // Event listeners - Buttons
        this.btnStart.addEventListener('click', () => this.startBrainstorm());
        this.btnClear.addEventListener('click', () => this.clearBrainstorm());
        this.btnCancel.addEventListener('click', () => this.cancelBrainstorm());
        this.btnExport.addEventListener('click', () => this.exportResults());
        this.btnBack.addEventListener('click', () => app.navigateTo('tools-view'));

        // Event listeners - Zoom
        this.btnZoomIn.addEventListener('click', () => this.treeVisualizer.zoomIn());
        this.btnZoomOut.addEventListener('click', () => this.treeVisualizer.zoomOut());
        this.btnZoomReset.addEventListener('click', () => this.treeVisualizer.resetView());

        // Event listeners - Sliders
        this._setupSliders();

        // Provider change handler
        this.providerSelect.addEventListener('change', (e) => {
            const providerId = e.target.value;
            if (providerId && typeof app !== 'undefined') {
                // Restore saved model for this provider
                const savedModel = app.getSavedModel(providerId);
                this.modelInput.value = savedModel;
            }
        });

        // Model change handler - save model per provider
        this.modelInput.addEventListener('input', (e) => {
            const providerId = this.providerSelect.value;
            if (providerId && e.target.value && typeof app !== 'undefined') {
                app.setSavedModel(providerId, e.target.value);
            }
            this._updateCodeSnippet();
        });

        // Code section - toggle collapse
        if (this.codeSectionHeader) {
            this.codeSectionHeader.addEventListener('click', () => this._toggleCodeSection());
        }

        // Code section - copy button
        if (this.btnCopyCode) {
            this.btnCopyCode.addEventListener('click', () => this._copyCode());
        }

        // Update code snippet when params change
        this.providerSelect.addEventListener('change', () => this._updateCodeSnippet());
        this.promptInput.addEventListener('input', () => this._updateCodeSnippet());

        // Initial code snippet
        this._updateCodeSnippet();
    }

    _setupSliders() {
        const sliders = [
            { slider: this.maxDepthSlider, value: this.maxDepthValue },
            { slider: this.ideasPerLevelSlider, value: this.ideasPerLevelValue },
            { slider: this.topNSlider, value: this.topNValue },
            { slider: this.minQualitySlider, value: this.minQualityValue }
        ];

        sliders.forEach(({ slider, value }) => {
            if (slider && value) {
                slider.addEventListener('input', () => {
                    value.textContent = slider.value;
                    this._updateCodeSnippet();
                });
            }
        });
    }

    _populateProviders() {
        if (!this.providerSelect) return;
        if (typeof app === 'undefined' || !app.providers) return;

        const providers = app.providers || [];
        this.providerSelect.innerHTML = '<option value="">Select provider...</option>';

        providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            const status = provider.configured ? '' : ' (Not configured)';
            option.textContent = provider.name + status;
            this.providerSelect.appendChild(option);
        });

        // Auto-select first configured provider
        const configured = providers.find(p => p.configured);
        if (configured) {
            this.providerSelect.value = configured.id;
            if (typeof app !== 'undefined' && app.getSavedModel) {
                const savedModel = app.getSavedModel(configured.id);
                if (savedModel) {
                    this.modelInput.value = savedModel;
                }
            }
        }
    }

    async startBrainstorm() {
        // Validation
        const prompt = this.promptInput.value.trim();
        if (!prompt) {
            app.showToast('Please enter a brainstorming prompt', 'warning');
            return;
        }

        const provider = this.providerSelect.value;
        const model = this.modelInput.value.trim();

        if (!provider) {
            app.showToast('Please select a provider', 'warning');
            return;
        }

        if (!model) {
            app.showToast('Please enter a model name', 'warning');
            return;
        }

        if (!app.hasApiKey(provider) && provider !== 'ollama') {
            app.showToast(`Please configure ${provider} API key first`, 'warning');
            settingsManager.openPanel();
            return;
        }

        // Get parameters
        const params = {
            max_depth: parseInt(this.maxDepthSlider.value),
            ideas_per_level: parseInt(this.ideasPerLevelSlider.value),
            top_n: parseInt(this.topNSlider.value),
            min_quality_threshold: parseFloat(this.minQualitySlider.value)
        };

        // Clear previous results
        this.treeVisualizer.clear();
        this.hideStats();
        this.hideIdeaDetail();
        this.ideasCount = 0;
        this.currentDepth = 0;
        this.result = null;

        // Show status section
        this.showStatus();
        this.setRunning(true);

        // Start streaming brainstorm
        this.streamController = api.streamBrainstorm(
            prompt,
            provider,
            model,
            params,
            (event) => this.handleStreamEvent(event),
            (error) => this.handleStreamError(error),
            (result) => this.handleComplete(result)
        );
    }

    handleStreamEvent(event) {
        switch (event.type) {
            case 'start':
                this.startTime = Date.now();
                this.startTimer();
                break;

            case 'iteration_start':
                this.currentDepth = event.depth;
                this.updateProgress();
                break;

            case 'idea':
                this.treeVisualizer.addNode(event.idea);
                this.ideasCount++;
                this.currentDepth = Math.max(this.currentDepth, event.idea.depth);
                this.updateProgress();
                break;

            case 'iteration_complete':
                // Could add visual feedback here
                break;
        }
    }

    handleComplete(data) {
        this.setRunning(false);
        this.result = data.result;
        this.stopTimer();
        this.hideStatus();
        this.showStats(data.result);
        app.showToast('Brainstorm complete!', 'success');
    }

    handleStreamError(error) {
        console.error('Stream error:', error);
        this.setRunning(false);
        this.stopTimer();
        this.hideStatus();
        app.showToast(`Error: ${error.message}`, 'error');
    }

    cancelBrainstorm() {
        if (this.streamController) {
            this.streamController.abort();
            this.streamController = null;
        }

        this.setRunning(false);
        this.stopTimer();
        this.hideStatus();
        app.showToast('Brainstorm cancelled', 'warning');
    }

    clearBrainstorm() {
        if (this.isRunning) {
            this.cancelBrainstorm();
        }

        this.treeVisualizer.clear();
        this.promptInput.value = '';
        this.hideStatus();
        this.hideStats();
        this.hideIdeaDetail();
        this.result = null;
        this.ideasCount = 0;
        this.currentDepth = 0;
    }

    setRunning(running) {
        this.isRunning = running;
        this.btnStart.disabled = running;
        this.btnCancel.style.display = running ? 'inline-flex' : 'none';

        const btnText = document.getElementById('brainstorm-btn-text');
        if (btnText) {
            btnText.textContent = running ? 'Running...' : 'Start Brainstorm';
        }
    }

    showStatus() {
        this.statusSection.classList.remove('hidden');
        this.statusBadge.textContent = 'Running...';
        this.statusBadge.className = 'badge badge--warning';
    }

    hideStatus() {
        this.statusSection.classList.add('hidden');
    }

    updateProgress() {
        if (this.currentDepthEl) {
            this.currentDepthEl.textContent = this.currentDepth;
        }
        if (this.ideasCountEl) {
            this.ideasCountEl.textContent = this.ideasCount;
        }

        // Update progress bar (estimate based on depth)
        const maxDepth = parseInt(this.maxDepthSlider.value);
        const progress = Math.min(100, ((this.currentDepth + 1) / maxDepth) * 100);
        if (this.progressBar) {
            this.progressBar.style.width = `${progress}%`;
        }
    }

    startTimer() {
        this.timerInterval = setInterval(() => {
            if (this.startTime && this.elapsedTimeEl) {
                const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
                this.elapsedTimeEl.textContent = `${elapsed}s`;
            }
        }, 1000);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    showStats(result) {
        this.statsSection.classList.remove('hidden');

        if (this.statTotalIdeas) {
            this.statTotalIdeas.textContent = result.total_ideas || 0;
        }

        if (this.statBestScore && result.overall_best_idea) {
            this.statBestScore.textContent = result.overall_best_idea.quality_score.toFixed(1);
        }

        if (this.statAvgScore && result.all_ideas) {
            const scores = result.all_ideas.map(i => i.quality_score);
            const avg = scores.length > 0
                ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1)
                : '0';
            this.statAvgScore.textContent = avg;
        }

        if (this.statExecTime) {
            this.statExecTime.textContent = `${result.execution_time.toFixed(1)}s`;
        }
    }

    hideStats() {
        this.statsSection.classList.add('hidden');
    }

    showIdeaDetail(idea) {
        if (!idea) return;

        this.ideaDetail.classList.remove('hidden');

        if (this.detailScore) {
            this.detailScore.textContent = idea.quality_score.toFixed(1);

            // Style score badge by quality
            this.detailScore.className = 'idea-score-badge';
            if (idea.quality_score >= 8) {
                this.detailScore.classList.add('score-high');
            } else if (idea.quality_score >= 5) {
                this.detailScore.classList.add('score-mid');
            } else {
                this.detailScore.classList.add('score-low');
            }
        }

        if (this.detailText) {
            this.detailText.textContent = idea.text;
        }

        if (this.detailReasoning) {
            this.detailReasoning.textContent = idea.reasoning || 'No reasoning provided';
        }

        if (this.detailDepth) {
            this.detailDepth.textContent = idea.depth;
        }

        if (this.detailId) {
            this.detailId.textContent = idea.id;
        }
    }

    hideIdeaDetail() {
        this.ideaDetail.classList.add('hidden');
    }

    exportResults() {
        if (!this.result) {
            app.showToast('No results to export', 'warning');
            return;
        }

        try {
            // Create download
            const data = JSON.stringify(this.result, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `brainstorm_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);

            app.showToast('Results exported!', 'success');
        } catch (error) {
            console.error('Export error:', error);
            app.showToast('Export failed', 'error');
        }
    }

    // Called when navigating to brainstorm view
    refresh() {
        this._populateProviders();
        this._updateCodeSnippet();
    }

    _toggleCodeSection() {
        if (this.codeSectionContent && this.codeToggleIcon) {
            this.codeSectionContent.classList.toggle('hidden');
            this.codeToggleIcon.classList.toggle('collapsed');
        }
    }

    _updateCodeSnippet() {
        if (!this.codeSnippet) return;

        // Get current values
        const provider = this.providerSelect?.value || 'openai';
        const model = this.modelInput?.value || 'gpt-4o';
        const prompt = this.promptInput?.value || 'Your brainstorming topic here';
        const maxDepth = this.maxDepthSlider?.value || 2;
        const ideasPerLevel = this.ideasPerLevelSlider?.value || 5;
        const topN = this.topNSlider?.value || 3;
        const minQuality = this.minQualitySlider?.value || 5;

        // Map provider to enum name
        const providerMap = {
            'openai': 'OPENAI',
            'anthropic': 'ANTHROPIC',
            'gemini': 'GEMINI',
            'deepseek': 'DEEPSEEK',
            'ollama': 'OLLAMA'
        };
        const providerEnum = providerMap[provider] || 'OPENAI';

        // Escape prompt for Python string
        const escapedPrompt = prompt.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');

        // Build code string (avoiding template literal issues with Python f-strings)
        const lines = [
            'from SimplerLLM.language.llm_brainstorm import RecursiveBrainstorm',
            'from SimplerLLM.language.llm import LLM, LLMProvider',
            '',
            '# Create LLM instance',
            `llm = LLM.create(provider=LLMProvider.${providerEnum}, model_name="${model}")`,
            '',
            '# Initialize brainstormer',
            'brainstormer = RecursiveBrainstorm(',
            '    llm=llm,',
            '    mode="hybrid",',
            `    max_depth=${maxDepth},`,
            `    ideas_per_level=${ideasPerLevel},`,
            `    top_n=${topN},`,
            `    min_quality_threshold=${minQuality}`,
            ')',
            '',
            '# Run brainstorming',
            `result = brainstormer.brainstorm("${escapedPrompt}")`,
            '',
            '# Access results',
            'print(f"Total ideas: {result.total_ideas}")',
            'print(f"Best idea: {result.overall_best_idea.text}")',
            'for idea in result.all_ideas:',
            '    print(f"[{idea.quality_score}] {idea.text}")'
        ];
        const code = lines.join('\n');

        this.codeSnippet.textContent = code;
    }

    async _copyCode() {
        if (!this.codeSnippet) return;

        try {
            await navigator.clipboard.writeText(this.codeSnippet.textContent);

            // Visual feedback
            const originalText = this.btnCopyCode.textContent;
            this.btnCopyCode.textContent = 'Copied!';
            setTimeout(() => {
                this.btnCopyCode.textContent = originalText;
            }, 2000);

            if (typeof app !== 'undefined') {
                app.showToast('Code copied to clipboard!', 'success');
            }
        } catch (error) {
            console.error('Copy failed:', error);
            if (typeof app !== 'undefined') {
                app.showToast('Failed to copy code', 'error');
            }
        }
    }
}

// Global instance
let brainstormManager;
