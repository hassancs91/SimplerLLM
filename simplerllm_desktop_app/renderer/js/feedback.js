/**
 * Feedback Manager
 * Handles LLM Feedback loop UI and interactions
 */

class FeedbackManager {
    constructor() {
        // State
        this.architecture = 'single';
        this.generatorConfig = { provider: '', model: '' };
        this.criticConfig = { provider: '', model: '' };
        this.providersConfig = [];
        this.selectedCriteria = ['accuracy', 'clarity', 'completeness'];
        this.maxIterations = 3;
        this.convergenceThreshold = 0.1;
        this.qualityThreshold = null;
        this.initialAnswer = null;
        this.iterations = [];
        this.result = null;
        this.isRunning = false;
        this.streamController = null;
        this.startTime = null;
        this.timerInterval = null;

        // Line chart
        this.lineChart = null;

        // DOM elements
        this.promptInput = document.getElementById('feedback-prompt');
        this.initialAnswerInput = document.getElementById('feedback-initial-answer');
        this.initialAnswerSection = document.getElementById('feedback-initial-answer-section');
        this.toggleInitialAnswer = document.getElementById('toggle-initial-answer');

        // Architecture
        this.architectureToggle = document.getElementById('feedback-architecture-toggle');
        this.architectureDescription = document.getElementById('architecture-description');
        this.singleConfig = document.getElementById('feedback-single-config');
        this.dualConfig = document.getElementById('feedback-dual-config');
        this.multiConfig = document.getElementById('feedback-multi-config');

        // Architecture descriptions
        this.architectureDescriptions = {
            single: 'Same LLM generates, critiques, and improves its own answers.',
            dual: 'Separate generator and critic LLMs work together for specialized feedback.',
            multi: 'Multiple providers rotate to generate and evaluate answers.'
        };

        // Provider selects
        this.singleProvider = document.getElementById('feedback-single-provider');
        this.singleModel = document.getElementById('feedback-single-model');
        this.generatorProvider = document.getElementById('feedback-generator-provider');
        this.generatorModel = document.getElementById('feedback-generator-model');
        this.criticProvider = document.getElementById('feedback-critic-provider');
        this.criticModel = document.getElementById('feedback-critic-model');
        this.multiProvidersList = document.getElementById('feedback-multi-providers');

        // Parameters
        this.iterationsSlider = document.getElementById('feedback-iterations');
        this.iterationsValue = document.getElementById('feedback-iterations-value');
        this.convergenceSlider = document.getElementById('feedback-convergence');
        this.convergenceValue = document.getElementById('feedback-convergence-value');
        this.qualityThresholdInput = document.getElementById('feedback-quality-threshold');

        // Criteria
        this.criteriaContainer = document.getElementById('feedback-criteria-tags');
        this.customCriterionInput = document.getElementById('feedback-custom-criterion');

        // Actions
        this.startBtn = document.getElementById('btn-start-feedback');
        this.cancelBtn = document.getElementById('btn-cancel-feedback');
        this.clearBtn = document.getElementById('btn-clear-feedback');
        this.addProviderBtn = document.getElementById('btn-add-provider');
        this.backBtn = document.getElementById('btn-back-from-feedback');

        // Status
        this.statusSection = document.getElementById('feedback-status');
        this.statusBadge = document.getElementById('feedback-status-badge');
        this.progressBar = document.getElementById('feedback-progress');
        this.iterationsDone = document.getElementById('iterations-done');
        this.iterationsTotal = document.getElementById('iterations-total');
        this.elapsedTime = document.getElementById('feedback-elapsed-time');

        // Results
        this.chartContainer = document.getElementById('feedback-chart-container');
        this.iterationsContainer = document.getElementById('feedback-iterations-container');
        this.finalAnswerSection = document.getElementById('feedback-final-answer');
        this.finalAnswerContent = document.getElementById('feedback-final-answer-content');
        this.statsSection = document.getElementById('feedback-stats');
        this.statInitialScore = document.getElementById('stat-initial-score');
        this.statFinalScore = document.getElementById('stat-final-score');
        this.statImprovement = document.getElementById('stat-improvement');
        this.statStoppedReason = document.getElementById('stat-stopped-reason');

        // Collapsible sections
        this.chartSectionHeader = document.getElementById('chart-section-header');
        this.chartSectionContent = document.getElementById('chart-section-content');
        this.iterationsSectionHeader = document.getElementById('iterations-section-header');
        this.iterationsSectionContent = document.getElementById('iterations-section-content');
        this.finalAnswerHeader = document.getElementById('final-answer-header');
        this.finalAnswerContentWrapper = document.getElementById('final-answer-content-wrapper');

        // Code section
        this.codeSection = document.getElementById('feedback-code-section');
        this.codeSnippet = document.getElementById('feedback-code-snippet');
        this.copyCodeBtn = document.getElementById('btn-copy-feedback-code');
        this.toggleCodeBtn = document.getElementById('toggle-feedback-code');
        this.codeContent = document.getElementById('feedback-code-content');
        this.codeToggleIcon = document.getElementById('feedback-code-toggle-icon');

        // Provider colors
        this.providerColors = {
            'OPENAI': '#10A37F',
            'ANTHROPIC': '#D97757',
            'GEMINI': '#4285F4',
            'GOOGLE': '#4285F4',
            'OLLAMA': '#1A1A2E',
            'DEEPSEEK': '#6366F1'
        };

        this.init();
    }

    init() {
        // Initialize line chart
        if (this.chartContainer) {
            this.lineChart = new LineChart('feedback-chart-container');
        }

        // Setup event listeners
        this._setupEventListeners();

        // Initialize provider dropdowns
        this._populateProviderDropdowns();

        // Add initial multi-provider rows
        this._addMultiProviderRow();
        this._addMultiProviderRow();

        // Update code snippet
        this._updateCodeSnippet();
    }

    _setupEventListeners() {
        // Back button
        if (this.backBtn) {
            this.backBtn.addEventListener('click', () => {
                if (typeof app !== 'undefined') {
                    app.navigateTo('tools-view');
                }
            });
        }

        // Toggle initial answer section
        if (this.toggleInitialAnswer) {
            this.toggleInitialAnswer.addEventListener('click', () => {
                const isHidden = this.initialAnswerSection.classList.contains('hidden');
                this.initialAnswerSection.classList.toggle('hidden');
                this.toggleInitialAnswer.textContent = isHidden ? '- Hide Initial Answer' : '+ Add Initial Answer';
            });
        }

        // Architecture toggle
        if (this.architectureToggle) {
            this.architectureToggle.addEventListener('click', (e) => {
                if (e.target.classList.contains('mode-btn')) {
                    this._selectArchitecture(e.target.dataset.architecture);
                }
            });
        }

        // Parameter sliders
        if (this.iterationsSlider) {
            this.iterationsSlider.addEventListener('input', (e) => {
                this.maxIterations = parseInt(e.target.value);
                if (this.iterationsValue) {
                    this.iterationsValue.textContent = this.maxIterations;
                }
                this._updateCodeSnippet();
            });
        }

        if (this.convergenceSlider) {
            this.convergenceSlider.addEventListener('input', (e) => {
                this.convergenceThreshold = parseFloat(e.target.value);
                if (this.convergenceValue) {
                    this.convergenceValue.textContent = (this.convergenceThreshold * 100).toFixed(0) + '%';
                }
                this._updateCodeSnippet();
            });
        }

        if (this.qualityThresholdInput) {
            this.qualityThresholdInput.addEventListener('input', (e) => {
                const value = e.target.value.trim();
                this.qualityThreshold = value ? parseFloat(value) : null;
                if (this.lineChart) {
                    this.lineChart.setThreshold(this.qualityThreshold);
                    this.lineChart.render();
                }
                this._updateCodeSnippet();
            });
        }

        // Criteria tags
        if (this.criteriaContainer) {
            this.criteriaContainer.addEventListener('click', (e) => {
                if (e.target.classList.contains('criteria-tag')) {
                    e.target.classList.toggle('active');
                    this._updateSelectedCriteria();
                    this._updateCodeSnippet();
                }
            });
        }

        // Custom criterion input
        if (this.customCriterionInput) {
            this.customCriterionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    const value = e.target.value.trim().toLowerCase();
                    if (value && !this.selectedCriteria.includes(value)) {
                        this._addCustomCriterion(value);
                        e.target.value = '';
                    }
                }
            });
        }

        // Action buttons
        if (this.startBtn) {
            this.startBtn.addEventListener('click', () => this.startFeedback());
        }

        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.cancelFeedback());
        }

        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => this.clearAll());
        }

        if (this.addProviderBtn) {
            this.addProviderBtn.addEventListener('click', () => this._addMultiProviderRow());
        }

        // Code section toggle
        if (this.toggleCodeBtn) {
            this.toggleCodeBtn.addEventListener('click', () => {
                this.codeContent.classList.toggle('hidden');
                if (this.codeToggleIcon) {
                    this.codeToggleIcon.classList.toggle('collapsed');
                }
            });
        }

        // Copy code button
        if (this.copyCodeBtn) {
            this.copyCodeBtn.addEventListener('click', () => this._copyCode());
        }

        // Collapsible sections
        this._setupCollapsibleSection(this.chartSectionHeader, this.chartSectionContent);
        this._setupCollapsibleSection(this.iterationsSectionHeader, this.iterationsSectionContent);
        this._setupCollapsibleSection(this.finalAnswerHeader, this.finalAnswerContentWrapper);

        // Provider/model change handlers
        this._setupProviderChangeHandlers();
    }

    _setupProviderChangeHandlers() {
        const updateConfig = () => {
            if (this.architecture === 'single') {
                this.generatorConfig = {
                    provider: this.singleProvider?.value || '',
                    model: this.singleModel?.value || ''
                };
            } else if (this.architecture === 'dual') {
                this.generatorConfig = {
                    provider: this.generatorProvider?.value || '',
                    model: this.generatorModel?.value || ''
                };
                this.criticConfig = {
                    provider: this.criticProvider?.value || '',
                    model: this.criticModel?.value || ''
                };
            }
            this._updateCodeSnippet();
        };

        [this.singleProvider, this.singleModel,
         this.generatorProvider, this.generatorModel,
         this.criticProvider, this.criticModel].forEach(el => {
            if (el) el.addEventListener('change', updateConfig);
            if (el) el.addEventListener('input', updateConfig);
        });
    }

    _populateProviderDropdowns() {
        const providers = typeof app !== 'undefined' ? app.providers : [];
        const dropdowns = [
            this.singleProvider,
            this.generatorProvider,
            this.criticProvider
        ];

        dropdowns.forEach(dropdown => {
            if (!dropdown) return;
            dropdown.innerHTML = '<option value="">Select provider...</option>';
            providers.forEach(p => {
                const option = document.createElement('option');
                option.value = p.id;
                option.textContent = p.name;
                dropdown.appendChild(option);
            });
        });
    }

    _refreshMultiProviderDropdowns() {
        if (!this.multiProvidersList) return;
        const providers = typeof app !== 'undefined' ? app.providers : [];

        this.multiProvidersList.querySelectorAll('.multi-provider-select').forEach(select => {
            const currentValue = select.value;
            select.innerHTML = '<option value="">Provider...</option>';
            providers.forEach(p => {
                const option = document.createElement('option');
                option.value = p.id;
                option.textContent = p.name;
                select.appendChild(option);
            });
            select.value = currentValue;
        });
    }

    _selectArchitecture(arch) {
        this.architecture = arch;

        // Update toggle buttons
        if (this.architectureToggle) {
            this.architectureToggle.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.architecture === arch);
            });
        }

        // Show/hide config sections
        if (this.singleConfig) this.singleConfig.classList.toggle('hidden', arch !== 'single');
        if (this.dualConfig) this.dualConfig.classList.toggle('hidden', arch !== 'dual');
        if (this.multiConfig) this.multiConfig.classList.toggle('hidden', arch !== 'multi');

        // Update description text
        if (this.architectureDescription && this.architectureDescriptions[arch]) {
            this.architectureDescription.textContent = this.architectureDescriptions[arch];
        }

        // Refresh multi-provider dropdowns when switching to multi
        if (arch === 'multi') {
            this._refreshMultiProviderDropdowns();
        }

        this._updateCodeSnippet();
    }

    _addMultiProviderRow() {
        if (!this.multiProvidersList) return;

        const providers = typeof app !== 'undefined' ? app.providers : [];
        const index = this.multiProvidersList.children.length;

        const row = document.createElement('div');
        row.className = 'multi-provider-row';
        row.innerHTML = `
            <select class="select multi-provider-select" data-index="${index}">
                <option value="">Provider...</option>
                ${providers.map(p => `<option value="${p.id}">${p.name}</option>`).join('')}
            </select>
            <input type="text" class="input multi-model-input" data-index="${index}" placeholder="Model name">
            <button class="btn btn--ghost btn--sm btn-remove-provider" data-index="${index}">×</button>
        `;

        // Remove button handler
        row.querySelector('.btn-remove-provider').addEventListener('click', () => {
            if (this.multiProvidersList.children.length > 2) {
                row.remove();
                this._updateMultiProvidersConfig();
            }
        });

        // Change handlers
        row.querySelector('.multi-provider-select').addEventListener('change', () => this._updateMultiProvidersConfig());
        row.querySelector('.multi-model-input').addEventListener('input', () => this._updateMultiProvidersConfig());

        this.multiProvidersList.appendChild(row);
    }

    _updateMultiProvidersConfig() {
        if (!this.multiProvidersList) return;

        this.providersConfig = [];
        this.multiProvidersList.querySelectorAll('.multi-provider-row').forEach(row => {
            const provider = row.querySelector('.multi-provider-select')?.value;
            const model = row.querySelector('.multi-model-input')?.value;
            if (provider && model) {
                this.providersConfig.push({ provider, model });
            }
        });
        this._updateCodeSnippet();
    }

    _updateSelectedCriteria() {
        this.selectedCriteria = [];
        if (this.criteriaContainer) {
            this.criteriaContainer.querySelectorAll('.criteria-tag.active').forEach(tag => {
                this.selectedCriteria.push(tag.dataset.criterion);
            });
        }
    }

    _addCustomCriterion(criterion) {
        if (!this.criteriaContainer) return;

        const tag = document.createElement('span');
        tag.className = 'criteria-tag active custom';
        tag.dataset.criterion = criterion;
        tag.innerHTML = `${criterion} <span class="remove-tag">×</span>`;

        tag.querySelector('.remove-tag').addEventListener('click', (e) => {
            e.stopPropagation();
            tag.remove();
            this._updateSelectedCriteria();
        });

        this.criteriaContainer.appendChild(tag);
        this.selectedCriteria.push(criterion);
        this._updateCodeSnippet();
    }

    refresh() {
        this._populateProviderDropdowns();
        this._refreshMultiProviderDropdowns();
        // Restore slider values from internal state
        if (this.iterationsSlider) {
            this.iterationsSlider.value = this.maxIterations;
        }
        if (this.iterationsValue) {
            this.iterationsValue.textContent = this.maxIterations;
        }
    }

    async startFeedback() {
        // Validate inputs
        const prompt = this.promptInput?.value?.trim();
        if (!prompt) {
            if (typeof app !== 'undefined') {
                app.showToast('Please enter a prompt', 'warning');
            }
            return;
        }

        // Validate provider config based on architecture
        if (this.architecture === 'single') {
            if (!this.singleProvider?.value || !this.singleModel?.value) {
                if (typeof app !== 'undefined') {
                    app.showToast('Please select provider and model', 'warning');
                }
                return;
            }
            this.generatorConfig = {
                provider: this.singleProvider.value,
                model: this.singleModel.value
            };
        } else if (this.architecture === 'dual') {
            if (!this.generatorProvider?.value || !this.generatorModel?.value) {
                if (typeof app !== 'undefined') {
                    app.showToast('Please configure generator LLM', 'warning');
                }
                return;
            }
            if (!this.criticProvider?.value || !this.criticModel?.value) {
                if (typeof app !== 'undefined') {
                    app.showToast('Please configure critic LLM', 'warning');
                }
                return;
            }
            this.generatorConfig = {
                provider: this.generatorProvider.value,
                model: this.generatorModel.value
            };
            this.criticConfig = {
                provider: this.criticProvider.value,
                model: this.criticModel.value
            };
        } else if (this.architecture === 'multi') {
            this._updateMultiProvidersConfig();
            if (this.providersConfig.length < 2) {
                if (typeof app !== 'undefined') {
                    app.showToast('At least 2 providers required for multi mode', 'warning');
                }
                return;
            }
        }

        if (this.selectedCriteria.length === 0) {
            if (typeof app !== 'undefined') {
                app.showToast('Please select at least one criterion', 'warning');
            }
            return;
        }

        // Get initial answer if provided
        this.initialAnswer = this.initialAnswerInput?.value?.trim() || null;

        // Clear previous results
        this._clearResults();
        this._showStatus();
        this._setRunning(true);

        // Update totals
        if (this.iterationsTotal) {
            this.iterationsTotal.textContent = this.maxIterations;
        }
        if (this.iterationsDone) {
            this.iterationsDone.textContent = '0';
        }

        // Build config
        const config = {
            architecture: this.architecture,
            generator: this.architecture !== 'multi' ? this.generatorConfig : null,
            critic: this.architecture === 'dual' ? this.criticConfig : null,
            providers: this.architecture === 'multi' ? this.providersConfig : [],
            max_iterations: this.maxIterations,
            criteria: this.selectedCriteria,
            initial_answer: this.initialAnswer,
            convergence_threshold: this.convergenceThreshold,
            quality_threshold: this.qualityThreshold
        };

        // Start streaming
        this.streamController = api.streamFeedback(
            prompt,
            config,
            (event) => this._handleStreamEvent(event),
            (error) => this._handleStreamError(error),
            (result) => this._handleComplete(result)
        );
    }

    _handleStreamEvent(event) {
        console.log('Feedback event:', event);

        switch (event.type) {
            case 'start':
                this.startTime = Date.now();
                this._startTimer();
                if (this.iterationsTotal) {
                    this.iterationsTotal.textContent = event.max_iterations;
                }
                this._updateStatusMessage(`Starting ${event.architecture} mode...`);
                break;

            case 'llm_ready':
                this._updateStatusMessage(event.message);
                break;

            case 'running':
                this._updateStatusMessage(event.message);
                break;

            case 'iteration_complete':
                this._addIterationResult(event.iteration);
                break;
        }
    }

    _handleComplete(data) {
        this._setRunning(false);
        this.result = data.result;
        this._stopTimer();
        this._hideStatus();

        // Update chart with final trajectory
        if (this.lineChart && this.result.improvement_trajectory) {
            this.lineChart.setData(this.result.improvement_trajectory);
        }

        // Show final answer
        this._showFinalAnswer(this.result.final_answer);

        // Show stats
        this._showStats(this.result);

        if (typeof app !== 'undefined') {
            app.showToast('Improvement complete!', 'success');
        }
    }

    _handleStreamError(error) {
        console.error('Stream error:', error);
        this._setRunning(false);
        this._stopTimer();
        this._hideStatus();
        if (typeof app !== 'undefined') {
            app.showToast(`Error: ${error.message}`, 'error');
        }
    }

    cancelFeedback() {
        if (this.streamController) {
            this.streamController.abort();
            this.streamController = null;
        }
        this._setRunning(false);
        this._stopTimer();
        this._hideStatus();
        if (typeof app !== 'undefined') {
            app.showToast('Feedback cancelled', 'info');
        }
    }

    clearAll() {
        this.cancelFeedback();

        // Clear inputs
        if (this.promptInput) this.promptInput.value = '';
        if (this.initialAnswerInput) this.initialAnswerInput.value = '';

        // Reset state
        this.iterations = [];
        this.result = null;

        // Clear results
        this._clearResults();
        this._hideStatus();
        this._hideStats();
        this._hideFinalAnswer();

        // Clear chart
        if (this.lineChart) {
            this.lineChart.clear();
        }

        if (typeof app !== 'undefined') {
            app.showToast('Cleared', 'info');
        }
    }

    _addIterationResult(iteration) {
        this.iterations.push(iteration);

        // Update progress
        if (this.iterationsDone) {
            this.iterationsDone.textContent = this.iterations.length;
        }
        this._updateProgress();

        // Add point to chart
        if (this.lineChart && iteration.critique) {
            this.lineChart.addPoint(iteration.iteration_number, iteration.critique.quality_score);
        }

        // Add iteration card
        this._addIterationCard(iteration);
    }

    _addIterationCard(iteration) {
        if (!this.iterationsContainer) return;

        const card = document.createElement('div');
        card.className = 'iteration-card';

        const score = iteration.critique?.quality_score || 0;
        const scoreClass = this._getScoreClass(score);
        const providerColor = this._getProviderColor(iteration.provider_used);
        const improvement = iteration.improvement_from_previous;

        card.innerHTML = `
            <div class="iteration-card__header" data-iteration="${iteration.iteration_number}">
                <span class="iteration-number">Iteration ${iteration.iteration_number}</span>
                <span class="score-badge ${scoreClass}">${score.toFixed(1)}</span>
                <span class="provider-badge" style="background: ${providerColor}">${iteration.provider_used}</span>
                ${improvement !== null ? `
                    <span class="improvement-badge ${improvement >= 0 ? 'positive' : 'negative'}">
                        ${improvement >= 0 ? '↑' : '↓'} ${Math.abs(improvement * 100).toFixed(0)}%
                    </span>
                ` : ''}
                <span class="expand-icon collapsed">▼</span>
            </div>
            <div class="iteration-card__body hidden">
                <div class="iteration-answer">
                    <h5>Answer</h5>
                    <p>${this._escapeHtml(iteration.answer)}</p>
                </div>
                ${iteration.critique ? `
                    <div class="iteration-critique">
                        <h5>Critique</h5>
                        <div class="critique-section">
                            <strong>Strengths:</strong>
                            <ul>${iteration.critique.strengths.map(s => `<li>${this._escapeHtml(s)}</li>`).join('')}</ul>
                        </div>
                        <div class="critique-section">
                            <strong>Weaknesses:</strong>
                            <ul>${iteration.critique.weaknesses.map(w => `<li>${this._escapeHtml(w)}</li>`).join('')}</ul>
                        </div>
                        <div class="critique-section">
                            <strong>Suggestions:</strong>
                            <ul>${iteration.critique.improvement_suggestions.map(s => `<li>${this._escapeHtml(s)}</li>`).join('')}</ul>
                        </div>
                        <div class="critique-reasoning">
                            <strong>Reasoning:</strong>
                            <p>${this._escapeHtml(iteration.critique.reasoning)}</p>
                        </div>
                    </div>
                ` : ''}
                <div class="iteration-meta">
                    <span>Time: ${iteration.execution_time?.toFixed(1) || 0}s</span>
                    <span>Model: ${iteration.model_used}</span>
                </div>
            </div>
        `;

        // Toggle expand/collapse
        card.querySelector('.iteration-card__header').addEventListener('click', () => {
            const body = card.querySelector('.iteration-card__body');
            const icon = card.querySelector('.expand-icon');
            body.classList.toggle('hidden');
            icon.classList.toggle('collapsed');
        });

        this.iterationsContainer.appendChild(card);

        // Scroll to bottom
        this.iterationsContainer.scrollTop = this.iterationsContainer.scrollHeight;
    }

    _showFinalAnswer(answer) {
        if (this.finalAnswerSection) {
            this.finalAnswerSection.classList.remove('hidden');
        }
        if (this.finalAnswerContent) {
            this.finalAnswerContent.textContent = answer;
        }
    }

    _hideFinalAnswer() {
        if (this.finalAnswerSection) {
            this.finalAnswerSection.classList.add('hidden');
        }
    }

    _showStats(result) {
        if (this.statsSection) {
            this.statsSection.classList.remove('hidden');
        }

        if (this.statInitialScore) {
            this.statInitialScore.textContent = result.initial_score?.toFixed(1) || '-';
        }

        if (this.statFinalScore) {
            this.statFinalScore.textContent = result.final_score?.toFixed(1) || '-';
        }

        if (this.statImprovement) {
            const improvement = result.final_score - result.initial_score;
            const sign = improvement >= 0 ? '+' : '';
            this.statImprovement.textContent = `${sign}${improvement.toFixed(1)}`;
            this.statImprovement.className = improvement >= 0 ? 'stat-value positive' : 'stat-value negative';
        }

        if (this.statStoppedReason) {
            const reasons = {
                'max_iterations': 'Max iterations',
                'converged': 'Converged',
                'threshold_met': 'Threshold met'
            };
            this.statStoppedReason.textContent = reasons[result.stopped_reason] || result.stopped_reason;
        }
    }

    _hideStats() {
        if (this.statsSection) {
            this.statsSection.classList.add('hidden');
        }
    }

    _clearResults() {
        this.iterations = [];
        if (this.iterationsContainer) {
            this.iterationsContainer.innerHTML = '';
        }
    }

    _setRunning(running) {
        this.isRunning = running;
        if (this.startBtn) {
            this.startBtn.disabled = running;
            const btnText = this.startBtn.querySelector('#feedback-btn-text');
            if (btnText) {
                btnText.textContent = running ? 'Running...' : 'Start Improvement';
            }
        }
        if (this.cancelBtn) {
            this.cancelBtn.style.display = running ? 'inline-flex' : 'none';
        }
    }

    _showStatus() {
        if (this.statusSection) {
            this.statusSection.classList.remove('hidden');
        }
    }

    _hideStatus() {
        if (this.statusSection) {
            this.statusSection.classList.add('hidden');
        }
    }

    _updateStatusMessage(message) {
        if (this.statusBadge) {
            this.statusBadge.textContent = message;
        }
    }

    _updateProgress() {
        if (this.progressBar) {
            const progress = (this.iterations.length / this.maxIterations) * 100;
            this.progressBar.style.width = `${progress}%`;
        }
    }

    _startTimer() {
        this.startTime = Date.now();
        this.timerInterval = setInterval(() => {
            if (this.elapsedTime) {
                const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
                this.elapsedTime.textContent = `${elapsed}s`;
            }
        }, 1000);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    _getScoreClass(score) {
        if (score === undefined || score === null) return '';
        if (score >= 8) return 'score-high';
        if (score >= 5) return 'score-mid';
        return 'score-low';
    }

    _getProviderColor(providerName) {
        return this.providerColors[providerName?.toUpperCase()] || '#888888';
    }

    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    _updateCodeSnippet() {
        if (!this.codeSnippet) return;

        const prompt = this.promptInput?.value?.trim() || 'Your prompt here';
        const criteria = this.selectedCriteria.length > 0
            ? `["${this.selectedCriteria.join('", "')}"]`
            : '["accuracy", "clarity", "completeness"]';

        let code = '';

        if (this.architecture === 'single') {
            const provider = this.singleProvider?.value?.toUpperCase() || 'OPENAI';
            const model = this.singleModel?.value || 'gpt-4o';

            code = `from SimplerLLM.language import LLM, LLMProvider, LLMFeedbackLoop

# Create LLM instance
llm = LLM.create(LLMProvider.${provider}, model_name="${model}")

# Create feedback loop (self-critique mode)
feedback = LLMFeedbackLoop(
    llm=llm,
    max_iterations=${this.maxIterations},
    convergence_threshold=${this.convergenceThreshold},
    ${this.qualityThreshold ? `quality_threshold=${this.qualityThreshold},` : ''}
    default_criteria=${criteria}
)

# Run improvement loop
result = feedback.improve("${prompt.replace(/"/g, '\\"')}")

# Results
print(f"Initial Score: {result.initial_score}")
print(f"Final Score: {result.final_score}")
print(f"Stopped: {result.stopped_reason}")
print(f"Final Answer: {result.final_answer}")`;

        } else if (this.architecture === 'dual') {
            const genProvider = this.generatorProvider?.value?.toUpperCase() || 'OPENAI';
            const genModel = this.generatorModel?.value || 'gpt-4o';
            const critProvider = this.criticProvider?.value?.toUpperCase() || 'ANTHROPIC';
            const critModel = this.criticModel?.value || 'claude-3-5-sonnet-20241022';

            code = `from SimplerLLM.language import LLM, LLMProvider, LLMFeedbackLoop

# Create generator and critic LLMs
generator = LLM.create(LLMProvider.${genProvider}, model_name="${genModel}")
critic = LLM.create(LLMProvider.${critProvider}, model_name="${critModel}")

# Create feedback loop (dual mode)
feedback = LLMFeedbackLoop(
    generator_llm=generator,
    critic_llm=critic,
    max_iterations=${this.maxIterations},
    convergence_threshold=${this.convergenceThreshold},
    ${this.qualityThreshold ? `quality_threshold=${this.qualityThreshold},` : ''}
    default_criteria=${criteria}
)

# Run improvement loop
result = feedback.improve("${prompt.replace(/"/g, '\\"')}")

# Results
print(f"Initial Score: {result.initial_score}")
print(f"Final Score: {result.final_score}")
print(f"Architecture: {result.architecture_used}")
print(f"Final Answer: {result.final_answer}")`;

        } else if (this.architecture === 'multi') {
            const providers = this.providersConfig.length > 0
                ? this.providersConfig
                : [{ provider: 'openai', model: 'gpt-4o' }, { provider: 'anthropic', model: 'claude-3-5-sonnet-20241022' }];

            const providerLines = providers.map(p =>
                `    LLM.create(LLMProvider.${p.provider.toUpperCase()}, model_name="${p.model}")`
            ).join(',\n');

            code = `from SimplerLLM.language import LLM, LLMProvider, LLMFeedbackLoop

# Create multiple LLM providers
providers = [
${providerLines}
]

# Create feedback loop (multi-provider rotation)
feedback = LLMFeedbackLoop(
    providers=providers,
    max_iterations=${this.maxIterations},
    convergence_threshold=${this.convergenceThreshold},
    ${this.qualityThreshold ? `quality_threshold=${this.qualityThreshold},` : ''}
    default_criteria=${criteria}
)

# Run improvement loop
result = feedback.improve("${prompt.replace(/"/g, '\\"')}")

# Results
print(f"Initial Score: {result.initial_score}")
print(f"Final Score: {result.final_score}")
print(f"Total Iterations: {result.total_iterations}")
for iteration in result.all_iterations:
    print(f"  Iteration {iteration.iteration_number}: {iteration.critique.quality_score}/10 by {iteration.provider_used}")`;
        }

        this.codeSnippet.textContent = code;
    }

    _copyCode() {
        if (!this.codeSnippet) return;

        try {
            navigator.clipboard.writeText(this.codeSnippet.textContent);
            if (typeof app !== 'undefined') {
                app.showToast('Code copied to clipboard', 'success');
            }
        } catch (e) {
            console.error('Failed to copy:', e);
            if (typeof app !== 'undefined') {
                app.showToast('Failed to copy code', 'error');
            }
        }
    }

    _setupCollapsibleSection(header, content) {
        if (!header || !content) return;
        header.addEventListener('click', () => {
            const icon = header.querySelector('.collapse-icon');
            content.classList.toggle('collapsed');
            if (icon) icon.classList.toggle('collapsed');
        });
    }
}

// Global instance
let feedbackManager;
