/**
 * Judge Manager
 * Handles LLM Judge evaluations with radar chart visualization
 */
class JudgeManager {
    constructor() {
        // DOM elements - Input
        this.promptInput = document.getElementById('judge-prompt');

        // DOM elements - Mode
        this.modeToggle = document.getElementById('judge-mode-toggle');
        this.modeDescription = document.getElementById('mode-description');

        // DOM elements - Contestants
        this.contestantsList = document.getElementById('contestants-list');
        this.btnAddContestant = document.getElementById('btn-add-contestant');

        // DOM elements - Judge config
        this.judgeProvider = document.getElementById('judge-provider');
        this.judgeModel = document.getElementById('judge-model');

        // DOM elements - Criteria
        this.criteriaTags = document.getElementById('criteria-tags');
        this.customCriterionInput = document.getElementById('custom-criterion');

        // DOM elements - Actions
        this.btnStart = document.getElementById('btn-start-judge');
        this.btnCancel = document.getElementById('btn-cancel-judge');
        this.btnClear = document.getElementById('btn-clear-judge');
        this.btnBack = document.getElementById('btn-back-from-judge');
        this.btnExport = document.getElementById('btn-export-judge');

        // DOM elements - Status
        this.statusSection = document.getElementById('judge-status');
        this.statusBadge = document.getElementById('judge-status-badge');
        this.progressBar = document.getElementById('judge-progress');
        this.contestantsDone = document.getElementById('contestants-done');
        this.contestantsTotal = document.getElementById('contestants-total');
        this.elapsedTimeEl = document.getElementById('judge-elapsed-time');

        // DOM elements - Stats
        this.statsSection = document.getElementById('judge-stats');
        this.statWinner = document.getElementById('stat-winner');
        this.statBestScore = document.getElementById('stat-best-score');
        this.statTotalTime = document.getElementById('stat-total-time');

        // DOM elements - Visualization
        this.radarContainer = document.getElementById('judge-radar-container');
        this.responsesContainer = document.getElementById('judge-responses-container');
        this.finalAnswerSection = document.getElementById('judge-final-answer');
        this.finalAnswerContent = document.getElementById('final-answer-content');

        // DOM elements - Code section
        this.codeSection = document.getElementById('judge-code-section');
        this.codeSectionHeader = document.getElementById('toggle-judge-code');
        this.codeSectionContent = document.getElementById('judge-code-content');
        this.codeToggleIcon = document.getElementById('judge-code-toggle-icon');
        this.codeSnippet = document.getElementById('judge-code-snippet');
        this.btnCopyCode = document.getElementById('btn-copy-judge-code');

        // State
        this.streamController = null;
        this.isRunning = false;
        this.startTime = null;
        this.timerInterval = null;
        this.result = null;
        this.currentMode = 'select_best';
        this.contestants = [];
        this.selectedCriteria = ['accuracy', 'clarity', 'completeness'];
        this.contestantsCompleted = 0;

        // Radar chart
        this.radarChart = null;

        // Provider colors
        this.providerColors = {
            'OPENAI': '#00D4AA',
            'ANTHROPIC': '#FF6B9D',
            'GEMINI': '#FFE156',
            'DEEPSEEK': '#9B59B6',
            'OLLAMA': '#3498DB',
            'COHERE': '#E67E22',
            'PERPLEXITY': '#1ABC9C'
        };

        // Mode descriptions
        this.modeDescriptions = {
            'select_best': 'Pick the single best response from all contestants.',
            'synthesize': 'Combine the best elements from all responses into an improved answer.',
            'compare': 'Get a detailed comparative analysis of all responses.'
        };

        this.init();
    }

    init() {
        // Initialize radar chart
        this.radarChart = new RadarChart('judge-radar-svg', 'radar-legend');

        // Setup event listeners
        this._setupModeToggle();
        this._setupContestants();
        this._setupCriteria();
        this._setupActions();
        this._setupCodeSection();

        // Add initial contestants (2 default)
        this._addContestantRow();
        this._addContestantRow();

        // Initial code snippet
        this._updateCodeSnippet();
    }

    _setupModeToggle() {
        if (!this.modeToggle) return;

        const buttons = this.modeToggle.querySelectorAll('.mode-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentMode = btn.dataset.mode;
                if (this.modeDescription) {
                    this.modeDescription.textContent = this.modeDescriptions[this.currentMode];
                }
                this._updateCodeSnippet();
            });
        });
    }

    _setupContestants() {
        if (this.btnAddContestant) {
            this.btnAddContestant.addEventListener('click', () => this._addContestantRow());
        }
    }

    _addContestantRow() {
        if (!this.contestantsList) return;

        const index = this.contestantsList.children.length;
        const row = document.createElement('div');
        row.className = 'contestant-row';
        row.dataset.index = index;

        row.innerHTML = `
            <div class="contestant-number">${index + 1}</div>
            <div class="contestant-inputs">
                <select class="select contestant-provider">
                    <option value="">Provider...</option>
                </select>
                <input type="text" class="input contestant-model" placeholder="Model name">
            </div>
            <button class="btn btn--ghost btn--sm btn-remove-contestant" title="Remove">
                <span>×</span>
            </button>
        `;

        // Populate provider select
        const providerSelect = row.querySelector('.contestant-provider');
        this._populateProviderSelect(providerSelect);

        // Model change handler
        const modelInput = row.querySelector('.contestant-model');
        providerSelect.addEventListener('change', (e) => {
            const providerId = e.target.value;
            if (providerId && typeof app !== 'undefined') {
                const savedModel = app.getSavedModel(providerId);
                modelInput.value = savedModel;
            }
            this._updateCodeSnippet();
        });

        modelInput.addEventListener('input', (e) => {
            const providerId = providerSelect.value;
            if (providerId && e.target.value && typeof app !== 'undefined') {
                app.setSavedModel(providerId, e.target.value);
            }
            this._updateCodeSnippet();
        });

        // Remove button
        const removeBtn = row.querySelector('.btn-remove-contestant');
        removeBtn.addEventListener('click', () => {
            if (this.contestantsList.children.length > 2) {
                row.remove();
                this._renumberContestants();
                this._updateCodeSnippet();
            } else {
                app.showToast('Minimum 2 contestants required', 'warning');
            }
        });

        this.contestantsList.appendChild(row);
        this._updateCodeSnippet();
    }

    _renumberContestants() {
        const rows = this.contestantsList.querySelectorAll('.contestant-row');
        rows.forEach((row, idx) => {
            row.dataset.index = idx;
            const number = row.querySelector('.contestant-number');
            if (number) number.textContent = idx + 1;
        });
    }

    _populateProviderSelect(select) {
        if (!select) return;
        if (typeof app === 'undefined' || !app.providers) return;

        const providers = app.providers || [];
        select.innerHTML = '<option value="">Provider...</option>';

        providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            const status = provider.configured ? '' : ' (Not configured)';
            option.textContent = provider.name + status;
            select.appendChild(option);
        });
    }

    _getContestantsConfig() {
        const contestants = [];
        if (!this.contestantsList) return contestants;

        const rows = this.contestantsList.querySelectorAll('.contestant-row');
        rows.forEach(row => {
            const provider = row.querySelector('.contestant-provider')?.value;
            const model = row.querySelector('.contestant-model')?.value?.trim();
            if (provider && model) {
                contestants.push({ provider, model });
            }
        });

        return contestants;
    }

    _setupCriteria() {
        if (this.criteriaTags) {
            const tags = this.criteriaTags.querySelectorAll('.criteria-tag');
            tags.forEach(tag => {
                tag.addEventListener('click', () => {
                    tag.classList.toggle('active');
                    this._updateSelectedCriteria();
                    this._updateCodeSnippet();
                });
            });
        }

        if (this.customCriterionInput) {
            this.customCriterionInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const criterion = this.customCriterionInput.value.trim().toLowerCase();
                    if (criterion && !this.selectedCriteria.includes(criterion)) {
                        this._addCriterionTag(criterion);
                        this.customCriterionInput.value = '';
                    }
                }
            });
        }
    }

    _updateSelectedCriteria() {
        this.selectedCriteria = [];
        if (!this.criteriaTags) return;

        const activeTags = this.criteriaTags.querySelectorAll('.criteria-tag.active');
        activeTags.forEach(tag => {
            const criterion = tag.dataset.criterion;
            if (criterion) {
                this.selectedCriteria.push(criterion);
            }
        });
    }

    _addCriterionTag(criterion) {
        if (!this.criteriaTags) return;

        const tag = document.createElement('span');
        tag.className = 'criteria-tag active custom';
        tag.dataset.criterion = criterion;
        tag.textContent = criterion;

        tag.addEventListener('click', () => {
            tag.classList.toggle('active');
            this._updateSelectedCriteria();
            this._updateCodeSnippet();
        });

        this.criteriaTags.appendChild(tag);
        this.selectedCriteria.push(criterion);
        this._updateCodeSnippet();
    }

    _setupActions() {
        if (this.btnStart) {
            this.btnStart.addEventListener('click', () => this.startJudge());
        }
        if (this.btnCancel) {
            this.btnCancel.addEventListener('click', () => this.cancelJudge());
        }
        if (this.btnClear) {
            this.btnClear.addEventListener('click', () => this.clearJudge());
        }
        if (this.btnBack) {
            this.btnBack.addEventListener('click', () => app.navigateTo('tools-view'));
        }
        if (this.btnExport) {
            this.btnExport.addEventListener('click', () => this.exportResults());
        }

        // Judge provider change handler
        if (this.judgeProvider) {
            this.judgeProvider.addEventListener('change', (e) => {
                const providerId = e.target.value;
                if (providerId && typeof app !== 'undefined') {
                    const savedModel = app.getSavedModel(providerId);
                    if (this.judgeModel) {
                        this.judgeModel.value = savedModel;
                    }
                }
                this._updateCodeSnippet();
            });
        }

        if (this.judgeModel) {
            this.judgeModel.addEventListener('input', (e) => {
                const providerId = this.judgeProvider?.value;
                if (providerId && e.target.value && typeof app !== 'undefined') {
                    app.setSavedModel(providerId, e.target.value);
                }
                this._updateCodeSnippet();
            });
        }

        // Prompt change
        if (this.promptInput) {
            this.promptInput.addEventListener('input', () => this._updateCodeSnippet());
        }
    }

    _setupCodeSection() {
        if (this.codeSectionHeader) {
            this.codeSectionHeader.addEventListener('click', () => this._toggleCodeSection());
        }

        if (this.btnCopyCode) {
            this.btnCopyCode.addEventListener('click', () => this._copyCode());
        }
    }

    async startJudge() {
        // Validation
        const prompt = this.promptInput?.value?.trim();
        if (!prompt) {
            app.showToast('Please enter a prompt', 'warning');
            return;
        }

        const contestants = this._getContestantsConfig();
        if (contestants.length < 2) {
            app.showToast('Please configure at least 2 contestants', 'warning');
            return;
        }

        const judgeConfig = {
            provider: this.judgeProvider?.value,
            model: this.judgeModel?.value?.trim()
        };

        if (!judgeConfig.provider || !judgeConfig.model) {
            app.showToast('Please configure the judge LLM', 'warning');
            return;
        }

        // Validate API keys for all providers
        const allProviders = [...contestants.map(c => c.provider), judgeConfig.provider];
        for (const provider of allProviders) {
            if (!app.hasApiKey(provider) && provider !== 'ollama') {
                app.showToast(`Please configure ${provider} API key first`, 'warning');
                settingsManager.openPanel();
                return;
            }
        }

        this._updateSelectedCriteria();
        if (this.selectedCriteria.length === 0) {
            app.showToast('Please select at least one evaluation criterion', 'warning');
            return;
        }

        // Clear previous results
        this.radarChart.clear();
        this._clearResponseCards();
        this._hideFinalAnswer();
        this._hideStats();
        this.contestantsCompleted = 0;
        this.result = null;

        // Show status
        this._showStatus();
        this._setRunning(true);

        if (this.contestantsTotal) {
            this.contestantsTotal.textContent = contestants.length;
        }
        if (this.contestantsDone) {
            this.contestantsDone.textContent = '0';
        }

        // Start streaming
        this.streamController = api.streamJudge(
            prompt,
            contestants,
            judgeConfig,
            this.currentMode,
            this.selectedCriteria,
            (event) => this._handleStreamEvent(event),
            (error) => this._handleStreamError(error),
            (result) => this._handleComplete(result)
        );
    }

    _handleStreamEvent(event) {
        switch (event.type) {
            case 'start':
                this.startTime = Date.now();
                this._startTimer();
                if (this.contestantsTotal) {
                    this.contestantsTotal.textContent = event.total_contestants;
                }
                break;

            case 'contestant_start':
                this._updateStatusMessage(`Initializing ${event.provider}...`);
                break;

            case 'contestant_ready':
                this._updateStatusMessage(`${event.provider} ready`);
                break;

            case 'contestants_running':
                this._updateStatusMessage('All contestants generating responses...');
                break;

            case 'contestant_complete':
                this.contestantsCompleted++;
                if (this.contestantsDone) {
                    this.contestantsDone.textContent = this.contestantsCompleted;
                }
                this._updateProgress();
                break;

            case 'judging_complete':
                this._updateStatusMessage('Evaluation complete!');
                break;
        }
    }

    _handleComplete(data) {
        this._setRunning(false);
        this.result = data.result;
        this._stopTimer();
        this._hideStatus();

        // Update visualizations
        this._renderRadarChart(data.result.evaluations, data.result.criteria_used);
        this._renderResponseCards(data.result.all_responses, data.result.evaluations);
        this._showFinalAnswer(data.result.final_answer, data.result.mode);
        this._showStats(data.result);

        app.showToast('Evaluation complete!', 'success');
    }

    _handleStreamError(error) {
        console.error('Stream error:', error);
        this._setRunning(false);
        this._stopTimer();
        this._hideStatus();
        app.showToast(`Error: ${error.message}`, 'error');
    }

    cancelJudge() {
        if (this.streamController) {
            this.streamController.abort();
            this.streamController = null;
        }

        this._setRunning(false);
        this._stopTimer();
        this._hideStatus();
        app.showToast('Evaluation cancelled', 'warning');
    }

    clearJudge() {
        if (this.isRunning) {
            this.cancelJudge();
        }

        this.radarChart.clear();
        this._clearResponseCards();
        if (this.promptInput) this.promptInput.value = '';
        this._hideStatus();
        this._hideStats();
        this._hideFinalAnswer();
        this.result = null;
        this.contestantsCompleted = 0;
    }

    _setRunning(running) {
        this.isRunning = running;
        if (this.btnStart) this.btnStart.disabled = running;
        if (this.btnCancel) {
            this.btnCancel.style.display = running ? 'inline-flex' : 'none';
        }

        const btnText = document.getElementById('judge-btn-text');
        if (btnText) {
            btnText.textContent = running ? 'Evaluating...' : 'Start Evaluation';
        }
    }

    _showStatus() {
        if (this.statusSection) {
            this.statusSection.classList.remove('hidden');
        }
        if (this.statusBadge) {
            this.statusBadge.textContent = 'Running...';
            this.statusBadge.className = 'badge badge--warning';
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
        const total = parseInt(this.contestantsTotal?.textContent || 0);
        if (total > 0 && this.progressBar) {
            const progress = (this.contestantsCompleted / total) * 100;
            this.progressBar.style.width = `${progress}%`;
        }
    }

    _startTimer() {
        this.timerInterval = setInterval(() => {
            if (this.startTime && this.elapsedTimeEl) {
                const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
                this.elapsedTimeEl.textContent = `${elapsed}s`;
            }
        }, 1000);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    _renderRadarChart(evaluations, criteria) {
        if (!evaluations || !criteria || criteria.length < 3) {
            // Need at least 3 criteria for radar chart
            return;
        }

        const datasets = evaluations.map(ev => ({
            label: ev.provider_name,
            values: criteria.map(c => ev.criterion_scores?.[c] || 0),
            color: this._getProviderColor(ev.provider_name)
        }));

        this.radarChart.render(criteria, datasets);
    }

    _renderResponseCards(responses, evaluations) {
        if (!this.responsesContainer) return;
        this._clearResponseCards();

        // Debug: log data to understand structure
        console.log('Responses:', responses);
        console.log('Evaluations for cards:', evaluations);

        // Create evaluation lookup - normalize provider names to uppercase
        const evalMap = {};
        evaluations.forEach(e => {
            evalMap[e.provider_name] = e;
            evalMap[e.provider_name?.toUpperCase()] = e;
            evalMap[e.provider_name?.toLowerCase()] = e;
        });

        // Sort responses by rank (if available) or by score
        const sortedResponses = [...responses].sort((a, b) => {
            const evalA = evalMap[a.provider_name] || evalMap[a.provider_name?.toUpperCase()];
            const evalB = evalMap[b.provider_name] || evalMap[b.provider_name?.toUpperCase()];
            const rankA = evalA?.rank ?? 999;
            const rankB = evalB?.rank ?? 999;
            return rankA - rankB;
        });

        sortedResponses.forEach((response, index) => {
            // Try to find evaluation with various name formats
            let evaluation = evalMap[response.provider_name]
                || evalMap[response.provider_name?.toUpperCase()]
                || evalMap[response.provider_name?.toLowerCase()];

            // Fallback: match by index if names don't match
            if (!evaluation && evaluations[index]) {
                evaluation = evaluations[index];
            }

            const card = this._createResponseCard(response, evaluation);
            this.responsesContainer.appendChild(card);
        });
    }

    _createResponseCard(response, evaluation) {
        const card = document.createElement('div');
        card.className = 'response-card';
        if (evaluation?.rank === 1) card.classList.add('response-card--winner');

        const color = this._getProviderColor(response.provider_name);
        const scoreClass = this._getScoreClass(evaluation?.overall_score);

        card.innerHTML = `
            <div class="response-card__header">
                <span class="provider-badge" style="background: ${color}">
                    ${response.provider_name}
                </span>
                <span class="model-name">${response.model_name}</span>
                ${evaluation?.rank === 1 ? '<span class="winner-badge">WINNER</span>' : ''}
                <span class="score-badge ${scoreClass}">${evaluation?.overall_score?.toFixed(1) || '-'}</span>
            </div>
            <div class="response-card__body">
                <p class="response-text">${this._truncateText(response.response_text, 300)}</p>
                ${response.error ? `<p class="response-error">Error: ${response.error}</p>` : ''}
            </div>
            <div class="response-card__footer">
                <span class="execution-time">${response.execution_time?.toFixed(1) || 0}s</span>
                ${evaluation ? `
                    <div class="strengths-weaknesses">
                        <span class="strengths">+ ${evaluation.strengths?.length || 0} strengths</span>
                        <span class="weaknesses">- ${evaluation.weaknesses?.length || 0} weaknesses</span>
                    </div>
                ` : ''}
            </div>
        `;

        // Expand on click
        card.addEventListener('click', () => this._showResponseDetail(response, evaluation));

        return card;
    }

    _showResponseDetail(response, evaluation) {
        // Create modal or expand card
        const modal = document.createElement('div');
        modal.className = 'response-detail-modal';

        const color = this._getProviderColor(response.provider_name);

        modal.innerHTML = `
            <div class="response-detail-content">
                <div class="response-detail-header">
                    <span class="provider-badge" style="background: ${color}">${response.provider_name}</span>
                    <span class="model-name">${response.model_name}</span>
                    <span class="score-badge">${evaluation?.overall_score?.toFixed(1) || '-'}/10</span>
                    <button class="btn btn--ghost btn--sm btn-close-detail">×</button>
                </div>
                <div class="response-detail-body">
                    <h4>Response</h4>
                    <p class="full-response-text">${response.response_text}</p>

                    ${evaluation ? `
                        <h4>Evaluation</h4>
                        <p class="evaluation-reasoning">${evaluation.reasoning}</p>

                        <div class="evaluation-lists">
                            <div class="strengths-list">
                                <h5>Strengths</h5>
                                <ul>
                                    ${(evaluation.strengths || []).map(s => `<li>${s}</li>`).join('')}
                                </ul>
                            </div>
                            <div class="weaknesses-list">
                                <h5>Weaknesses</h5>
                                <ul>
                                    ${(evaluation.weaknesses || []).map(w => `<li>${w}</li>`).join('')}
                                </ul>
                            </div>
                        </div>

                        <h4>Criterion Scores</h4>
                        <div class="criterion-scores">
                            ${Object.entries(evaluation.criterion_scores || {}).map(([k, v]) =>
                                `<div class="criterion-score">
                                    <span class="criterion-name">${k}</span>
                                    <span class="criterion-value">${v.toFixed(1)}</span>
                                </div>`
                            ).join('')}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        // Close on click outside or close button
        modal.addEventListener('click', (e) => {
            if (e.target === modal || e.target.classList.contains('btn-close-detail')) {
                modal.remove();
            }
        });

        document.body.appendChild(modal);
    }

    _clearResponseCards() {
        if (this.responsesContainer) {
            this.responsesContainer.innerHTML = '';
        }
    }

    _showFinalAnswer(answer, mode) {
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

        // Debug: log evaluations to see what we have
        console.log('Evaluations:', result.evaluations);

        // Find winner - try rank 1 first, then fall back to highest score
        let winner = result.evaluations?.find(e => e.rank === 1);

        if (!winner && result.evaluations?.length > 0) {
            // Fallback: find highest overall_score
            winner = result.evaluations.reduce((best, current) => {
                const bestScore = best?.overall_score ?? 0;
                const currentScore = current?.overall_score ?? 0;
                return currentScore > bestScore ? current : best;
            }, result.evaluations[0]);
        }

        if (this.statWinner) {
            this.statWinner.textContent = winner?.provider_name || '-';
        }

        if (this.statBestScore) {
            const score = winner?.overall_score;
            this.statBestScore.textContent = (score !== undefined && score !== null)
                ? score.toFixed(1)
                : '-';
        }

        if (this.statTotalTime) {
            this.statTotalTime.textContent = `${result.total_execution_time?.toFixed(1) || 0}s`;
        }
    }

    _hideStats() {
        if (this.statsSection) {
            this.statsSection.classList.add('hidden');
        }
    }

    exportResults() {
        if (!this.result) {
            app.showToast('No results to export', 'warning');
            return;
        }

        try {
            const data = JSON.stringify(this.result, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `judge_result_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);

            app.showToast('Results exported!', 'success');
        } catch (error) {
            console.error('Export error:', error);
            app.showToast('Export failed', 'error');
        }
    }

    // Utility methods
    _getProviderColor(providerName) {
        return this.providerColors[providerName?.toUpperCase()] || '#888888';
    }

    _getScoreClass(score) {
        if (score === undefined || score === null) return '';
        if (score >= 8) return 'score-high';
        if (score >= 5) return 'score-mid';
        return 'score-low';
    }

    _truncateText(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    _capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // Called when navigating to judge view
    refresh() {
        this._populateProviders();
        this._updateCodeSnippet();
    }

    _populateProviders() {
        // Populate judge provider
        if (this.judgeProvider) {
            this._populateProviderSelect(this.judgeProvider);
        }

        // Populate contestant providers
        const contestantSelects = this.contestantsList?.querySelectorAll('.contestant-provider');
        contestantSelects?.forEach(select => {
            this._populateProviderSelect(select);
        });
    }

    _toggleCodeSection() {
        if (this.codeSectionContent && this.codeToggleIcon) {
            this.codeSectionContent.classList.toggle('hidden');
            this.codeToggleIcon.classList.toggle('collapsed');
        }
    }

    _updateCodeSnippet() {
        if (!this.codeSnippet) return;

        const contestants = this._getContestantsConfig();
        const judgeProvider = this.judgeProvider?.value || 'anthropic';
        const judgeModel = this.judgeModel?.value || 'claude-3-opus-20240229';
        const prompt = this.promptInput?.value || 'Your prompt here';
        const mode = this.currentMode;
        const criteria = this.selectedCriteria;

        // Map provider to enum name
        const providerEnumMap = {
            'openai': 'OPENAI',
            'anthropic': 'ANTHROPIC',
            'gemini': 'GEMINI',
            'deepseek': 'DEEPSEEK',
            'ollama': 'OLLAMA',
            'cohere': 'COHERE',
            'perplexity': 'PERPLEXITY'
        };

        // Escape prompt for Python string
        const escapedPrompt = prompt.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');

        const lines = [
            'from SimplerLLM.language.llm import LLM, LLMProvider',
            'from SimplerLLM.language.llm_judge import LLMJudge',
            '',
            '# Create contestant LLMs',
            'providers = ['
        ];

        if (contestants.length > 0) {
            contestants.forEach(c => {
                const enumName = providerEnumMap[c.provider] || 'OPENAI';
                lines.push(`    LLM.create(LLMProvider.${enumName}, model_name="${c.model}"),`);
            });
        } else {
            lines.push('    LLM.create(LLMProvider.OPENAI, model_name="gpt-4o"),');
            lines.push('    LLM.create(LLMProvider.ANTHROPIC, model_name="claude-3-5-sonnet-20241022"),');
        }

        lines.push(']');
        lines.push('');
        lines.push('# Create judge LLM');
        lines.push(`judge_llm = LLM.create(LLMProvider.${providerEnumMap[judgeProvider] || 'ANTHROPIC'}, model_name="${judgeModel}")`);
        lines.push('');
        lines.push('# Initialize LLMJudge');
        lines.push('judge = LLMJudge(');
        lines.push('    providers=providers,');
        lines.push('    judge_llm=judge_llm,');
        lines.push('    parallel=True,');
        lines.push(`    default_criteria=${JSON.stringify(criteria)}`);
        lines.push(')');
        lines.push('');
        lines.push('# Run evaluation');
        lines.push('result = judge.generate(');
        lines.push(`    prompt="${escapedPrompt}",`);
        lines.push(`    mode="${mode}"`);
        lines.push(')');
        lines.push('');
        lines.push('# Access results');
        lines.push('print(f"Final Answer: {result.final_answer}")');
        lines.push('print(f"Winner: {result.evaluations[0].provider_name}")');
        lines.push('for eval in result.evaluations:');
        lines.push('    print(f"  {eval.provider_name}: {eval.overall_score}/10 (Rank #{eval.rank})")');

        this.codeSnippet.textContent = lines.join('\n');
    }

    async _copyCode() {
        if (!this.codeSnippet) return;

        try {
            await navigator.clipboard.writeText(this.codeSnippet.textContent);

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
let judgeManager;
