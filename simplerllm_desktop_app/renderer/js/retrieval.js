/**
 * Retrieval Manager
 * Handles the LLM Retrieval tool UI and interactions
 */
class RetrievalManager {
    constructor() {
        // State
        this.isBuilding = false;
        this.isQuerying = false;
        this.indexBuilt = false;
        this.buildController = null;
        this.queryController = null;
        this.treeData = null;
        this.samples = [];
        this.startTime = null;
        this.timerInterval = null;

        // Tree visualizer (initialized in init())
        this.treeVisualizer = null;

        // DOM elements (cached in init)
        this.elements = {};
    }

    init() {
        // Cache DOM elements
        this.elements = {
            // Navigation
            backBtn: document.getElementById('btn-back-from-retrieval'),

            // Tree visualization
            treeContainer: document.getElementById('retrieval-tree-container'),
            treeSvg: document.getElementById('retrieval-svg'),
            zoomInBtn: document.getElementById('btn-retrieval-zoom-in'),
            zoomOutBtn: document.getElementById('btn-retrieval-zoom-out'),
            zoomResetBtn: document.getElementById('btn-retrieval-zoom-reset'),

            // Input source
            sourceRadios: document.querySelectorAll('input[name="retrieval-source"]'),
            sampleSection: document.getElementById('retrieval-sample-section'),
            textSection: document.getElementById('retrieval-text-section'),
            sampleSelect: document.getElementById('retrieval-sample-select'),
            textInput: document.getElementById('retrieval-text-input'),

            // Provider/Model
            providerSelect: document.getElementById('retrieval-provider'),
            modelInput: document.getElementById('retrieval-model'),

            // Build controls
            buildBtn: document.getElementById('btn-build-index'),
            buildBtnText: document.getElementById('build-btn-text'),
            cancelBuildBtn: document.getElementById('btn-cancel-build'),
            clearIndexBtn: document.getElementById('btn-clear-index'),

            // Build status
            buildStatus: document.getElementById('retrieval-build-status'),
            buildStatusBadge: document.getElementById('retrieval-build-status-badge'),
            buildProgress: document.getElementById('retrieval-build-progress'),
            buildElapsedTime: document.getElementById('build-elapsed-time'),
            buildStage: document.getElementById('build-stage'),

            // Query section
            querySection: document.getElementById('retrieval-query-section'),
            queryInput: document.getElementById('retrieval-query-input'),
            topKSlider: document.getElementById('retrieval-top-k'),
            topKValue: document.getElementById('retrieval-top-k-value'),
            searchBtn: document.getElementById('btn-search'),
            searchBtnText: document.getElementById('search-btn-text'),
            cancelSearchBtn: document.getElementById('btn-cancel-search'),

            // Query status
            queryStatus: document.getElementById('retrieval-query-status'),
            queryStatusBadge: document.getElementById('retrieval-query-status-badge'),
            queryProgress: document.getElementById('retrieval-query-progress'),
            queryElapsedTime: document.getElementById('query-elapsed-time'),

            // Results
            resultsSection: document.getElementById('retrieval-results-section'),
            navigationPath: document.getElementById('retrieval-navigation-path'),
            resultsContainer: document.getElementById('retrieval-results-container'),

            // Stats
            statsSection: document.getElementById('retrieval-stats'),
            statChunks: document.getElementById('stat-chunks'),
            statClusters: document.getElementById('stat-clusters'),
            statLlmCalls: document.getElementById('stat-llm-calls'),
            statTime: document.getElementById('stat-time'),

            // Node detail
            nodeDetail: document.getElementById('retrieval-node-detail'),
            detailName: document.getElementById('detail-name'),
            detailDescription: document.getElementById('detail-description'),
            detailKeywords: document.getElementById('detail-keywords'),
            detailChunkCount: document.getElementById('detail-chunk-count'),

            // Code section
            codeSection: document.getElementById('retrieval-code-section'),
            codeToggle: document.getElementById('toggle-retrieval-code'),
            codeContent: document.getElementById('retrieval-code-content'),
            codeSnippet: document.getElementById('retrieval-code-snippet'),
            copyCodeBtn: document.getElementById('btn-copy-retrieval-code')
        };

        // Initialize tree visualizer
        if (this.elements.treeContainer && this.elements.treeSvg) {
            this.treeVisualizer = new ClusterTreeVisualizer('retrieval-tree-container', 'retrieval-svg');
            this.treeVisualizer.onNodeClick = (id, data) => this._handleNodeClick(id, data);
        }

        // Setup event listeners
        this._setupEventListeners();

        // Load samples
        this._loadSamples();

        // Populate providers
        this._populateProviders();

        // Update code example
        this._updateCodeExample();
    }

    _setupEventListeners() {
        // Back button
        if (this.elements.backBtn) {
            this.elements.backBtn.addEventListener('click', () => {
                app.navigateTo('tools-view');
            });
        }

        // Zoom controls
        if (this.elements.zoomInBtn) {
            this.elements.zoomInBtn.addEventListener('click', () => this.treeVisualizer?.zoomIn());
        }
        if (this.elements.zoomOutBtn) {
            this.elements.zoomOutBtn.addEventListener('click', () => this.treeVisualizer?.zoomOut());
        }
        if (this.elements.zoomResetBtn) {
            this.elements.zoomResetBtn.addEventListener('click', () => this.treeVisualizer?.resetView());
        }

        // Source radio buttons
        this.elements.sourceRadios.forEach(radio => {
            radio.addEventListener('change', () => this._handleSourceChange());
        });

        // Top-K slider
        if (this.elements.topKSlider) {
            this.elements.topKSlider.addEventListener('input', () => {
                this.elements.topKValue.textContent = this.elements.topKSlider.value;
            });
        }

        // Build button
        if (this.elements.buildBtn) {
            this.elements.buildBtn.addEventListener('click', () => this.startBuildIndex());
        }

        // Cancel build button
        if (this.elements.cancelBuildBtn) {
            this.elements.cancelBuildBtn.addEventListener('click', () => this.cancelBuild());
        }

        // Clear index button
        if (this.elements.clearIndexBtn) {
            this.elements.clearIndexBtn.addEventListener('click', () => this.clearIndex());
        }

        // Search button
        if (this.elements.searchBtn) {
            this.elements.searchBtn.addEventListener('click', () => this.startQuery());
        }

        // Cancel search button
        if (this.elements.cancelSearchBtn) {
            this.elements.cancelSearchBtn.addEventListener('click', () => this.cancelQuery());
        }

        // Code section toggle
        if (this.elements.codeToggle) {
            this.elements.codeToggle.addEventListener('click', () => {
                this.elements.codeContent.classList.toggle('hidden');
                const icon = document.getElementById('retrieval-code-toggle-icon');
                if (icon) {
                    icon.textContent = this.elements.codeContent.classList.contains('hidden') ? '\u25BC' : '\u25B2';
                }
            });
        }

        // Copy code button
        if (this.elements.copyCodeBtn) {
            this.elements.copyCodeBtn.addEventListener('click', () => this._copyCode());
        }

        // Enter key on query input
        if (this.elements.queryInput) {
            this.elements.queryInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.startQuery();
                }
            });
        }
    }

    _handleSourceChange() {
        const selectedSource = document.querySelector('input[name="retrieval-source"]:checked')?.value;

        if (selectedSource === 'sample') {
            this.elements.sampleSection?.classList.remove('hidden');
            this.elements.textSection?.classList.add('hidden');
        } else {
            this.elements.sampleSection?.classList.add('hidden');
            this.elements.textSection?.classList.remove('hidden');
        }
    }

    async _loadSamples() {
        try {
            const response = await api.getRetrievalSamples();
            if (response.success && response.samples) {
                this.samples = response.samples;
                this._populateSampleSelect();
            }
        } catch (error) {
            console.error('Failed to load samples:', error);
        }
    }

    _populateSampleSelect() {
        if (!this.elements.sampleSelect) return;

        this.elements.sampleSelect.innerHTML = '';

        for (const sample of this.samples) {
            const option = document.createElement('option');
            option.value = sample.id;
            option.textContent = `${sample.name} (${sample.chunk_count} chunks)`;
            this.elements.sampleSelect.appendChild(option);
        }
    }

    _populateProviders() {
        if (!this.elements.providerSelect) return;
        if (typeof app === 'undefined' || !app.providers) return;

        const providers = app.providers || [];
        this.elements.providerSelect.innerHTML = '<option value="">Select provider...</option>';

        providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            const status = provider.configured ? '' : ' (Not configured)';
            option.textContent = provider.name + status;
            this.elements.providerSelect.appendChild(option);
        });

        // Auto-select first configured provider
        const configured = providers.find(p => p.configured);
        if (configured) {
            this.elements.providerSelect.value = configured.id;
            if (typeof app !== 'undefined' && app.getSavedModel) {
                const savedModel = app.getSavedModel(configured.id);
                if (savedModel && this.elements.modelInput) {
                    this.elements.modelInput.value = savedModel;
                }
            }
        }
    }

    startBuildIndex() {
        if (this.isBuilding) return;

        const source = document.querySelector('input[name="retrieval-source"]:checked')?.value || 'sample';
        const text = source === 'text' ? this.elements.textInput?.value.trim() : null;
        const sampleId = source === 'sample' ? this.elements.sampleSelect?.value : null;
        const provider = this.elements.providerSelect?.value || 'openai';
        const model = this.elements.modelInput?.value.trim() || 'gpt-4o';

        // Validation
        if (source === 'text' && (!text || text.length < 50)) {
            app?.showToast('Please enter at least 50 characters of text', 'error');
            return;
        }
        if (source === 'sample' && !sampleId) {
            app?.showToast('Please select a sample dataset', 'error');
            return;
        }

        // Check API key
        const apiKey = app?.getApiKey(provider);
        if (!apiKey && provider !== 'ollama') {
            app?.showToast(`Please configure API key for ${provider}`, 'error');
            return;
        }

        this.isBuilding = true;
        this._updateBuildUI(true);
        this._startTimer('build');

        // Clear previous tree
        this.treeVisualizer?.clear();
        this.elements.statsSection?.classList.add('hidden');
        this.elements.querySection?.classList.add('hidden');
        this.elements.resultsSection?.classList.add('hidden');
        this.indexBuilt = false;

        // Start streaming build
        this.buildController = api.streamBuildIndex(
            source,
            text,
            sampleId,
            provider,
            model,
            (event) => this._handleBuildEvent(event),
            (error) => this._handleBuildError(error),
            (complete) => this._handleBuildComplete(complete)
        );
    }

    _handleBuildEvent(event) {
        switch (event.type) {
            case 'start':
                this._updateBuildStage('Starting...');
                break;
            case 'chunking_complete':
                this._updateBuildStage(`Chunked into ${event.chunks_count} chunks`);
                this._updateBuildProgress(20);
                break;
            case 'clustering_progress':
                this._updateBuildStage(event.message || 'Clustering...');
                this._updateBuildProgress(50);
                break;
        }
    }

    _handleBuildError(error) {
        this.isBuilding = false;
        this._stopTimer();
        this._updateBuildUI(false);
        app?.showToast(`Build failed: ${error.message}`, 'error');
    }

    _handleBuildComplete(data) {
        this.isBuilding = false;
        this._stopTimer();
        this._updateBuildUI(false);
        this._updateBuildProgress(100);

        // Store tree data
        this.treeData = data.tree;
        this.indexBuilt = true;

        // Build tree visualization
        if (this.treeVisualizer && data.tree) {
            this.treeVisualizer.buildTree(data.tree);
        }

        // Update stats
        if (data.stats) {
            this._updateStats(data.stats);
        }

        // Show query section
        this.elements.querySection?.classList.remove('hidden');

        app?.showToast('Index built successfully!', 'success');
    }

    cancelBuild() {
        if (this.buildController) {
            this.buildController.abort();
            this.buildController = null;
        }
        this.isBuilding = false;
        this._stopTimer();
        this._updateBuildUI(false);
    }

    async clearIndex() {
        try {
            await api.clearRetrievalIndex();
            this.indexBuilt = false;
            this.treeData = null;
            this.treeVisualizer?.clear();
            this.elements.querySection?.classList.add('hidden');
            this.elements.resultsSection?.classList.add('hidden');
            this.elements.statsSection?.classList.add('hidden');
            app?.showToast('Index cleared', 'success');
        } catch (error) {
            app?.showToast(`Failed to clear index: ${error.message}`, 'error');
        }
    }

    startQuery() {
        if (this.isQuerying || !this.indexBuilt) return;

        const query = this.elements.queryInput?.value.trim();
        if (!query) {
            app?.showToast('Please enter a search query', 'error');
            return;
        }

        const topK = parseInt(this.elements.topKSlider?.value || '3');
        const provider = this.elements.providerSelect?.value || 'openai';
        const model = this.elements.modelInput?.value.trim() || 'gpt-4o';

        this.isQuerying = true;
        this._updateQueryUI(true);
        this._startTimer('query');

        // Clear previous results
        this.elements.resultsContainer.innerHTML = '';
        this.elements.navigationPath.innerHTML = '';
        this.treeVisualizer?.clearHighlight();

        // Start streaming query
        this.queryController = api.streamRetrieve(
            query,
            topK,
            provider,
            model,
            (event) => this._handleQueryEvent(event),
            (error) => this._handleQueryError(error),
            (complete) => this._handleQueryComplete(complete)
        );
    }

    _handleQueryEvent(event) {
        switch (event.type) {
            case 'start':
                this._updateQueryProgress(10);
                break;
            case 'navigation_step':
                this._addNavigationStep(event);
                this._updateQueryProgress(30 + event.level * 10);
                break;
            case 'result':
                this._addResultCard(event);
                this._updateQueryProgress(70 + event.rank * 10);
                break;
        }
    }

    _handleQueryError(error) {
        this.isQuerying = false;
        this._stopTimer();
        this._updateQueryUI(false);
        app?.showToast(`Query failed: ${error.message}`, 'error');
    }

    _handleQueryComplete(data) {
        this.isQuerying = false;
        this._stopTimer();
        this._updateQueryUI(false);
        this._updateQueryProgress(100);

        // Highlight navigation path in tree
        if (data.navigation_path && data.navigation_path.length > 0) {
            const pathIds = data.navigation_path.map(step => step.cluster_id).filter(id => id);
            this.treeVisualizer?.highlightPath(pathIds);
        }

        // Show results section
        this.elements.resultsSection?.classList.remove('hidden');

        // Update query stats
        if (data.stats) {
            this.elements.statLlmCalls.textContent = data.stats.total_llm_calls || '-';
            this.elements.statTime.textContent = data.stats.total_time_ms ?
                `${(data.stats.total_time_ms / 1000).toFixed(1)}s` : '-';
        }
    }

    cancelQuery() {
        if (this.queryController) {
            this.queryController.abort();
            this.queryController = null;
        }
        this.isQuerying = false;
        this._stopTimer();
        this._updateQueryUI(false);
    }

    _addNavigationStep(step) {
        if (!this.elements.navigationPath) return;

        const stepEl = document.createElement('span');
        stepEl.className = 'nav-step';
        stepEl.innerHTML = `
            <span class="nav-cluster">${step.cluster_name}</span>
            <span class="nav-confidence">${(step.confidence * 100).toFixed(0)}%</span>
        `;

        if (this.elements.navigationPath.children.length > 0) {
            const arrow = document.createElement('span');
            arrow.className = 'nav-arrow';
            arrow.textContent = ' \u2192 ';
            this.elements.navigationPath.appendChild(arrow);
        }

        this.elements.navigationPath.appendChild(stepEl);
    }

    _addResultCard(result) {
        if (!this.elements.resultsContainer) return;

        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="result-header">
                <span class="result-rank">#${result.rank}</span>
                <span class="result-confidence">${(result.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="result-text">${this._truncateText(result.chunk_text, 200)}</div>
            <div class="result-path">${result.cluster_path.join(' \u2192 ')}</div>
            ${result.reasoning ? `<div class="result-reasoning">${result.reasoning}</div>` : ''}
        `;

        this.elements.resultsContainer.appendChild(card);
    }

    _truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    _handleNodeClick(id, data) {
        if (!data) return;

        // Show node detail
        this.elements.nodeDetail?.classList.remove('hidden');
        this.elements.detailName.textContent = data.name || `Cluster ${id}`;
        this.elements.detailDescription.textContent = data.description || 'No description';
        this.elements.detailKeywords.textContent = data.keywords?.join(', ') || 'None';
        this.elements.detailChunkCount.textContent = data.chunk_count || 0;
    }

    _updateBuildUI(building) {
        if (building) {
            this.elements.buildBtn?.classList.add('hidden');
            this.elements.cancelBuildBtn?.classList.remove('hidden');
            this.elements.buildStatus?.classList.remove('hidden');
            this.elements.buildStatusBadge.textContent = 'Building...';
            this.elements.buildStatusBadge.className = 'badge badge--warning';
        } else {
            this.elements.buildBtn?.classList.remove('hidden');
            this.elements.cancelBuildBtn?.classList.add('hidden');
            if (this.indexBuilt) {
                this.elements.buildStatusBadge.textContent = 'Ready';
                this.elements.buildStatusBadge.className = 'badge badge--success';
            } else {
                this.elements.buildStatus?.classList.add('hidden');
            }
        }
    }

    _updateQueryUI(querying) {
        if (querying) {
            this.elements.searchBtn?.classList.add('hidden');
            this.elements.cancelSearchBtn?.classList.remove('hidden');
            this.elements.queryStatus?.classList.remove('hidden');
            this.elements.queryStatusBadge.textContent = 'Searching...';
        } else {
            this.elements.searchBtn?.classList.remove('hidden');
            this.elements.cancelSearchBtn?.classList.add('hidden');
            this.elements.queryStatus?.classList.add('hidden');
        }
    }

    _updateBuildProgress(percent) {
        if (this.elements.buildProgress) {
            this.elements.buildProgress.style.width = `${percent}%`;
        }
    }

    _updateQueryProgress(percent) {
        if (this.elements.queryProgress) {
            this.elements.queryProgress.style.width = `${percent}%`;
        }
    }

    _updateBuildStage(stage) {
        if (this.elements.buildStage) {
            this.elements.buildStage.textContent = stage;
        }
    }

    _updateStats(stats) {
        this.elements.statsSection?.classList.remove('hidden');
        this.elements.statChunks.textContent = stats.total_chunks || '-';
        this.elements.statClusters.textContent = stats.total_clusters || '-';
        this.elements.statLlmCalls.textContent = stats.llm_calls || '-';
    }

    _startTimer(type) {
        this.startTime = Date.now();
        const displayEl = type === 'build' ? this.elements.buildElapsedTime : this.elements.queryElapsedTime;

        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            if (displayEl) displayEl.textContent = `${elapsed}s`;
        }, 1000);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    _updateCodeExample() {
        const code = `from SimplerLLM.language import (
    LLM, LLMProvider, LLMClusterer,
    LLMRetriever, ChunkReference
)
from SimplerLLM.language.llm_router import LLMRouter

# Initialize LLM
llm = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o"
)

# Prepare text chunks
texts = ["Your text chunk 1...", "Your text chunk 2...", ...]
chunks = [
    ChunkReference(chunk_id=i, text=text)
    for i, text in enumerate(texts)
]

# Build hierarchical index
clusterer = LLMClusterer(llm)
result = clusterer.cluster(chunks, build_hierarchy=True)

# Setup retriever
router = LLMRouter(llm)
retriever = LLMRetriever(
    llm_router=router,
    cluster_tree=result.tree
)

# Query the index
response = retriever.retrieve(
    query="Your search query",
    top_k=3
)

# Access results
for result in response.results:
    print(f"Chunk: {result.chunk_text[:100]}...")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Path: {' -> '.join(result.cluster_path)}")`;

        if (this.elements.codeSnippet) {
            this.elements.codeSnippet.textContent = code;
        }
    }

    // Called when navigating to retrieval view
    refresh() {
        this._populateProviders();
        this._updateCodeExample();
    }

    _copyCode() {
        const code = this.elements.codeSnippet?.textContent;
        if (code) {
            navigator.clipboard.writeText(code);
            app?.showToast('Code copied to clipboard', 'success');
        }
    }
}

// Create global instance
const retrievalManager = new RetrievalManager();
