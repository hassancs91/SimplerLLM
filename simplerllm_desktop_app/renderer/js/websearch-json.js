/**
 * Web Search + JSON Manager
 * Combines web search with Pydantic structured output generation
 */

/**
 * Schema Builder Component
 * Manages dynamic form for defining JSON schema fields
 */
class SchemaBuilder {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.fields = [];
        this.fieldCounter = 0;
    }

    addField(fieldData = null) {
        const field = {
            id: `field_${++this.fieldCounter}`,
            name: fieldData?.name || '',
            type: fieldData?.type || 'string',
            description: fieldData?.description || '',
            itemType: fieldData?.itemType || 'string'
        };
        this.fields.push(field);
        this.renderField(field);
        return field;
    }

    removeField(fieldId) {
        this.fields = this.fields.filter(f => f.id !== fieldId);
        const fieldEl = document.getElementById(fieldId);
        if (fieldEl) {
            fieldEl.remove();
        }
    }

    renderField(field) {
        const fieldEl = document.createElement('div');
        fieldEl.className = 'schema-field';
        fieldEl.id = field.id;

        fieldEl.innerHTML = `
            <div class="schema-field__row">
                <input type="text" class="input schema-field__name"
                       placeholder="Field name (e.g., company_name)"
                       value="${this.escapeHtml(field.name)}">
                <select class="select schema-field__type">
                    <option value="string" ${field.type === 'string' ? 'selected' : ''}>String</option>
                    <option value="number" ${field.type === 'number' ? 'selected' : ''}>Number</option>
                    <option value="integer" ${field.type === 'integer' ? 'selected' : ''}>Integer</option>
                    <option value="boolean" ${field.type === 'boolean' ? 'selected' : ''}>Boolean</option>
                    <option value="list" ${field.type === 'list' ? 'selected' : ''}>List</option>
                </select>
                <button class="btn btn--ghost btn--sm schema-field__remove" title="Remove field">
                    <span class="icon">X</span>
                </button>
            </div>
            <div class="schema-field__list-type ${field.type !== 'list' ? 'hidden' : ''}">
                <label class="input-label">Item type:</label>
                <select class="select schema-field__item-type">
                    <option value="string" ${field.itemType === 'string' ? 'selected' : ''}>String</option>
                    <option value="number" ${field.itemType === 'number' ? 'selected' : ''}>Number</option>
                    <option value="integer" ${field.itemType === 'integer' ? 'selected' : ''}>Integer</option>
                </select>
            </div>
            <input type="text" class="input schema-field__description"
                   placeholder="Description (helps the LLM understand what data to extract)"
                   value="${this.escapeHtml(field.description)}">
        `;

        // Event listeners
        const typeSelect = fieldEl.querySelector('.schema-field__type');
        typeSelect.addEventListener('change', (e) => {
            field.type = e.target.value;
            fieldEl.querySelector('.schema-field__list-type')
                   .classList.toggle('hidden', field.type !== 'list');
        });

        const removeBtn = fieldEl.querySelector('.schema-field__remove');
        removeBtn.addEventListener('click', () => {
            this.removeField(field.id);
        });

        // Update field data on input changes
        fieldEl.querySelector('.schema-field__name').addEventListener('input', (e) => {
            field.name = e.target.value;
        });
        fieldEl.querySelector('.schema-field__description').addEventListener('input', (e) => {
            field.description = e.target.value;
        });
        fieldEl.querySelector('.schema-field__item-type').addEventListener('change', (e) => {
            field.itemType = e.target.value;
        });

        this.container.appendChild(fieldEl);
    }

    getSchema() {
        return {
            fields: this.fields
                .map(f => {
                    const fieldEl = document.getElementById(f.id);
                    if (!fieldEl) return null;

                    const name = fieldEl.querySelector('.schema-field__name').value.trim();
                    if (!name) return null;

                    return {
                        name: name,
                        type: fieldEl.querySelector('.schema-field__type').value,
                        description: fieldEl.querySelector('.schema-field__description').value.trim(),
                        item_type: fieldEl.querySelector('.schema-field__item-type')?.value || 'string'
                    };
                })
                .filter(f => f !== null)
        };
    }

    clear() {
        this.fields = [];
        this.container.innerHTML = '';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}


/**
 * Web Search JSON Manager
 * Main manager class for the Web Search + JSON tool
 */
class WebSearchJsonManager {
    constructor() {
        // DOM elements
        this.promptInput = document.getElementById('websearch-prompt');
        this.providerSelect = document.getElementById('websearch-provider');
        this.modelInput = document.getElementById('websearch-model');
        this.schemaFieldsContainer = document.getElementById('schema-fields');
        this.btnAddField = document.getElementById('btn-add-field');
        this.btnGenerate = document.getElementById('btn-generate-websearch');
        this.btnClear = document.getElementById('btn-clear-websearch');
        this.resultsContainer = document.getElementById('websearch-results');
        this.sourcesContainer = document.getElementById('websearch-sources');
        this.statusContainer = document.getElementById('websearch-status');
        this.statusBadge = document.getElementById('websearch-status-badge');
        this.codeToggle = document.getElementById('toggle-websearch-code');
        this.codeContent = document.getElementById('websearch-code-content');
        this.codeSnippet = document.getElementById('websearch-code-snippet');
        this.btnCopyCode = document.getElementById('btn-copy-websearch-code');

        // Schema mode elements
        this.schemaModeToggle = document.getElementById('schema-mode-toggle');
        this.schemaFormSection = document.getElementById('schema-form-section');
        this.schemaCodeSection = document.getElementById('schema-code-section');
        this.schemaCodeInput = document.getElementById('schema-code-input');

        // State
        this.schemaBuilder = null;
        this.isRunning = false;
        this.providers = [];
        this.schemaMode = 'form';  // 'form' or 'code'

        this.init();
    }

    init() {
        // Initialize schema builder
        this.schemaBuilder = new SchemaBuilder('schema-fields');

        // Add initial fields as examples
        this.schemaBuilder.addField({ name: 'title', type: 'string', description: 'Main title or name' });
        this.schemaBuilder.addField({ name: 'summary', type: 'string', description: 'Brief summary of the information' });

        // Event listeners
        if (this.btnAddField) {
            this.btnAddField.addEventListener('click', () => this.schemaBuilder.addField());
        }

        if (this.btnGenerate) {
            this.btnGenerate.addEventListener('click', () => this.generate());
        }

        if (this.btnClear) {
            this.btnClear.addEventListener('click', () => this.clearAll());
        }

        if (this.providerSelect) {
            this.providerSelect.addEventListener('change', (e) => this.updateModelSuggestion(e.target.value));
        }

        if (this.codeToggle) {
            this.codeToggle.addEventListener('click', () => this.toggleCodeSection());
        }

        if (this.btnCopyCode) {
            this.btnCopyCode.addEventListener('click', () => this.copyCode());
        }

        // Schema mode toggle
        if (this.schemaModeToggle) {
            this.schemaModeToggle.querySelectorAll('.mode-btn').forEach(btn => {
                btn.addEventListener('click', (e) => this.setSchemaMode(e.target.dataset.mode));
            });
        }

        // Back button
        const btnBack = document.getElementById('btn-back-from-websearch');
        if (btnBack) {
            btnBack.addEventListener('click', () => {
                if (typeof app !== 'undefined') {
                    app.navigateTo('tools-view');
                }
            });
        }

        // Load providers
        this.loadProviders();
    }

    async loadProviders() {
        try {
            const response = await api.getWebSearchProviders();
            if (response.success && response.providers) {
                this.providers = response.providers;
                this.renderProviderDropdown();
            }
        } catch (error) {
            console.error('Failed to load web search providers:', error);
            // Use defaults
            this.providers = [
                { id: 'openai', name: 'OpenAI (Web Search)', default_model: 'gpt-4o' },
                { id: 'perplexity', name: 'Perplexity (Native Search)', default_model: 'sonar-pro' }
            ];
            this.renderProviderDropdown();
        }
    }

    renderProviderDropdown() {
        if (!this.providerSelect) return;

        this.providerSelect.innerHTML = '<option value="">Select provider...</option>';
        this.providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            option.textContent = provider.name;
            this.providerSelect.appendChild(option);
        });
    }

    updateModelSuggestion(providerId) {
        if (!this.modelInput) return;

        const provider = this.providers.find(p => p.id === providerId);
        if (provider && provider.default_model) {
            this.modelInput.value = provider.default_model;
        }
    }

    setSchemaMode(mode) {
        this.schemaMode = mode;

        // Update toggle buttons
        if (this.schemaModeToggle) {
            this.schemaModeToggle.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
        }

        // Show/hide sections
        if (this.schemaFormSection) {
            this.schemaFormSection.classList.toggle('hidden', mode !== 'form');
        }
        if (this.schemaCodeSection) {
            this.schemaCodeSection.classList.toggle('hidden', mode !== 'code');
        }
    }

    async generate() {
        // Validation
        const prompt = this.promptInput?.value.trim();
        if (!prompt) {
            this.showToast('Please enter a search query', 'warning');
            return;
        }

        const provider = this.providerSelect?.value;
        const model = this.modelInput?.value.trim();

        if (!provider) {
            this.showToast('Please select a provider', 'warning');
            return;
        }

        if (!model) {
            this.showToast('Please enter a model name', 'warning');
            return;
        }

        // Validate based on schema mode
        let schema = null;
        let schemaCode = null;

        if (this.schemaMode === 'code') {
            schemaCode = this.schemaCodeInput?.value.trim();
            if (!schemaCode) {
                this.showToast('Please paste your Pydantic model code', 'warning');
                return;
            }
        } else {
            schema = this.schemaBuilder.getSchema();
            if (schema.fields.length === 0) {
                this.showToast('Please add at least one field to the schema', 'warning');
                return;
            }
        }

        // Start generation
        this.setRunning(true);
        this.clearResults();
        this.showStatus('Searching the web and generating structured output...');

        try {
            const response = await api.generateWebSearchJson(
                prompt,
                provider,
                model,
                schema,
                {
                    temperature: 0.7,
                    max_tokens: 2000
                },
                this.schemaMode,
                schemaCode
            );

            if (response.success) {
                this.displayResults(response.result);
                if (this.schemaMode === 'form' && schema) {
                    this.updateCodeExample(prompt, provider, model, schema);
                }
                this.showToast('Generation complete!', 'success');
            } else {
                throw new Error(response.error || 'Generation failed');
            }
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
            this.displayError(error.message);
        } finally {
            this.setRunning(false);
            this.hideStatus();
        }
    }

    displayResults(result) {
        if (!this.resultsContainer) return;

        // Display JSON output
        const jsonHtml = this.syntaxHighlight(JSON.stringify(result.data, null, 2));

        this.resultsContainer.innerHTML = `
            <div class="websearch-result-card">
                <div class="result-header">
                    <h3>Structured Output</h3>
                    <button class="btn btn--ghost btn--sm" id="btn-copy-json" title="Copy JSON">
                        Copy
                    </button>
                </div>
                <pre class="json-output">${jsonHtml}</pre>
                <div class="result-meta">
                    <span class="meta-item">
                        <span class="meta-label">Time:</span> ${result.process_time}s
                    </span>
                    <span class="meta-item">
                        <span class="meta-label">Tokens:</span> ${result.tokens.input} in / ${result.tokens.output} out
                    </span>
                    <span class="meta-item">
                        <span class="meta-label">Provider:</span> ${result.provider_used}/${result.model_used}
                    </span>
                </div>
            </div>
        `;

        // Copy JSON button handler
        const btnCopyJson = document.getElementById('btn-copy-json');
        if (btnCopyJson) {
            btnCopyJson.addEventListener('click', () => {
                navigator.clipboard.writeText(JSON.stringify(result.data, null, 2));
                this.showToast('JSON copied to clipboard', 'success');
            });
        }

        // Display sources
        if (this.sourcesContainer) {
            if (result.sources && result.sources.length > 0) {
                this.sourcesContainer.innerHTML = `
                    <div class="websearch-sources-card">
                        <h4>Sources (${result.sources.length})</h4>
                        <ul class="sources-list">
                            ${result.sources.map(source => `
                                <li class="source-item">
                                    <a href="${this.escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">
                                        ${this.escapeHtml(source.title || source.url)}
                                    </a>
                                    <span class="source-url">${this.escapeHtml(source.url)}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                `;
            } else {
                this.sourcesContainer.innerHTML = `
                    <div class="websearch-sources-card websearch-sources-card--empty">
                        <p class="no-sources">No source citations available for this query.</p>
                    </div>
                `;
            }
        }
    }

    displayError(message) {
        if (!this.resultsContainer) return;

        this.resultsContainer.innerHTML = `
            <div class="websearch-error">
                <div class="error-icon">!</div>
                <h3>Generation Failed</h3>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;

        if (this.sourcesContainer) {
            this.sourcesContainer.innerHTML = '';
        }
    }

    async updateCodeExample(prompt, provider, model, schema) {
        if (!this.codeSnippet) return;

        try {
            const response = await api.getWebSearchCodeExample(prompt, provider, model, schema);
            if (response.success) {
                this.codeSnippet.textContent = response.code;
            }
        } catch (error) {
            console.error('Failed to generate code example:', error);
        }
    }

    toggleCodeSection() {
        if (this.codeContent) {
            this.codeContent.classList.toggle('hidden');
            const icon = this.codeToggle?.querySelector('.toggle-icon');
            if (icon) {
                icon.textContent = this.codeContent.classList.contains('hidden') ? '+' : '-';
            }
        }
    }

    async copyCode() {
        if (this.codeSnippet) {
            try {
                await navigator.clipboard.writeText(this.codeSnippet.textContent);
                this.showToast('Code copied to clipboard', 'success');
            } catch (error) {
                this.showToast('Failed to copy code', 'error');
            }
        }
    }

    clearResults() {
        if (this.resultsContainer) {
            this.resultsContainer.innerHTML = `
                <div class="websearch-empty-state">
                    <div class="empty-icon">?</div>
                    <h3>No Results Yet</h3>
                    <p>Define your schema and run a search to get structured data</p>
                </div>
            `;
        }
        if (this.sourcesContainer) {
            this.sourcesContainer.innerHTML = '';
        }
    }

    clearAll() {
        if (this.promptInput) this.promptInput.value = '';
        this.schemaBuilder.clear();
        // Add back default fields
        this.schemaBuilder.addField({ name: 'title', type: 'string', description: 'Main title or name' });
        this.schemaBuilder.addField({ name: 'summary', type: 'string', description: 'Brief summary of the information' });
        // Clear code input
        if (this.schemaCodeInput) this.schemaCodeInput.value = '';
        // Reset to form mode
        this.setSchemaMode('form');
        this.clearResults();
        if (this.codeSnippet) this.codeSnippet.textContent = '';
    }

    setRunning(running) {
        this.isRunning = running;
        if (this.btnGenerate) {
            this.btnGenerate.disabled = running;
            this.btnGenerate.textContent = running ? 'Generating...' : 'Generate';
        }
    }

    showStatus(message) {
        if (this.statusContainer) {
            this.statusContainer.classList.remove('hidden');
        }
        if (this.statusBadge) {
            this.statusBadge.textContent = message;
        }
    }

    hideStatus() {
        if (this.statusContainer) {
            this.statusContainer.classList.add('hidden');
        }
    }

    showToast(message, type = 'info') {
        if (typeof app !== 'undefined' && app.showToast) {
            app.showToast(message, type);
        } else {
            console.log(`[${type}] ${message}`);
        }
    }

    syntaxHighlight(json) {
        // Simple JSON syntax highlighting
        return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, (match) => {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return `<span class="${cls}">${match}</span>`;
        });
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    refresh() {
        // Called when navigating to this view
        this.loadProviders();
    }
}

// Global instance
let websearchJsonManager;
