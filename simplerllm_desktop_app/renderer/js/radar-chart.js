/**
 * Radar Chart Visualizer
 * SVG-based radar chart for displaying multi-criteria scores
 */
class RadarChart {
    constructor(svgId, legendId) {
        this.svg = document.getElementById(svgId);
        this.legendContainer = document.getElementById(legendId);

        // Configuration
        this.config = {
            width: 400,
            height: 400,
            centerX: 200,
            centerY: 200,
            radius: 140,
            levels: 5,
            maxValue: 10,
            labelOffset: 25,
            colors: [
                '#00D4AA',  // Teal
                '#FF6B9D',  // Pink
                '#FFE156',  // Yellow
                '#9B59B6',  // Purple
                '#3498DB',  // Blue
                '#E67E22'   // Orange
            ],
            animationDuration: 500
        };

        this.datasets = [];
        this.criteria = [];
        this.tooltip = null;
    }

    /**
     * Render the radar chart with given data.
     * @param {Array<string>} criteria - Array of criteria names
     * @param {Array<Object>} datasets - Array of {label, values[], color?}
     */
    render(criteria, datasets) {
        if (!this.svg) return;

        this.criteria = criteria;
        this.datasets = datasets;

        // Clear previous content
        this.svg.innerHTML = '';

        // Update viewBox for proper sizing
        this.svg.setAttribute('viewBox', `0 0 ${this.config.width} ${this.config.height}`);

        // Create main group centered
        const g = this._createSVGElement('g', {
            transform: `translate(${this.config.centerX}, ${this.config.centerY})`
        });
        this.svg.appendChild(g);

        // Draw background grid
        this._drawGrid(g);

        // Draw axes
        this._drawAxes(g);

        // Draw data polygons
        datasets.forEach((dataset, idx) => {
            this._drawDataPolygon(g, dataset, idx);
        });

        // Draw axis labels
        this._drawLabels(g);

        // Update legend
        this._renderLegend();
    }

    /**
     * Draw concentric polygons as grid background.
     */
    _drawGrid(g) {
        const { levels, radius } = this.config;
        const numAxes = this.criteria.length;
        if (numAxes < 3) return;

        const angleSlice = (Math.PI * 2) / numAxes;

        for (let level = 1; level <= levels; level++) {
            const r = (radius / levels) * level;
            const points = [];

            for (let i = 0; i < numAxes; i++) {
                const angle = angleSlice * i - Math.PI / 2;
                points.push(`${r * Math.cos(angle)},${r * Math.sin(angle)}`);
            }

            const polygon = this._createSVGElement('polygon', {
                points: points.join(' '),
                fill: 'none',
                stroke: 'rgba(26, 26, 46, 0.15)',
                'stroke-width': 1
            });
            g.appendChild(polygon);

            // Value label on first axis
            const labelValue = (this.config.maxValue / levels) * level;
            const text = this._createSVGElement('text', {
                x: 5,
                y: -r + 4,
                'font-size': '10px',
                fill: 'rgba(26, 26, 46, 0.5)'
            });
            text.textContent = labelValue.toFixed(0);
            g.appendChild(text);
        }
    }

    /**
     * Draw axis lines from center to each criterion.
     */
    _drawAxes(g) {
        const { radius } = this.config;
        const numAxes = this.criteria.length;
        if (numAxes < 3) return;

        const angleSlice = (Math.PI * 2) / numAxes;

        for (let i = 0; i < numAxes; i++) {
            const angle = angleSlice * i - Math.PI / 2;
            const x = radius * Math.cos(angle);
            const y = radius * Math.sin(angle);

            const line = this._createSVGElement('line', {
                x1: 0,
                y1: 0,
                x2: x,
                y2: y,
                stroke: 'rgba(26, 26, 46, 0.2)',
                'stroke-width': 1
            });
            g.appendChild(line);
        }
    }

    /**
     * Draw a data polygon for one dataset.
     */
    _drawDataPolygon(g, dataset, colorIndex) {
        const { radius, maxValue } = this.config;
        const numAxes = this.criteria.length;
        if (numAxes < 3) return;

        const angleSlice = (Math.PI * 2) / numAxes;
        const color = dataset.color || this.config.colors[colorIndex % this.config.colors.length];

        // Calculate points
        const points = [];
        const pointCoords = [];

        for (let i = 0; i < numAxes; i++) {
            const value = dataset.values[i] || 0;
            const r = (value / maxValue) * radius;
            const angle = angleSlice * i - Math.PI / 2;
            const x = r * Math.cos(angle);
            const y = r * Math.sin(angle);
            points.push(`${x},${y}`);
            pointCoords.push({ x, y, value, criterion: this.criteria[i] });
        }

        // Draw filled polygon
        const polygon = this._createSVGElement('polygon', {
            points: points.join(' '),
            fill: color,
            'fill-opacity': 0.25,
            stroke: color,
            'stroke-width': 2,
            class: 'radar-polygon'
        });
        g.appendChild(polygon);

        // Draw data points
        pointCoords.forEach((coord, i) => {
            const circle = this._createSVGElement('circle', {
                cx: coord.x,
                cy: coord.y,
                r: 5,
                fill: color,
                stroke: '#1A1A2E',
                'stroke-width': 2,
                class: 'radar-point',
                'data-label': dataset.label,
                'data-criterion': coord.criterion,
                'data-value': coord.value.toFixed(1)
            });

            // Tooltip on hover
            circle.addEventListener('mouseenter', (e) => {
                this._showTooltip(e, `${dataset.label}\n${coord.criterion}: ${coord.value.toFixed(1)}`);
            });
            circle.addEventListener('mouseleave', () => this._hideTooltip());

            g.appendChild(circle);
        });
    }

    /**
     * Draw criterion labels around the chart.
     */
    _drawLabels(g) {
        const { radius, labelOffset } = this.config;
        const numAxes = this.criteria.length;
        if (numAxes < 3) return;

        const angleSlice = (Math.PI * 2) / numAxes;

        for (let i = 0; i < numAxes; i++) {
            const criterion = this.criteria[i];
            const angle = angleSlice * i - Math.PI / 2;
            const x = (radius + labelOffset) * Math.cos(angle);
            const y = (radius + labelOffset) * Math.sin(angle);

            // Determine text anchor based on position
            let textAnchor = 'middle';
            if (x < -10) textAnchor = 'end';
            else if (x > 10) textAnchor = 'start';

            const text = this._createSVGElement('text', {
                x: x,
                y: y,
                'text-anchor': textAnchor,
                'dominant-baseline': 'middle',
                'font-size': '12px',
                'font-weight': '600',
                fill: '#1A1A2E'
            });
            text.textContent = this._capitalize(criterion);
            g.appendChild(text);
        }
    }

    /**
     * Render the legend below the chart.
     */
    _renderLegend() {
        if (!this.legendContainer) return;

        this.legendContainer.innerHTML = '';

        this.datasets.forEach((dataset, idx) => {
            const color = dataset.color || this.config.colors[idx % this.config.colors.length];

            const item = document.createElement('div');
            item.className = 'radar-legend-item';
            item.innerHTML = `
                <span class="radar-legend-color" style="background: ${color}"></span>
                <span class="radar-legend-label">${dataset.label}</span>
            `;
            this.legendContainer.appendChild(item);
        });
    }

    /**
     * Show tooltip at mouse position.
     */
    _showTooltip(event, text) {
        if (!this.tooltip) {
            this.tooltip = document.createElement('div');
            this.tooltip.className = 'radar-tooltip';
            document.body.appendChild(this.tooltip);
        }

        this.tooltip.textContent = text;
        this.tooltip.style.display = 'block';
        this.tooltip.style.left = `${event.pageX + 10}px`;
        this.tooltip.style.top = `${event.pageY - 10}px`;
    }

    /**
     * Hide the tooltip.
     */
    _hideTooltip() {
        if (this.tooltip) {
            this.tooltip.style.display = 'none';
        }
    }

    /**
     * Create an SVG element with attributes.
     */
    _createSVGElement(tag, attrs = {}) {
        const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [key, value] of Object.entries(attrs)) {
            el.setAttribute(key, value);
        }
        return el;
    }

    /**
     * Capitalize first letter of string.
     */
    _capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    /**
     * Clear the chart.
     */
    clear() {
        if (this.svg) {
            this.svg.innerHTML = '';
        }
        if (this.legendContainer) {
            this.legendContainer.innerHTML = '';
        }
        this.datasets = [];
        this.criteria = [];
    }

    /**
     * Get statistics from current data.
     */
    getStats() {
        if (!this.datasets.length) {
            return { total: 0, avgScores: {} };
        }

        const avgScores = {};
        this.datasets.forEach(dataset => {
            const sum = dataset.values.reduce((a, b) => a + b, 0);
            avgScores[dataset.label] = sum / dataset.values.length;
        });

        return {
            total: this.datasets.length,
            avgScores
        };
    }
}
