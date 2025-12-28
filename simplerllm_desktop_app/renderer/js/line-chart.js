/**
 * Line Chart Component
 * SVG-based line chart for visualizing score trajectory
 */

class LineChart {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = null;
        this.data = [];
        this.qualityThreshold = null;

        // Chart dimensions
        this.margin = { top: 30, right: 40, bottom: 40, left: 50 };
        this.width = 0;
        this.height = 0;

        // Colors
        this.lineColor = '#00D4AA'; // Teal
        this.pointColor = '#00D4AA';
        this.gridColor = '#E0E0E0';
        this.thresholdColor = '#FF6B9D'; // Pink
        this.textColor = '#333333';

        this.init();
    }

    init() {
        if (!this.container) return;

        // Create SVG element
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', '100%');
        this.svg.setAttribute('height', '100%');
        this.svg.style.minHeight = '250px';
        this.container.appendChild(this.svg);

        // Handle resize
        window.addEventListener('resize', () => this.render());
    }

    setThreshold(threshold) {
        this.qualityThreshold = threshold;
    }

    addPoint(iterationNumber, score) {
        this.data.push({ x: iterationNumber, y: score });
        this.render();
    }

    setData(scores) {
        this.data = scores.map((score, index) => ({
            x: index + 1,
            y: score
        }));
        this.render();
    }

    clear() {
        this.data = [];
        this.qualityThreshold = null;
        if (this.svg) {
            this.svg.innerHTML = '';
        }
    }

    render() {
        if (!this.svg || !this.container) return;

        // Get container dimensions
        const rect = this.container.getBoundingClientRect();
        this.width = rect.width - this.margin.left - this.margin.right;
        this.height = Math.max(200, rect.height - this.margin.top - this.margin.bottom);

        // Clear previous content
        this.svg.innerHTML = '';

        // Create main group with margin offset
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('transform', `translate(${this.margin.left}, ${this.margin.top})`);
        this.svg.appendChild(g);

        // Draw components
        this._drawGrid(g);
        this._drawAxes(g);
        if (this.qualityThreshold) {
            this._drawThresholdLine(g);
        }
        if (this.data.length > 0) {
            this._drawLine(g);
            this._drawPoints(g);
        } else {
            this._drawEmptyState(g);
        }
    }

    _drawGrid(g) {
        const gridGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        gridGroup.setAttribute('class', 'grid');

        // Horizontal grid lines at score levels 2, 4, 6, 8, 10
        for (let score = 2; score <= 10; score += 2) {
            const y = this._scaleY(score);
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', 0);
            line.setAttribute('y1', y);
            line.setAttribute('x2', this.width);
            line.setAttribute('y2', y);
            line.setAttribute('stroke', this.gridColor);
            line.setAttribute('stroke-width', '1');
            line.setAttribute('stroke-dasharray', '4,4');
            line.setAttribute('opacity', '0.5');
            gridGroup.appendChild(line);
        }

        g.appendChild(gridGroup);
    }

    _drawAxes(g) {
        const axesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        axesGroup.setAttribute('class', 'axes');

        // Y-axis
        const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        yAxis.setAttribute('x1', 0);
        yAxis.setAttribute('y1', 0);
        yAxis.setAttribute('x2', 0);
        yAxis.setAttribute('y2', this.height);
        yAxis.setAttribute('stroke', this.textColor);
        yAxis.setAttribute('stroke-width', '2');
        axesGroup.appendChild(yAxis);

        // X-axis
        const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        xAxis.setAttribute('x1', 0);
        xAxis.setAttribute('y1', this.height);
        xAxis.setAttribute('x2', this.width);
        xAxis.setAttribute('y2', this.height);
        xAxis.setAttribute('stroke', this.textColor);
        xAxis.setAttribute('stroke-width', '2');
        axesGroup.appendChild(xAxis);

        // Y-axis labels
        for (let score = 0; score <= 10; score += 2) {
            const y = this._scaleY(score);
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', -10);
            label.setAttribute('y', y + 4);
            label.setAttribute('text-anchor', 'end');
            label.setAttribute('fill', this.textColor);
            label.setAttribute('font-size', '12');
            label.setAttribute('font-family', 'JetBrains Mono, monospace');
            label.textContent = score;
            axesGroup.appendChild(label);
        }

        // Y-axis title
        const yTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        yTitle.setAttribute('x', -this.height / 2);
        yTitle.setAttribute('y', -35);
        yTitle.setAttribute('text-anchor', 'middle');
        yTitle.setAttribute('fill', this.textColor);
        yTitle.setAttribute('font-size', '12');
        yTitle.setAttribute('font-weight', '600');
        yTitle.setAttribute('transform', 'rotate(-90)');
        yTitle.textContent = 'Quality Score';
        axesGroup.appendChild(yTitle);

        // X-axis title
        const xTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        xTitle.setAttribute('x', this.width / 2);
        xTitle.setAttribute('y', this.height + 35);
        xTitle.setAttribute('text-anchor', 'middle');
        xTitle.setAttribute('fill', this.textColor);
        xTitle.setAttribute('font-size', '12');
        xTitle.setAttribute('font-weight', '600');
        xTitle.textContent = 'Iteration';
        axesGroup.appendChild(xTitle);

        // X-axis labels (iteration numbers)
        if (this.data.length > 0) {
            const maxX = Math.max(...this.data.map(d => d.x));
            for (let i = 1; i <= maxX; i++) {
                const x = this._scaleX(i, maxX);
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', x);
                label.setAttribute('y', this.height + 20);
                label.setAttribute('text-anchor', 'middle');
                label.setAttribute('fill', this.textColor);
                label.setAttribute('font-size', '12');
                label.setAttribute('font-family', 'JetBrains Mono, monospace');
                label.textContent = i;
                axesGroup.appendChild(label);
            }
        }

        g.appendChild(axesGroup);
    }

    _drawThresholdLine(g) {
        const y = this._scaleY(this.qualityThreshold);

        const thresholdGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        thresholdGroup.setAttribute('class', 'threshold');

        // Threshold line
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', 0);
        line.setAttribute('y1', y);
        line.setAttribute('x2', this.width);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', this.thresholdColor);
        line.setAttribute('stroke-width', '2');
        line.setAttribute('stroke-dasharray', '8,4');
        thresholdGroup.appendChild(line);

        // Threshold label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', this.width + 5);
        label.setAttribute('y', y + 4);
        label.setAttribute('fill', this.thresholdColor);
        label.setAttribute('font-size', '11');
        label.setAttribute('font-weight', '600');
        label.textContent = `Target: ${this.qualityThreshold}`;
        thresholdGroup.appendChild(label);

        g.appendChild(thresholdGroup);
    }

    _drawLine(g) {
        if (this.data.length < 2) return;

        const maxX = Math.max(...this.data.map(d => d.x));
        const lineGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        lineGroup.setAttribute('class', 'line');

        // Create path
        let pathData = '';
        this.data.forEach((point, index) => {
            const x = this._scaleX(point.x, maxX);
            const y = this._scaleY(point.y);
            if (index === 0) {
                pathData += `M ${x} ${y}`;
            } else {
                pathData += ` L ${x} ${y}`;
            }
        });

        // Main line
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', pathData);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', this.lineColor);
        path.setAttribute('stroke-width', '3');
        path.setAttribute('stroke-linecap', 'round');
        path.setAttribute('stroke-linejoin', 'round');

        // Add animation
        const pathLength = path.getTotalLength ? path.getTotalLength() : 1000;
        path.style.strokeDasharray = pathLength;
        path.style.strokeDashoffset = pathLength;
        path.style.animation = 'drawLine 0.5s ease-out forwards';

        lineGroup.appendChild(path);

        // Add gradient fill under the line
        const areaPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const lastPoint = this.data[this.data.length - 1];
        const firstPoint = this.data[0];
        const areaData = pathData +
            ` L ${this._scaleX(lastPoint.x, maxX)} ${this.height}` +
            ` L ${this._scaleX(firstPoint.x, maxX)} ${this.height} Z`;
        areaPath.setAttribute('d', areaData);
        areaPath.setAttribute('fill', this.lineColor);
        areaPath.setAttribute('fill-opacity', '0.1');
        lineGroup.insertBefore(areaPath, path);

        g.appendChild(lineGroup);
    }

    _drawPoints(g) {
        const maxX = Math.max(...this.data.map(d => d.x));
        const pointsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        pointsGroup.setAttribute('class', 'points');

        this.data.forEach((point, index) => {
            const x = this._scaleX(point.x, maxX);
            const y = this._scaleY(point.y);

            // Outer circle (border)
            const outerCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            outerCircle.setAttribute('cx', x);
            outerCircle.setAttribute('cy', y);
            outerCircle.setAttribute('r', '8');
            outerCircle.setAttribute('fill', 'white');
            outerCircle.setAttribute('stroke', this.pointColor);
            outerCircle.setAttribute('stroke-width', '3');
            outerCircle.style.cursor = 'pointer';
            pointsGroup.appendChild(outerCircle);

            // Inner circle
            const innerCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            innerCircle.setAttribute('cx', x);
            innerCircle.setAttribute('cy', y);
            innerCircle.setAttribute('r', '4');
            innerCircle.setAttribute('fill', this.pointColor);
            innerCircle.style.cursor = 'pointer';
            pointsGroup.appendChild(innerCircle);

            // Score label above point
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', x);
            label.setAttribute('y', y - 15);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('fill', this.textColor);
            label.setAttribute('font-size', '12');
            label.setAttribute('font-weight', '600');
            label.setAttribute('font-family', 'JetBrains Mono, monospace');
            label.textContent = point.y.toFixed(1);
            pointsGroup.appendChild(label);

            // Highlight last point
            if (index === this.data.length - 1) {
                outerCircle.setAttribute('r', '10');
                outerCircle.setAttribute('stroke-width', '4');
                innerCircle.setAttribute('r', '5');
                label.setAttribute('font-size', '14');
            }
        });

        g.appendChild(pointsGroup);
    }

    _drawEmptyState(g) {
        const emptyGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        emptyGroup.setAttribute('class', 'empty-state');

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', this.width / 2);
        text.setAttribute('y', this.height / 2);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#999999');
        text.setAttribute('font-size', '14');
        text.textContent = 'Score trajectory will appear here';
        emptyGroup.appendChild(text);

        g.appendChild(emptyGroup);
    }

    _scaleX(value, maxX) {
        const padding = 40;
        const availableWidth = this.width - padding * 2;
        if (maxX <= 1) return this.width / 2;
        return padding + ((value - 1) / (maxX - 1)) * availableWidth;
    }

    _scaleY(value) {
        // Scale from 0-10 to height-0 (inverted for SVG coordinates)
        return this.height - (value / 10) * this.height;
    }
}

// Add CSS animation for line drawing
const style = document.createElement('style');
style.textContent = `
    @keyframes drawLine {
        to {
            stroke-dashoffset: 0;
        }
    }
`;
document.head.appendChild(style);
