/**
 * Tree Visualizer for Brainstorm Ideas
 * Pure SVG implementation with animations and interactivity
 */
class TreeVisualizer {
    constructor(containerId, svgId) {
        this.container = document.getElementById(containerId);
        this.svg = document.getElementById(svgId);
        this.nodes = new Map();  // id -> {element, data, position}
        this.edges = [];

        // Configuration
        this.config = {
            nodeWidth: 240,
            nodeHeight: 80,
            horizontalSpacing: 60,
            verticalSpacing: 120,
            padding: 80,
            animationDuration: 300,
            maxTextLength: 28
        };

        // State
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        this.selectedNodeId = null;
        this.onNodeClick = null;  // Callback for node selection

        // Track nodes per depth for layout
        this.nodesPerDepth = new Map();

        // Drag state (centralized to avoid multiple event listeners)
        this.dragState = {
            isDragging: false,
            nodeId: null,
            element: null,
            startX: 0,
            startY: 0,
            originalX: 0,
            originalY: 0,
            hasMoved: false  // Track if actual drag occurred (vs just click)
        };

        this.init();
    }

    init() {
        // Create main group for transformations
        this.mainGroup = this._createSVGElement('g', { id: 'main-group' });
        this.svg.appendChild(this.mainGroup);

        // Create layers
        this.edgesGroup = this._createSVGElement('g', { id: 'edges-group' });
        this.nodesGroup = this._createSVGElement('g', { id: 'nodes-group' });
        this.mainGroup.appendChild(this.edgesGroup);
        this.mainGroup.appendChild(this.nodesGroup);

        // Setup pan & zoom
        this._setupPanZoom();

        // Setup centralized drag handlers
        this._setupDragHandlers();

        // Initial viewBox
        this._updateViewBox();
    }

    _setupDragHandlers() {
        // Centralized mousemove handler for all node drags
        document.addEventListener('mousemove', (e) => {
            if (!this.dragState.isDragging) return;

            const dx = (e.clientX / this.zoom) - this.dragState.startX;
            const dy = (e.clientY / this.zoom) - this.dragState.startY;

            // Only consider it a drag if moved more than 5 pixels
            if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                this.dragState.hasMoved = true;
            }

            const newX = this.dragState.originalX + dx;
            const newY = this.dragState.originalY + dy;

            const node = this.nodes.get(this.dragState.nodeId);
            if (node) {
                node.position.x = newX;
                node.position.y = newY;
                this.dragState.element.setAttribute('transform', `translate(${newX}, ${newY})`);

                // Update connected edges
                this._updateEdgesForNode(this.dragState.nodeId);
            }
        });

        // Centralized mouseup handler
        document.addEventListener('mouseup', () => {
            if (this.dragState.isDragging) {
                this.dragState.isDragging = false;
                if (this.dragState.element) {
                    this.dragState.element.style.cursor = 'grab';
                }
                this._updateViewBox();

                // Reset hasMoved after a short delay so click handler can check it
                setTimeout(() => {
                    this.dragState.hasMoved = false;
                }, 10);
            }
        });
    }

    /**
     * Add a new idea node to the tree
     */
    addNode(idea) {
        const { id, text, quality_score, depth, parent_id } = idea;

        // Track nodes per depth
        if (!this.nodesPerDepth.has(depth)) {
            this.nodesPerDepth.set(depth, []);
        }
        this.nodesPerDepth.get(depth).push(id);

        // Calculate position
        const position = this._calculateNodePosition(id, depth, parent_id);

        // Create node element
        const nodeGroup = this._createNodeElement(idea, position);
        this.nodesGroup.appendChild(nodeGroup);

        // Store node reference
        this.nodes.set(id, { element: nodeGroup, data: idea, position });

        // Create edge to parent
        if (parent_id && this.nodes.has(parent_id)) {
            const parentNode = this.nodes.get(parent_id);
            const edge = this._createEdge(parentNode.position, position, quality_score);
            this.edgesGroup.appendChild(edge);
            this.edges.push({ from: parent_id, to: id, element: edge });
        }

        // Animate entry
        this._animateNodeEntry(nodeGroup);

        // Update layout and viewBox
        this._rebalanceDepth(depth);
        this._updateViewBox();
    }

    _createSVGElement(tag, attrs = {}) {
        const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [key, value] of Object.entries(attrs)) {
            el.setAttribute(key, value);
        }
        return el;
    }

    _createNodeElement(idea, position) {
        const { id, text, quality_score, depth, reasoning } = idea;
        const { nodeWidth, nodeHeight, maxTextLength } = this.config;

        // Get color based on quality score
        const colors = this._getScoreColors(quality_score);

        // Create group
        const group = this._createSVGElement('g', {
            class: 'tree-node',
            transform: `translate(${position.x}, ${position.y})`,
            'data-id': id
        });

        // Create clipPath for text
        const clipId = `clip-${id.replace(/[^a-zA-Z0-9]/g, '-')}`;
        const defs = this._createSVGElement('defs');
        const clipPath = this._createSVGElement('clipPath', { id: clipId });
        const clipRect = this._createSVGElement('rect', {
            x: 10,
            y: 8,
            width: nodeWidth - 60,  // Leave space for score badge
            height: 45
        });
        clipPath.appendChild(clipRect);
        defs.appendChild(clipPath);
        group.appendChild(defs);

        // Shadow
        const shadow = this._createSVGElement('rect', {
            x: 4,
            y: 4,
            width: nodeWidth,
            height: nodeHeight,
            rx: 12,
            ry: 12,
            fill: 'rgba(0,0,0,0.15)'
        });

        // Background rectangle
        const rect = this._createSVGElement('rect', {
            width: nodeWidth,
            height: nodeHeight,
            rx: 12,
            ry: 12,
            fill: colors.bg,
            stroke: '#1A1A2E',
            'stroke-width': 3,
            class: 'node-bg'
        });

        // Truncate text - split into two lines if needed
        const maxCharsPerLine = Math.floor(maxTextLength * 0.9);
        let line1 = text.substring(0, maxCharsPerLine);
        let line2 = '';

        if (text.length > maxCharsPerLine) {
            // Find a good break point (space)
            const breakPoint = line1.lastIndexOf(' ');
            if (breakPoint > maxCharsPerLine * 0.5) {
                line1 = text.substring(0, breakPoint);
                line2 = text.substring(breakPoint + 1, breakPoint + 1 + maxCharsPerLine);
                if (text.length > breakPoint + 1 + maxCharsPerLine) {
                    line2 = line2.substring(0, line2.length - 3) + '...';
                }
            } else {
                line1 = text.substring(0, maxCharsPerLine - 3) + '...';
            }
        }

        // Text container with clip
        const textGroup = this._createSVGElement('g', {
            'clip-path': `url(#${clipId})`
        });

        // Main text - line 1
        const textEl1 = this._createSVGElement('text', {
            x: 12,
            y: 26,
            class: 'node-text',
            fill: colors.text
        });
        textEl1.textContent = line1;
        textGroup.appendChild(textEl1);

        // Main text - line 2 (if exists)
        if (line2) {
            const textEl2 = this._createSVGElement('text', {
                x: 12,
                y: 44,
                class: 'node-text',
                fill: colors.text
            });
            textEl2.textContent = line2;
            textGroup.appendChild(textEl2);
        }

        // Score badge
        const scoreGroup = this._createSVGElement('g', {
            transform: `translate(${nodeWidth - 48}, 10)`
        });

        const scoreBg = this._createSVGElement('rect', {
            width: 40,
            height: 24,
            rx: 12,
            fill: '#1A1A2E'
        });

        const scoreText = this._createSVGElement('text', {
            x: 20,
            y: 17,
            'text-anchor': 'middle',
            class: 'node-score',
            fill: colors.bg
        });
        scoreText.textContent = quality_score.toFixed(1);

        scoreGroup.appendChild(scoreBg);
        scoreGroup.appendChild(scoreText);

        // Depth indicator
        const depthGroup = this._createSVGElement('g', {
            transform: `translate(12, ${nodeHeight - 22})`
        });

        const depthText = this._createSVGElement('text', {
            x: 0,
            y: 12,
            class: 'node-depth',
            fill: colors.textMuted
        });
        depthText.textContent = `Depth ${depth}`;

        depthGroup.appendChild(depthText);

        // Assemble node
        group.appendChild(shadow);
        group.appendChild(rect);
        group.appendChild(textGroup);
        group.appendChild(scoreGroup);
        group.appendChild(depthGroup);

        // Click handler (only fires if not dragged)
        group.addEventListener('click', (e) => {
            e.stopPropagation();
            // Only handle click if we didn't just drag
            if (!this.dragState.hasMoved) {
                this._handleNodeClick(id);
            }
        });

        // Hover effects
        group.addEventListener('mouseenter', () => {
            rect.style.filter = 'brightness(1.1)';
            group.style.cursor = 'grab';
        });

        group.addEventListener('mouseleave', () => {
            rect.style.filter = '';
        });

        // Drag functionality - only mousedown, handlers are centralized
        this._makeDraggable(group, id);

        return group;
    }

    _makeDraggable(element, nodeId) {
        element.addEventListener('mousedown', (e) => {
            // Only start drag on left mouse button
            if (e.button !== 0) return;

            const node = this.nodes.get(nodeId);
            if (!node) return;

            // Set centralized drag state
            this.dragState.isDragging = true;
            this.dragState.nodeId = nodeId;
            this.dragState.element = element;
            this.dragState.originalX = node.position.x;
            this.dragState.originalY = node.position.y;
            this.dragState.startX = e.clientX / this.zoom;
            this.dragState.startY = e.clientY / this.zoom;
            this.dragState.hasMoved = false;

            element.style.cursor = 'grabbing';

            e.stopPropagation();
            e.preventDefault();
        });
    }

    _getScoreColors(score) {
        if (score >= 8) {
            return {
                bg: '#00D4AA',      // Teal - high quality
                text: '#1A1A2E',
                textMuted: 'rgba(26, 26, 46, 0.7)'
            };
        } else if (score >= 5) {
            return {
                bg: '#FFE156',      // Yellow - medium quality
                text: '#1A1A2E',
                textMuted: 'rgba(26, 26, 46, 0.7)'
            };
        } else {
            return {
                bg: '#FF6B9D',      // Pink - low quality
                text: '#FEFEFE',
                textMuted: 'rgba(254, 254, 254, 0.7)'
            };
        }
    }

    _calculateNodePosition(id, depth, parentId) {
        const { nodeWidth, nodeHeight, horizontalSpacing, verticalSpacing, padding } = this.config;

        // Get siblings at this depth
        const siblingsAtDepth = this.nodesPerDepth.get(depth) || [];
        const siblingIndex = siblingsAtDepth.indexOf(id);

        // Calculate vertical position based on depth
        const y = padding + depth * (nodeHeight + verticalSpacing);

        // Calculate horizontal position
        // If has parent, center children under parent
        if (parentId && this.nodes.has(parentId)) {
            const parentNode = this.nodes.get(parentId);
            const parentX = parentNode.position.x;

            // Get all children of this parent at this depth
            const childrenOfParent = siblingsAtDepth.filter(nodeId => {
                const node = this.nodes.get(nodeId);
                return node && node.data.parent_id === parentId;
            });

            // Add current node if not yet in nodes map
            if (!this.nodes.has(id)) {
                childrenOfParent.push(id);
            }

            const childIndex = childrenOfParent.indexOf(id);
            const totalChildren = childrenOfParent.length;

            // Calculate x position centered under parent
            const totalWidth = totalChildren * nodeWidth + (totalChildren - 1) * horizontalSpacing;
            const startX = parentX + (nodeWidth / 2) - (totalWidth / 2);
            const x = startX + childIndex * (nodeWidth + horizontalSpacing);

            return { x, y };
        } else {
            // Root level - just space out horizontally
            const x = padding + siblingIndex * (nodeWidth + horizontalSpacing);
            return { x, y };
        }
    }

    _createEdge(fromPos, toPos, score) {
        const { nodeWidth, nodeHeight } = this.config;
        const colors = this._getScoreColors(score);

        // Calculate connection points
        const x1 = fromPos.x + nodeWidth / 2;
        const y1 = fromPos.y + nodeHeight;
        const x2 = toPos.x + nodeWidth / 2;
        const y2 = toPos.y;

        // Create curved path (bezier)
        const midY = (y1 + y2) / 2;
        const path = this._createSVGElement('path', {
            d: `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`,
            stroke: colors.bg,
            'stroke-width': 3,
            fill: 'none',
            class: 'tree-edge',
            opacity: 0.8
        });

        // Animate the edge
        const length = path.getTotalLength ? path.getTotalLength() : 100;
        path.style.strokeDasharray = length;
        path.style.strokeDashoffset = length;
        path.style.transition = `stroke-dashoffset ${this.config.animationDuration}ms ease-out`;

        requestAnimationFrame(() => {
            path.style.strokeDashoffset = 0;
        });

        return path;
    }

    _animateNodeEntry(nodeGroup) {
        nodeGroup.style.opacity = '0';
        nodeGroup.style.transform += ' scale(0.8)';

        requestAnimationFrame(() => {
            nodeGroup.style.transition = `opacity ${this.config.animationDuration}ms ease, transform ${this.config.animationDuration}ms ease`;
            nodeGroup.style.opacity = '1';

            // Extract current translate and add scale
            const currentTransform = nodeGroup.getAttribute('transform');
            nodeGroup.style.transform = '';
            nodeGroup.setAttribute('transform', currentTransform);
        });
    }

    _rebalanceDepth(depth) {
        const nodeIds = this.nodesPerDepth.get(depth) || [];
        if (nodeIds.length <= 1) return;

        // Group nodes by parent
        const nodesByParent = new Map();
        for (const nodeId of nodeIds) {
            const node = this.nodes.get(nodeId);
            if (!node) continue;

            const parentId = node.data.parent_id || 'root';
            if (!nodesByParent.has(parentId)) {
                nodesByParent.set(parentId, []);
            }
            nodesByParent.get(parentId).push(nodeId);
        }

        // Sort parent groups by parent's X position for consistent layout
        const sortedParentGroups = Array.from(nodesByParent.entries()).sort((a, b) => {
            if (a[0] === 'root') return -1;
            if (b[0] === 'root') return 1;
            const parentA = this.nodes.get(a[0]);
            const parentB = this.nodes.get(b[0]);
            return (parentA?.position.x || 0) - (parentB?.position.x || 0);
        });

        // Extra spacing between sibling groups
        const groupSpacing = this.config.horizontalSpacing * 2;
        let currentX = this.config.padding;

        for (const [parentId, children] of sortedParentGroups) {
            const parentNode = parentId !== 'root' ? this.nodes.get(parentId) : null;

            // Calculate width needed for this group
            const groupWidth = children.length * this.config.nodeWidth +
                             (children.length - 1) * this.config.horizontalSpacing;

            // Determine starting X for this group
            let groupStartX;
            if (parentNode) {
                // Try to center under parent, but don't overlap with previous groups
                const parentCenterX = parentNode.position.x + this.config.nodeWidth / 2;
                const centeredStartX = parentCenterX - groupWidth / 2;
                groupStartX = Math.max(currentX, centeredStartX);
            } else {
                groupStartX = currentX;
            }

            // Position each child in this group
            for (let i = 0; i < children.length; i++) {
                const nodeId = children[i];
                const node = this.nodes.get(nodeId);
                if (!node) continue;

                const newX = groupStartX + i * (this.config.nodeWidth + this.config.horizontalSpacing);

                // Update position
                node.position.x = newX;
                node.element.setAttribute('transform', `translate(${newX}, ${node.position.y})`);

                // Update edges connected to this node
                this._updateEdgesForNode(nodeId);
            }

            // Update currentX for next group
            currentX = groupStartX + groupWidth + groupSpacing;
        }
    }

    _updateEdgesForNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return;

        for (const edge of this.edges) {
            if (edge.to === nodeId || edge.from === nodeId) {
                const fromNode = this.nodes.get(edge.from);
                const toNode = this.nodes.get(edge.to);

                if (fromNode && toNode) {
                    const { nodeWidth, nodeHeight } = this.config;
                    const x1 = fromNode.position.x + nodeWidth / 2;
                    const y1 = fromNode.position.y + nodeHeight;
                    const x2 = toNode.position.x + nodeWidth / 2;
                    const y2 = toNode.position.y;
                    const midY = (y1 + y2) / 2;

                    edge.element.setAttribute('d',
                        `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`
                    );
                }
            }
        }
    }

    _handleNodeClick(nodeId) {
        // Update selection state
        if (this.selectedNodeId) {
            const prevNode = this.nodes.get(this.selectedNodeId);
            if (prevNode) {
                prevNode.element.classList.remove('selected');
            }
        }

        this.selectedNodeId = nodeId;
        const node = this.nodes.get(nodeId);
        if (node) {
            node.element.classList.add('selected');
        }

        // Call callback
        if (this.onNodeClick) {
            this.onNodeClick(nodeId, node ? node.data : null);
        }
    }

    _setupPanZoom() {
        let isPanning = false;
        let startPoint = { x: 0, y: 0 };

        this.svg.addEventListener('mousedown', (e) => {
            if (e.target === this.svg || e.target.tagName === 'svg') {
                isPanning = true;
                startPoint = { x: e.clientX - this.pan.x, y: e.clientY - this.pan.y };
                this.svg.style.cursor = 'grabbing';
            }
        });

        this.svg.addEventListener('mousemove', (e) => {
            if (!isPanning) return;
            this.pan.x = e.clientX - startPoint.x;
            this.pan.y = e.clientY - startPoint.y;
            this._applyTransform();
        });

        this.svg.addEventListener('mouseup', () => {
            isPanning = false;
            this.svg.style.cursor = '';
        });

        this.svg.addEventListener('mouseleave', () => {
            isPanning = false;
            this.svg.style.cursor = '';
        });

        // Zoom with wheel
        this.svg.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoom = Math.max(0.2, Math.min(5, this.zoom * delta));
            this._applyTransform();
        });
    }

    _applyTransform() {
        this.mainGroup.setAttribute('transform',
            `translate(${this.pan.x}, ${this.pan.y}) scale(${this.zoom})`
        );
    }

    _updateViewBox() {
        if (this.nodes.size === 0) {
            this.svg.setAttribute('viewBox', '0 0 800 600');
            return;
        }

        // Calculate bounds
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (const [id, node] of this.nodes) {
            minX = Math.min(minX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxX = Math.max(maxX, node.position.x + this.config.nodeWidth);
            maxY = Math.max(maxY, node.position.y + this.config.nodeHeight);
        }

        const padding = this.config.padding;
        const width = Math.max(800, maxX - minX + padding * 2);
        const height = Math.max(600, maxY - minY + padding * 2);

        this.svg.setAttribute('viewBox', `${minX - padding} ${minY - padding} ${width} ${height}`);
        this.svg.style.width = '100%';
        this.svg.style.height = '100%';
    }

    // Public methods for zoom controls
    zoomIn() {
        this.zoom = Math.min(5, this.zoom * 1.2);
        this._applyTransform();
    }

    zoomOut() {
        this.zoom = Math.max(0.2, this.zoom / 1.2);
        this._applyTransform();
    }

    resetView() {
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        this._applyTransform();
        this._updateViewBox();
    }

    /**
     * Clear all nodes and edges
     */
    clear() {
        this.nodes.clear();
        this.edges = [];
        this.nodesPerDepth.clear();
        this.selectedNodeId = null;
        this.nodesGroup.innerHTML = '';
        this.edgesGroup.innerHTML = '';
        this.resetView();
    }

    /**
     * Get node data by ID
     */
    getNode(nodeId) {
        const node = this.nodes.get(nodeId);
        return node ? node.data : null;
    }

    /**
     * Get all nodes
     */
    getAllNodes() {
        return Array.from(this.nodes.values()).map(n => n.data);
    }

    /**
     * Get statistics
     */
    getStats() {
        const allNodes = this.getAllNodes();
        if (allNodes.length === 0) {
            return { total: 0, avgScore: 0, maxDepth: 0 };
        }

        const scores = allNodes.map(n => n.quality_score);
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const maxDepth = Math.max(...allNodes.map(n => n.depth));

        return {
            total: allNodes.length,
            avgScore: avgScore.toFixed(1),
            maxDepth
        };
    }
}
