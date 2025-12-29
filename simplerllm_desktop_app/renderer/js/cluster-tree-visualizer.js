/**
 * Cluster Tree Visualizer for LLM Retrieval
 * Displays hierarchical cluster structure with navigation path highlighting
 */
class ClusterTreeVisualizer {
    constructor(containerId, svgId) {
        this.container = document.getElementById(containerId);
        this.svg = document.getElementById(svgId);
        this.nodes = new Map();  // id -> {element, data, position}
        this.edges = [];
        this.highlightedPath = new Set();  // Cluster IDs in current navigation path

        // Configuration
        this.config = {
            nodeWidth: 200,
            nodeHeight: 70,
            horizontalSpacing: 40,
            verticalSpacing: 100,
            padding: 60,
            animationDuration: 300,
            maxTextLength: 22
        };

        // State
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        this.selectedNodeId = null;
        this.onNodeClick = null;  // Callback for node selection

        // Track nodes per level for layout
        this.nodesPerLevel = new Map();

        this.init();
    }

    init() {
        // Create main group for transformations
        this.mainGroup = this._createSVGElement('g', { id: 'cluster-main-group' });
        this.svg.appendChild(this.mainGroup);

        // Create layers
        this.edgesGroup = this._createSVGElement('g', { id: 'cluster-edges-group' });
        this.nodesGroup = this._createSVGElement('g', { id: 'cluster-nodes-group' });
        this.mainGroup.appendChild(this.edgesGroup);
        this.mainGroup.appendChild(this.nodesGroup);

        // Setup pan & zoom
        this._setupPanZoom();

        // Initial viewBox
        this._updateViewBox();
    }

    /**
     * Build tree from cluster data
     * @param {Object} treeData - {nodes: [], edges: [], root_ids: []}
     */
    buildTree(treeData) {
        this.clear();

        if (!treeData || !treeData.nodes) return;

        const { nodes, edges, root_ids } = treeData;

        // Calculate positions for all nodes using level-based layout
        const positions = this._calculateAllPositions(nodes, edges, root_ids);

        // Create nodes
        for (const node of nodes) {
            const position = positions.get(node.id);
            if (position) {
                this._createNode(node, position);
            }
        }

        // Create edges
        for (const edge of edges) {
            this._createEdge(edge.from, edge.to);
        }

        this._updateViewBox();
    }

    _calculateAllPositions(nodes, edges, rootIds) {
        const positions = new Map();
        const nodeById = new Map(nodes.map(n => [n.id, n]));

        // Build parent-child relationships
        const children = new Map();
        const parents = new Map();

        for (const edge of edges) {
            if (!children.has(edge.from)) children.set(edge.from, []);
            children.get(edge.from).push(edge.to);
            parents.set(edge.to, edge.from);
        }

        // Group nodes by level
        const nodesByLevel = new Map();
        for (const node of nodes) {
            const level = node.level || 0;
            if (!nodesByLevel.has(level)) nodesByLevel.set(level, []);
            nodesByLevel.get(level).push(node.id);
            this.nodesPerLevel.set(level, nodesByLevel.get(level));
        }

        // Calculate positions level by level
        const levels = Array.from(nodesByLevel.keys()).sort((a, b) => b - a);  // Top-down (high to low level)

        for (const level of levels) {
            const nodeIds = nodesByLevel.get(level);
            const y = this.config.padding + (levels.indexOf(level)) * (this.config.nodeHeight + this.config.verticalSpacing);

            // Calculate x positions
            const totalWidth = nodeIds.length * this.config.nodeWidth + (nodeIds.length - 1) * this.config.horizontalSpacing;
            let startX = this.config.padding;

            // If not root level, try to center under parent
            if (level < Math.max(...levels)) {
                // Group by parent and center each group
                const groups = new Map();
                for (const nodeId of nodeIds) {
                    const parentId = parents.get(nodeId) || 'root';
                    if (!groups.has(parentId)) groups.set(parentId, []);
                    groups.get(parentId).push(nodeId);
                }

                let currentX = this.config.padding;
                for (const [parentId, childIds] of groups) {
                    const parentPos = positions.get(parentId);
                    const groupWidth = childIds.length * this.config.nodeWidth + (childIds.length - 1) * this.config.horizontalSpacing;

                    let groupStartX = currentX;
                    if (parentPos) {
                        const parentCenterX = parentPos.x + this.config.nodeWidth / 2;
                        groupStartX = Math.max(currentX, parentCenterX - groupWidth / 2);
                    }

                    for (let i = 0; i < childIds.length; i++) {
                        positions.set(childIds[i], {
                            x: groupStartX + i * (this.config.nodeWidth + this.config.horizontalSpacing),
                            y
                        });
                    }

                    currentX = groupStartX + groupWidth + this.config.horizontalSpacing * 2;
                }
            } else {
                // Root level - simple horizontal layout
                for (let i = 0; i < nodeIds.length; i++) {
                    positions.set(nodeIds[i], {
                        x: startX + i * (this.config.nodeWidth + this.config.horizontalSpacing),
                        y
                    });
                }
            }
        }

        return positions;
    }

    _createNode(nodeData, position) {
        const { id, name, description, keywords, tags, chunk_count, is_root, is_leaf, level } = nodeData;
        const { nodeWidth, nodeHeight, maxTextLength } = this.config;

        // Get color based on node type
        const colors = this._getNodeColors(is_root, is_leaf, level);

        // Create group
        const group = this._createSVGElement('g', {
            class: 'cluster-node',
            transform: `translate(${position.x}, ${position.y})`,
            'data-id': id
        });

        // Shadow
        const shadow = this._createSVGElement('rect', {
            x: 3,
            y: 3,
            width: nodeWidth,
            height: nodeHeight,
            rx: 10,
            ry: 10,
            fill: 'rgba(0,0,0,0.12)'
        });

        // Background rectangle
        const rect = this._createSVGElement('rect', {
            width: nodeWidth,
            height: nodeHeight,
            rx: 10,
            ry: 10,
            fill: colors.bg,
            stroke: '#1A1A2E',
            'stroke-width': 2.5,
            class: 'cluster-node-bg'
        });

        // Truncate name
        let displayName = name || `Cluster ${id}`;
        if (displayName.length > maxTextLength) {
            displayName = displayName.substring(0, maxTextLength - 2) + '...';
        }

        // Name text
        const nameText = this._createSVGElement('text', {
            x: 10,
            y: 24,
            class: 'cluster-node-name',
            fill: colors.text
        });
        nameText.textContent = displayName;

        // Keywords text (smaller)
        const keywordsText = this._createSVGElement('text', {
            x: 10,
            y: 42,
            class: 'cluster-node-keywords',
            fill: colors.textMuted
        });
        const keywordStr = keywords && keywords.length > 0 ? keywords.slice(0, 3).join(', ') : '';
        keywordsText.textContent = keywordStr.length > 28 ? keywordStr.substring(0, 26) + '...' : keywordStr;

        // Chunk count badge
        const badgeGroup = this._createSVGElement('g', {
            transform: `translate(${nodeWidth - 45}, 8)`
        });

        const badgeBg = this._createSVGElement('rect', {
            width: 38,
            height: 20,
            rx: 10,
            fill: '#1A1A2E'
        });

        const badgeText = this._createSVGElement('text', {
            x: 19,
            y: 14,
            'text-anchor': 'middle',
            class: 'cluster-node-badge',
            fill: colors.bg
        });
        badgeText.textContent = `${chunk_count}`;

        badgeGroup.appendChild(badgeBg);
        badgeGroup.appendChild(badgeText);

        // Type indicator
        const typeText = this._createSVGElement('text', {
            x: 10,
            y: nodeHeight - 10,
            class: 'cluster-node-type',
            fill: colors.textMuted
        });
        typeText.textContent = is_root ? 'Root' : is_leaf ? 'Leaf' : 'Branch';

        // Assemble node
        group.appendChild(shadow);
        group.appendChild(rect);
        group.appendChild(nameText);
        group.appendChild(keywordsText);
        group.appendChild(badgeGroup);
        group.appendChild(typeText);

        // Click handler
        group.addEventListener('click', (e) => {
            e.stopPropagation();
            this._handleNodeClick(id);
        });

        // Hover effects
        group.addEventListener('mouseenter', () => {
            rect.style.filter = 'brightness(1.1)';
            group.style.cursor = 'pointer';
        });

        group.addEventListener('mouseleave', () => {
            rect.style.filter = '';
        });

        this.nodesGroup.appendChild(group);

        // Store node reference
        this.nodes.set(id, { element: group, data: nodeData, position });

        // Animate entry
        this._animateNodeEntry(group);
    }

    _getNodeColors(isRoot, isLeaf, level) {
        if (isRoot) {
            return {
                bg: '#9B5DE5',      // Purple - root clusters
                text: '#FEFEFE',
                textMuted: 'rgba(254, 254, 254, 0.75)'
            };
        } else if (isLeaf) {
            return {
                bg: '#00D4AA',      // Teal - leaf clusters (contains chunks)
                text: '#1A1A2E',
                textMuted: 'rgba(26, 26, 46, 0.7)'
            };
        } else {
            return {
                bg: '#00BBF9',      // Blue - branch clusters
                text: '#1A1A2E',
                textMuted: 'rgba(26, 26, 46, 0.7)'
            };
        }
    }

    _createEdge(fromId, toId) {
        const fromNode = this.nodes.get(fromId);
        const toNode = this.nodes.get(toId);

        if (!fromNode || !toNode) return;

        const { nodeWidth, nodeHeight } = this.config;

        // Calculate connection points
        const x1 = fromNode.position.x + nodeWidth / 2;
        const y1 = fromNode.position.y + nodeHeight;
        const x2 = toNode.position.x + nodeWidth / 2;
        const y2 = toNode.position.y;

        // Determine if edge is highlighted
        const isHighlighted = this.highlightedPath.has(fromId) && this.highlightedPath.has(toId);

        // Create curved path
        const midY = (y1 + y2) / 2;
        const path = this._createSVGElement('path', {
            d: `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`,
            stroke: isHighlighted ? '#FFE156' : '#888',
            'stroke-width': isHighlighted ? 4 : 2,
            fill: 'none',
            class: 'cluster-edge',
            opacity: isHighlighted ? 1 : 0.5
        });

        this.edgesGroup.appendChild(path);
        this.edges.push({ from: fromId, to: toId, element: path });
    }

    /**
     * Highlight navigation path
     * @param {string[]} clusterIds - Array of cluster IDs in the path
     */
    highlightPath(clusterIds) {
        this.highlightedPath = new Set(clusterIds);

        // Update node styles
        for (const [id, node] of this.nodes) {
            const isInPath = this.highlightedPath.has(id);
            const bg = node.element.querySelector('.cluster-node-bg');

            if (isInPath) {
                bg.setAttribute('stroke', '#FFE156');
                bg.setAttribute('stroke-width', '4');
                node.element.classList.add('highlighted');
            } else {
                bg.setAttribute('stroke', '#1A1A2E');
                bg.setAttribute('stroke-width', '2.5');
                node.element.classList.remove('highlighted');
            }
        }

        // Rebuild edges with new highlight state
        this._rebuildEdges();
    }

    _rebuildEdges() {
        // Store edge data
        const edgeData = this.edges.map(e => ({ from: e.from, to: e.to }));

        // Clear edges
        this.edgesGroup.innerHTML = '';
        this.edges = [];

        // Recreate edges
        for (const edge of edgeData) {
            this._createEdge(edge.from, edge.to);
        }
    }

    clearHighlight() {
        this.highlightedPath.clear();

        // Reset node styles
        for (const [id, node] of this.nodes) {
            const bg = node.element.querySelector('.cluster-node-bg');
            bg.setAttribute('stroke', '#1A1A2E');
            bg.setAttribute('stroke-width', '2.5');
            node.element.classList.remove('highlighted');
        }

        this._rebuildEdges();
    }

    _createSVGElement(tag, attrs = {}) {
        const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [key, value] of Object.entries(attrs)) {
            el.setAttribute(key, value);
        }
        return el;
    }

    _animateNodeEntry(nodeGroup) {
        nodeGroup.style.opacity = '0';
        nodeGroup.style.transform += ' scale(0.9)';

        requestAnimationFrame(() => {
            nodeGroup.style.transition = `opacity ${this.config.animationDuration}ms ease, transform ${this.config.animationDuration}ms ease`;
            nodeGroup.style.opacity = '1';

            const currentTransform = nodeGroup.getAttribute('transform');
            nodeGroup.style.transform = '';
            nodeGroup.setAttribute('transform', currentTransform);
        });
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
            this.zoom = Math.max(0.3, Math.min(3, this.zoom * delta));
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
            this.svg.setAttribute('viewBox', '0 0 600 400');
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
        const width = Math.max(600, maxX - minX + padding * 2);
        const height = Math.max(400, maxY - minY + padding * 2);

        this.svg.setAttribute('viewBox', `${minX - padding} ${minY - padding} ${width} ${height}`);
        this.svg.style.width = '100%';
        this.svg.style.height = '100%';
    }

    // Public methods for zoom controls
    zoomIn() {
        this.zoom = Math.min(3, this.zoom * 1.2);
        this._applyTransform();
    }

    zoomOut() {
        this.zoom = Math.max(0.3, this.zoom / 1.2);
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
        this.nodesPerLevel.clear();
        this.highlightedPath.clear();
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
}
