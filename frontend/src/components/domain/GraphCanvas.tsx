import { useCallback, useRef, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { NODE_TYPE_COLORS, RELATION_COLORS } from '@/lib/constants';
import type { GraphNodeItem, GraphEdgeItem } from '@/api/hooks/useGraph';

interface GraphCanvasProps {
  nodes: GraphNodeItem[];
  edges: GraphEdgeItem[];
  selectedNodeId?: string;
  onNodeClick?: (nodeId: string) => void;
  width?: number;
  height?: number;
}

interface GraphNode {
  id: string;
  name: string;
  node_type: string;
  metadata: Record<string, unknown>;
}

interface GraphLink {
  source: string;
  target: string;
  relation: string;
  confidence: number;
}

export function GraphCanvas({
  nodes,
  edges,
  selectedNodeId,
  onNodeClick,
  width,
  height = 500,
}: GraphCanvasProps) {
  const fgRef = useRef<{ centerAt: (x: number, y: number, ms: number) => void }>(null);

  const graphData = useMemo(() => {
    const graphNodes: GraphNode[] = nodes.map((n) => ({
      id: n.node_id,
      name: n.name,
      node_type: n.node_type,
      metadata: n.metadata,
    }));

    const nodeIds = new Set(nodes.map((n) => n.node_id));
    const graphLinks: GraphLink[] = edges
      .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        relation: e.relation,
        confidence: e.confidence,
      }));

    return { nodes: graphNodes, links: graphLinks };
  }, [nodes, edges]);

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      onNodeClick?.(node.id);
    },
    [onNodeClick],
  );

  const nodeColor = useCallback(
    (node: GraphNode) => {
      if (node.id === selectedNodeId) return '#f59e0b';
      return NODE_TYPE_COLORS[node.node_type] ?? '#6b7280';
    },
    [selectedNodeId],
  );

  const linkColor = useCallback((link: GraphLink) => {
    return RELATION_COLORS[link.relation] ?? '#4b5563';
  }, []);

  if (nodes.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-border bg-card" style={{ height }}>
        <p className="text-sm text-muted-foreground">No graph data available</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-[#0f1117]">
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={width}
        height={height}
        nodeLabel={(node: GraphNode) => `${node.name} (${node.node_type})`}
        nodeColor={nodeColor}
        nodeRelSize={6}
        nodeVal={(node: GraphNode) => (node.id === selectedNodeId ? 2 : 1)}
        linkColor={linkColor}
        linkWidth={(link: GraphLink) => Math.max(1, link.confidence * 3)}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={0.9}
        linkLabel={(link: GraphLink) => link.relation}
        onNodeClick={handleNodeClick}
        cooldownTicks={100}
        backgroundColor="#0f1117"
        nodeCanvasObjectMode={() => 'after'}
        nodeCanvasObject={(node: GraphNode & { x: number; y: number }, ctx: CanvasRenderingContext2D) => {
          const label = node.name;
          const fontSize = 10;
          ctx.font = `${fontSize}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillStyle = node.id === selectedNodeId ? '#fbbf24' : '#9ca3af';
          ctx.fillText(label, node.x, node.y + 8);
        }}
      />
    </div>
  );
}
