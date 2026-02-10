import { Network, Tag } from 'lucide-react';
import { cn } from '@/lib/utils';
import { NODE_TYPE_COLORS, RELATION_COLORS } from '@/lib/constants';
import type { GraphNodeItem, GraphEdgeItem } from '@/api/hooks/useGraph';

interface GraphNodePanelProps {
  node: GraphNodeItem;
  edges: GraphEdgeItem[];
  allNodes: GraphNodeItem[];
  onNavigateNode?: (nodeId: string) => void;
}

export function GraphNodePanel({ node, edges, allNodes, onNavigateNode }: GraphNodePanelProps) {
  const nodeMap = new Map(allNodes.map((n) => [n.node_id, n]));

  const outgoing = edges.filter((e) => e.source === node.node_id);
  const incoming = edges.filter((e) => e.target === node.node_id);

  return (
    <div className="space-y-4">
      {/* Node header */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2">
          <Network className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{node.name}</span>
        </div>
        <div className="mt-2 flex items-center gap-2 text-xs">
          <span
            className="rounded-full px-2 py-0.5"
            style={{
              backgroundColor: `${NODE_TYPE_COLORS[node.node_type] ?? '#6b7280'}20`,
              color: NODE_TYPE_COLORS[node.node_type] ?? '#6b7280',
            }}
          >
            {node.node_type}
          </span>
          <span className="font-mono text-muted-foreground">{node.node_id}</span>
        </div>
        {Object.keys(node.metadata).length > 0 && (
          <div className="mt-3 space-y-1">
            {Object.entries(node.metadata).map(([key, value]) => (
              <div key={key} className="flex items-center gap-2 text-xs">
                <span className="text-muted-foreground">{key}:</span>
                <span className="text-foreground">{String(value)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Outgoing edges */}
      {outgoing.length > 0 && (
        <div>
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            Outgoing ({outgoing.length})
          </div>
          <div className="space-y-1">
            {outgoing.map((edge) => {
              const target = nodeMap.get(edge.target);
              return (
                <button
                  key={`${edge.target}-${edge.relation}`}
                  type="button"
                  onClick={() => onNavigateNode?.(edge.target)}
                  className="flex w-full items-center gap-2 rounded border border-border bg-background px-3 py-2 text-xs hover:bg-secondary/50"
                >
                  <span
                    className="rounded px-1.5 py-0.5 text-[10px]"
                    style={{
                      backgroundColor: `${RELATION_COLORS[edge.relation] ?? '#6b7280'}20`,
                      color: RELATION_COLORS[edge.relation] ?? '#6b7280',
                    }}
                  >
                    {edge.relation}
                  </span>
                  <Tag className="h-3 w-3 text-muted-foreground" />
                  <span className="text-foreground">{target?.name ?? edge.target}</span>
                  <span className="ml-auto text-muted-foreground">
                    {(edge.confidence * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Incoming edges */}
      {incoming.length > 0 && (
        <div>
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            Incoming ({incoming.length})
          </div>
          <div className="space-y-1">
            {incoming.map((edge) => {
              const source = nodeMap.get(edge.source);
              return (
                <button
                  key={`${edge.source}-${edge.relation}`}
                  type="button"
                  onClick={() => onNavigateNode?.(edge.source)}
                  className="flex w-full items-center gap-2 rounded border border-border bg-background px-3 py-2 text-xs hover:bg-secondary/50"
                >
                  <span className="text-foreground">{source?.name ?? edge.source}</span>
                  <span
                    className="rounded px-1.5 py-0.5 text-[10px]"
                    style={{
                      backgroundColor: `${RELATION_COLORS[edge.relation] ?? '#6b7280'}20`,
                      color: RELATION_COLORS[edge.relation] ?? '#6b7280',
                    }}
                  >
                    {edge.relation}
                  </span>
                  <span className="ml-auto text-muted-foreground">
                    {(edge.confidence * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {outgoing.length === 0 && incoming.length === 0 && (
        <p className="text-xs text-muted-foreground">No edges connected to this node.</p>
      )}
    </div>
  );
}
