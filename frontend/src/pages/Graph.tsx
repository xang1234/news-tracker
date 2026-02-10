import { useState, useMemo } from 'react';
import { Network } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { GraphCanvas } from '@/components/domain/GraphCanvas';
import { GraphNodePanel } from '@/components/domain/GraphNodePanel';
import { PropagationPanel } from '@/components/domain/PropagationPanel';
import { useGraphNodes, useGraphSubgraph } from '@/api/hooks/useGraph';
import { latency } from '@/lib/formatters';
import { cn } from '@/lib/utils';

type RightTab = 'details' | 'propagation';

export default function Graph() {
  const [nodeTypeFilter, setNodeTypeFilter] = useState('');
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>();
  const [rightTab, setRightTab] = useState<RightTab>('details');

  const allNodes = useGraphNodes({
    node_type: nodeTypeFilter || undefined,
    limit: 500,
  });

  const subgraph = useGraphSubgraph(selectedNodeId, 2);

  // Merge: use subgraph data when a node is selected, otherwise full node list
  const displayNodes = subgraph.data?.nodes ?? allNodes.data?.nodes ?? [];
  const displayEdges = subgraph.data?.edges ?? [];

  const selectedNode = useMemo(
    () => displayNodes.find((n) => n.node_id === selectedNodeId),
    [displayNodes, selectedNodeId],
  );

  function handleNodeClick(nodeId: string) {
    setSelectedNodeId(nodeId);
    setRightTab('details');
  }

  const isLoading = allNodes.isLoading;
  const isError = allNodes.isError;

  const NODE_TYPES = ['', 'ticker', 'theme', 'technology'];

  return (
    <>
      <Header title="Causal Graph" />
      <div className="mx-auto max-w-7xl p-6">
        {/* Top controls */}
        <div className="mb-4 flex items-center gap-3">
          <select
            value={nodeTypeFilter}
            onChange={(e) => { setNodeTypeFilter(e.target.value); setSelectedNodeId(undefined); }}
            aria-label="Filter by node type"
            className="rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          >
            <option value="">All Types</option>
            {NODE_TYPES.filter(Boolean).map((t) => (
              <option key={t} value={t}>
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </option>
            ))}
          </select>
          {allNodes.data && (
            <span className="text-xs text-muted-foreground">
              {allNodes.data.total} node{allNodes.data.total !== 1 && 's'} &middot; {latency(allNodes.data.latency_ms)}
            </span>
          )}
          {selectedNodeId && (
            <button
              type="button"
              onClick={() => setSelectedNodeId(undefined)}
              className="ml-auto text-xs text-muted-foreground hover:text-foreground"
            >
              Clear selection
            </button>
          )}
        </div>

        {/* Error */}
        {isError && (
          <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load graph nodes
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="flex items-center justify-center py-32 text-muted-foreground">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" role="status" aria-label="Loading" />
          </div>
        )}

        {/* Empty */}
        {allNodes.data && allNodes.data.nodes.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
            <Network className="h-12 w-12" />
            <p className="mt-3 text-sm">No graph nodes found</p>
            <p className="mt-1 text-xs">
              Seed the graph with `uv run news-tracker graph seed`
            </p>
          </div>
        )}

        {/* Main layout */}
        {allNodes.data && allNodes.data.nodes.length > 0 && (
          <div className="flex flex-col gap-4 lg:flex-row">
            {/* Left: Graph canvas */}
            <div className={cn('flex-1', selectedNodeId ? 'w-full lg:w-2/3' : 'w-full')}>
              <GraphCanvas
                nodes={displayNodes}
                edges={displayEdges}
                selectedNodeId={selectedNodeId}
                onNodeClick={handleNodeClick}
                height={600}
              />
            </div>

            {/* Right: Detail/Propagation panel */}
            {selectedNodeId && (
              <div className="w-full shrink-0 lg:w-80">
                {/* Tab toggle */}
                <div className="mb-3 flex gap-0 border-b border-border">
                  <button
                    type="button"
                    onClick={() => setRightTab('details')}
                    className={cn(
                      'border-b-2 px-3 py-2 text-xs font-medium',
                      rightTab === 'details'
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground',
                    )}
                  >
                    Node Details
                  </button>
                  <button
                    type="button"
                    onClick={() => setRightTab('propagation')}
                    className={cn(
                      'border-b-2 px-3 py-2 text-xs font-medium',
                      rightTab === 'propagation'
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground',
                    )}
                  >
                    Propagation
                  </button>
                </div>

                {rightTab === 'details' && selectedNode && (
                  <GraphNodePanel
                    node={selectedNode}
                    edges={displayEdges}
                    allNodes={displayNodes}
                    onNavigateNode={handleNodeClick}
                  />
                )}

                {rightTab === 'propagation' && (
                  <PropagationPanel
                    sourceNodeId={selectedNodeId}
                    sourceNodeName={selectedNode?.name}
                  />
                )}

                {subgraph.isLoading && (
                  <div className="mt-4 flex items-center justify-center py-8">
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  </div>
                )}
                {subgraph.isError && (
                  <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Failed to load subgraph
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}
