import { useState } from 'react';
import { Zap } from 'lucide-react';
import { cn } from '@/lib/utils';
import { RELATION_COLORS } from '@/lib/constants';
import { useGraphPropagate, type PropagationImpactItem } from '@/api/hooks/useGraph';
import { latency } from '@/lib/formatters';

interface PropagationPanelProps {
  sourceNodeId?: string;
  sourceNodeName?: string;
}

export function PropagationPanel({ sourceNodeId, sourceNodeName }: PropagationPanelProps) {
  const [delta, setDelta] = useState(0.5);
  const { mutate, data, isPending } = useGraphPropagate();

  function handlePropagate() {
    if (!sourceNodeId) return;
    mutate({ source_node: sourceNodeId, sentiment_delta: delta });
  }

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="text-xs font-medium text-muted-foreground">Sentiment Propagation</div>
        <div className="mt-2">
          <div className="text-xs text-muted-foreground">Source Node</div>
          <div className="mt-1 text-sm font-medium text-foreground">
            {sourceNodeName ?? sourceNodeId ?? 'Select a node'}
          </div>
        </div>
        <div className="mt-3">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Sentiment Delta</span>
            <span className={cn(
              'font-medium',
              delta > 0 ? 'text-emerald-400' : delta < 0 ? 'text-red-400' : 'text-muted-foreground',
            )}>
              {delta > 0 ? '+' : ''}{delta.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={-1}
            max={1}
            step={0.05}
            value={delta}
            onChange={(e) => setDelta(parseFloat(e.target.value))}
            className="mt-1 w-full accent-primary"
          />
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>-1.0</span>
            <span>0</span>
            <span>+1.0</span>
          </div>
        </div>
        <button
          type="button"
          disabled={!sourceNodeId || isPending}
          onClick={handlePropagate}
          className="mt-3 flex w-full items-center justify-center gap-2 rounded border border-border bg-primary/10 px-3 py-2 text-sm text-primary hover:bg-primary/20 disabled:opacity-40"
        >
          <Zap className="h-4 w-4" />
          {isPending ? 'Propagating...' : 'Propagate'}
        </button>
      </div>

      {/* Results */}
      {data && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{data.total_affected} node{data.total_affected !== 1 && 's'} affected</span>
            <span>{latency(data.latency_ms)}</span>
          </div>
          <div className="mt-3 space-y-1.5">
            {data.impacts.map((impact: PropagationImpactItem) => (
              <div
                key={impact.node_id}
                className="flex items-center gap-2 rounded border border-border bg-background px-3 py-2 text-xs"
              >
                <span className="font-mono text-foreground">{impact.node_id}</span>
                <span
                  className="rounded px-1.5 py-0.5 text-[10px]"
                  style={{
                    backgroundColor: `${RELATION_COLORS[impact.relation] ?? '#6b7280'}20`,
                    color: RELATION_COLORS[impact.relation] ?? '#6b7280',
                  }}
                >
                  {impact.relation}
                </span>
                <span className="rounded-full bg-secondary px-1.5 py-0.5 text-[10px] text-muted-foreground">
                  depth {impact.depth}
                </span>
                <span className={cn(
                  'ml-auto font-medium',
                  impact.impact > 0 ? 'text-emerald-400' : 'text-red-400',
                )}>
                  {impact.impact > 0 ? '+' : ''}{impact.impact.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
