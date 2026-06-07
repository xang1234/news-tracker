import { ArrowRight } from 'lucide-react';
import { RELATION_COLORS } from '@/lib/constants';
import { pct } from '@/lib/formatters';
import type { PropagationHopItem } from '@/api/hooks/useGraph';

interface EdgePathProps {
  hops: PropagationHopItem[];
}

/**
 * Renders a propagation causal chain (o59.3) as `src →relation(conf)→ tgt …`.
 *
 * A focused renderer for the propagation hop shape, distinct from
 * PathExplanation (which carries freshness/sign/total_score for structural
 * scored paths that propagation hops don't have).
 */
export function EdgePath({ hops }: EdgePathProps) {
  if (hops.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-1">
      {hops.map((hop, i) => {
        const color = RELATION_COLORS[hop.relation] ?? '#6b7280';
        return (
          <div key={i} className="flex items-center gap-1">
            {i === 0 && (
              <span className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-foreground">
                {hop.from_node}
              </span>
            )}
            <div className="flex items-center gap-1 px-0.5">
              <ArrowRight className="h-3 w-3 text-muted-foreground" />
              <span
                className="rounded px-1.5 py-0.5 text-[10px]"
                style={{ backgroundColor: `${color}20`, color }}
              >
                {hop.relation}
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                {pct(hop.edge_confidence, 0)}
              </span>
            </div>
            <span className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-foreground">
              {hop.to_node}
            </span>
          </div>
        );
      })}
    </div>
  );
}
