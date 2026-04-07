import { ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { pct } from '@/lib/formatters';

interface PathHop {
  from_concept: string;
  to_concept: string;
  predicate: string;
  sign: number;
  confidence: number;
  freshness: string;
}

interface Path {
  hops: PathHop[];
  total_score: number;
}

interface PathExplanationProps {
  paths: Path[];
}

const FRESHNESS_COLORS: Record<string, string> = {
  FRESH: 'bg-emerald-500/20 text-emerald-400',
  AGING: 'bg-amber-500/20 text-amber-400',
  STALE: 'bg-red-500/20 text-red-400',
};

export function PathExplanation({ paths }: PathExplanationProps) {
  if (paths.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-6 text-center text-sm text-muted-foreground">
        No propagation paths
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {paths.map((path, pathIdx) => (
        <div
          key={pathIdx}
          className="rounded-lg border border-border bg-card p-4"
        >
          {/* Path chain */}
          <div className="flex flex-wrap items-center gap-1">
            {path.hops.map((hop, hopIdx) => {
              const isPositive = hop.sign >= 0;
              const freshnessColor =
                FRESHNESS_COLORS[hop.freshness] ?? 'bg-slate-500/20 text-slate-400';

              return (
                <div key={hopIdx} className="flex items-center gap-1">
                  {/* Source node (only for first hop) */}
                  {hopIdx === 0 && (
                    <span className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs text-foreground">
                      {hop.from_concept}
                    </span>
                  )}

                  {/* Arrow + predicate */}
                  <div className="flex items-center gap-1 px-1">
                    <ArrowRight
                      className={cn(
                        'h-3 w-3',
                        isPositive ? 'text-emerald-400' : 'text-red-400',
                      )}
                    />
                    <span className="flex items-center gap-1">
                      <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-muted-foreground">
                        {hop.predicate}
                      </span>
                      <span
                        className={cn(
                          'font-mono text-[10px]',
                          isPositive ? 'text-emerald-400' : 'text-red-400',
                        )}
                      >
                        {isPositive ? '+' : '-'}
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {pct(hop.confidence, 0)}
                      </span>
                      <span className={cn('rounded-full px-1.5 py-0.5 text-[10px]', freshnessColor)}>
                        {hop.freshness}
                      </span>
                    </span>
                  </div>

                  {/* Target node */}
                  <span className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs text-foreground">
                    {hop.to_concept}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Total score */}
          <div className="mt-3 flex items-center justify-end gap-2 text-xs">
            <span className="text-muted-foreground">Total score:</span>
            <span
              className={cn(
                'font-mono font-medium',
                path.total_score >= 0 ? 'text-emerald-400' : 'text-red-400',
              )}
            >
              {path.total_score >= 0 ? '+' : ''}
              {path.total_score.toFixed(3)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function PathExplanationSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 2 }).map((_, i) => (
        <div key={i} className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <div className="h-5 w-16 animate-pulse rounded bg-secondary" />
            <div className="h-3 w-3 animate-pulse rounded bg-secondary" />
            <div className="h-5 w-20 animate-pulse rounded bg-secondary" />
            <div className="h-3 w-3 animate-pulse rounded bg-secondary" />
            <div className="h-5 w-16 animate-pulse rounded bg-secondary" />
            <div className="h-3 w-3 animate-pulse rounded bg-secondary" />
            <div className="h-5 w-16 animate-pulse rounded bg-secondary" />
          </div>
          <div className="mt-3 flex justify-end">
            <div className="h-4 w-24 animate-pulse rounded bg-secondary" />
          </div>
        </div>
      ))}
    </div>
  );
}
