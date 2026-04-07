import { cn } from '@/lib/utils';
import { humanize } from '@/lib/formatters';

interface DriftDimension {
  dimension: string;
  z_score: number;
  magnitude: number;
  is_unusual: boolean;
}

interface DriftRadarProps {
  dimensions: DriftDimension[];
}

export function DriftRadar({ dimensions }: DriftRadarProps) {
  if (dimensions.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-6 text-center text-sm text-muted-foreground">
        No drift dimensions available
      </div>
    );
  }

  const maxMagnitude = Math.max(...dimensions.map((d) => d.magnitude), 1);

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="space-y-3">
        {dimensions.map((dim) => {
          const isUnusual = dim.is_unusual;
          const magnitudePct = Math.min(100, Math.round((dim.magnitude / maxMagnitude) * 100));

          return (
            <div key={dim.dimension} className="flex items-center gap-3">
              {/* Unusual flag */}
              <span
                className={cn(
                  'inline-block h-2 w-2 flex-shrink-0 rounded-full',
                  isUnusual ? 'bg-red-500' : 'bg-emerald-500',
                )}
              />

              {/* Dimension name */}
              <span className="min-w-[120px] text-sm text-foreground">
                {humanize(dim.dimension)}
              </span>

              {/* Z-score */}
              <span
                className={cn(
                  'min-w-[56px] text-right font-mono text-xs',
                  isUnusual ? 'text-red-400' : 'text-emerald-400',
                )}
              >
                {dim.z_score.toFixed(2)}
              </span>

              {/* Magnitude bar */}
              <div className="flex flex-1 items-center gap-2">
                <div className="h-1.5 flex-1 rounded-full bg-secondary">
                  <div
                    className={cn(
                      'h-1.5 rounded-full transition-all',
                      isUnusual ? 'bg-red-500' : 'bg-emerald-500',
                    )}
                    style={{ width: `${magnitudePct}%` }}
                  />
                </div>
                <span className="min-w-[40px] text-right font-mono text-xs text-muted-foreground">
                  {dim.magnitude.toFixed(2)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function DriftRadarSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="h-2 w-2 animate-pulse rounded-full bg-secondary" />
            <div className="h-4 w-28 animate-pulse rounded bg-secondary" />
            <div className="h-4 w-12 animate-pulse rounded bg-secondary" />
            <div className="h-1.5 flex-1 animate-pulse rounded-full bg-secondary" />
            <div className="h-4 w-10 animate-pulse rounded bg-secondary" />
          </div>
        ))}
      </div>
    </div>
  );
}
