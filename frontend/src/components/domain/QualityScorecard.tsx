import { cn } from '@/lib/utils';
import { pct } from '@/lib/formatters';

interface QualityMetric {
  metric_type: string;
  value: number;
  severity: string;
  message: string;
}

interface QualityScorecardProps {
  metrics: QualityMetric[];
  overallSeverity: string;
}

const SEVERITY_BANNER: Record<string, string> = {
  ok: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  warning: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  critical: 'bg-red-500/20 text-red-400 border-red-500/30',
};

const SEVERITY_BADGE: Record<string, string> = {
  ok: 'bg-emerald-500/20 text-emerald-400',
  warning: 'bg-amber-500/20 text-amber-400',
  critical: 'bg-red-500/20 text-red-400',
};

function humanize(snakeCase: string): string {
  return snakeCase
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export function QualityScorecard({ metrics, overallSeverity }: QualityScorecardProps) {
  const bannerColor =
    SEVERITY_BANNER[overallSeverity] ?? 'bg-slate-500/20 text-slate-400 border-slate-500/30';

  return (
    <div className="rounded-lg border border-border bg-card">
      {/* Overall severity banner */}
      <div
        className={cn(
          'rounded-t-lg border-b px-4 py-2 text-sm font-medium',
          bannerColor,
        )}
      >
        Overall: {overallSeverity.toUpperCase()}
      </div>

      {/* Metrics table */}
      <div className="divide-y divide-border">
        {metrics.map((m) => {
          const badgeColor = SEVERITY_BADGE[m.severity] ?? 'bg-slate-500/20 text-slate-400';

          return (
            <div
              key={m.metric_type}
              className="flex items-center gap-3 px-4 py-3 text-sm"
            >
              <span className="min-w-[120px] font-medium text-foreground">
                {humanize(m.metric_type)}
              </span>
              <span className="min-w-[56px] text-right font-mono text-foreground">
                {pct(m.value)}
              </span>
              <span
                className={cn(
                  'rounded-full px-2 py-0.5 text-xs',
                  badgeColor,
                )}
              >
                {m.severity}
              </span>
              <span className="flex-1 truncate text-xs text-muted-foreground">
                {m.message}
              </span>
            </div>
          );
        })}
        {metrics.length === 0 && (
          <div className="px-4 py-6 text-center text-sm text-muted-foreground">
            No quality metrics available
          </div>
        )}
      </div>
    </div>
  );
}

export function QualityScorecardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card">
      <div className="rounded-t-lg border-b border-border px-4 py-2">
        <div className="h-4 w-32 animate-pulse rounded bg-secondary" />
      </div>
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 border-b border-border px-4 py-3 last:border-b-0">
          <div className="h-4 w-28 animate-pulse rounded bg-secondary" />
          <div className="h-4 w-14 animate-pulse rounded bg-secondary" />
          <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
          <div className="h-3 w-40 animate-pulse rounded bg-secondary" />
        </div>
      ))}
    </div>
  );
}
