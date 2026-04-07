import { cn } from '@/lib/utils';
import { timeAgo, truncate } from '@/lib/formatters';

interface Divergence {
  id: string;
  issuer_concept_id: string;
  issuer_name: string;
  theme_concept_id: string;
  theme_name: string;
  reason: string;
  severity: string;
  title: string;
  summary: string;
  narrative_score: number | null;
  filing_adoption_score: number | null;
  created_at: string;
}

interface DivergenceCardProps {
  divergence: Divergence;
  onClick?: () => void;
}

const SEVERITY_BORDER: Record<string, string> = {
  critical: 'border-l-red-500',
  warning: 'border-l-amber-500',
  info: 'border-l-sky-500',
};

const SEVERITY_BADGE: Record<string, string> = {
  critical: 'bg-red-500/20 text-red-400',
  warning: 'bg-amber-500/20 text-amber-400',
  info: 'bg-sky-500/20 text-sky-400',
};

const REASON_COLORS: Record<string, string> = {
  narrative_without_filing: 'bg-amber-500/20 text-amber-400',
  filing_without_narrative: 'bg-sky-500/20 text-sky-400',
  adverse_drift: 'bg-red-500/20 text-red-400',
  contradictory_drift: 'bg-rose-500/20 text-rose-400',
  lagging_adoption: 'bg-yellow-500/20 text-yellow-400',
};

function humanizeReason(reason: string): string {
  return reason
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function scoreBar(label: string, value: number | null) {
  const pctValue = value != null ? Math.round(value * 100) : null;

  return (
    <div className="flex-1">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      {pctValue != null ? (
        <div className="mt-1 flex items-center gap-2">
          <div className="h-1.5 flex-1 rounded-full bg-secondary">
            <div
              className={cn(
                'h-1.5 rounded-full',
                pctValue >= 60
                  ? 'bg-emerald-500'
                  : pctValue >= 30
                    ? 'bg-amber-500'
                    : 'bg-red-500',
              )}
              style={{ width: `${pctValue}%` }}
            />
          </div>
          <span className="font-mono text-xs text-foreground">{pctValue}%</span>
        </div>
      ) : (
        <div className="mt-1 text-xs text-muted-foreground">--</div>
      )}
    </div>
  );
}

export function DivergenceCard({ divergence, onClick }: DivergenceCardProps) {
  const borderColor = SEVERITY_BORDER[divergence.severity] ?? 'border-l-slate-500';
  const severityBadge = SEVERITY_BADGE[divergence.severity] ?? 'bg-slate-500/20 text-slate-400';
  const reasonColor = REASON_COLORS[divergence.reason] ?? 'bg-secondary text-muted-foreground';

  return (
    <div
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={onClick ? (e) => { if (e.key === 'Enter') onClick(); } : undefined}
      className={cn(
        'rounded-lg border border-border border-l-4 bg-card p-4 transition-colors',
        borderColor,
        onClick && 'cursor-pointer hover:border-border/80',
      )}
    >
      {/* Top row: issuer + theme + severity */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="font-medium text-foreground">{divergence.issuer_name}</span>
        <span className="text-muted-foreground">{divergence.theme_name}</span>
        <span className={cn('ml-auto rounded-full px-2 py-0.5', severityBadge)}>
          {divergence.severity}
        </span>
      </div>

      {/* Title */}
      <div className="mt-2 text-sm font-medium text-foreground">{divergence.title}</div>

      {/* Reason badge */}
      <div className="mt-2">
        <span className={cn('rounded-full px-2 py-0.5 text-xs', reasonColor)}>
          {humanizeReason(divergence.reason)}
        </span>
      </div>

      {/* Score comparison */}
      <div className="mt-3 flex gap-4">
        {scoreBar('Narrative', divergence.narrative_score)}
        {scoreBar('Filing', divergence.filing_adoption_score)}
      </div>

      {/* Summary */}
      {divergence.summary && (
        <p className="mt-3 text-xs text-muted-foreground">
          {truncate(divergence.summary, 180)}
        </p>
      )}

      {/* Time */}
      <div className="mt-3 text-xs text-muted-foreground">
        {timeAgo(divergence.created_at)}
      </div>
    </div>
  );
}

export function DivergenceCardSkeleton() {
  return (
    <div className="rounded-lg border border-border border-l-4 border-l-secondary bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="h-4 w-24 animate-pulse rounded bg-secondary" />
        <div className="h-4 w-20 animate-pulse rounded bg-secondary" />
        <div className="ml-auto h-5 w-16 animate-pulse rounded-full bg-secondary" />
      </div>
      <div className="mt-3 h-4 w-3/4 animate-pulse rounded bg-secondary" />
      <div className="mt-2 h-5 w-32 animate-pulse rounded-full bg-secondary" />
      <div className="mt-3 flex gap-4">
        <div className="h-6 flex-1 animate-pulse rounded bg-secondary" />
        <div className="h-6 flex-1 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3 h-3 w-full animate-pulse rounded bg-secondary" />
      <div className="mt-3 h-3 w-20 animate-pulse rounded bg-secondary" />
    </div>
  );
}
