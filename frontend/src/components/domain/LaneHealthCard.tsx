import { cn } from '@/lib/utils';
import { timeAgo } from '@/lib/formatters';

interface LaneHealthCardProps {
  lane: string;
  freshness: string;
  quality: string;
  quarantine: string;
  readiness: string;
  lastCompletedAt: string | null;
}

const FRESHNESS_COLORS: Record<string, string> = {
  FRESH: 'bg-emerald-500/20 text-emerald-400',
  AGING: 'bg-amber-500/20 text-amber-400',
  STALE: 'bg-red-500/20 text-red-400',
  UNKNOWN: 'bg-slate-500/20 text-slate-400',
};

const QUALITY_COLORS: Record<string, string> = {
  HEALTHY: 'bg-emerald-500/20 text-emerald-400',
  DEGRADED: 'bg-amber-500/20 text-amber-400',
  CRITICAL: 'bg-red-500/20 text-red-400',
};

const QUARANTINE_COLORS: Record<string, string> = {
  CLEAR: 'bg-emerald-500/20 text-emerald-400',
  WATCH: 'bg-amber-500/20 text-amber-400',
  QUARANTINED: 'bg-red-500/20 text-red-400',
};

const READINESS_DOT: Record<string, string> = {
  READY: 'bg-emerald-400',
  WARN: 'bg-amber-400',
  BLOCKED: 'bg-red-400',
};

function titleize(value: string): string {
  return value
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export function LaneHealthCard({
  lane,
  freshness,
  quality,
  quarantine,
  readiness,
  lastCompletedAt,
}: LaneHealthCardProps) {
  const dotColor = READINESS_DOT[readiness] ?? 'bg-slate-400';

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Title + readiness */}
      <div className="flex items-center justify-between">
        <span className="font-medium text-foreground">{titleize(lane)}</span>
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className={cn('inline-block h-2 w-2 rounded-full', dotColor)} />
          {readiness}
        </span>
      </div>

      {/* Badges */}
      <div className="mt-3 flex flex-wrap gap-2 text-xs">
        <span
          className={cn(
            'rounded-full px-2 py-0.5',
            FRESHNESS_COLORS[freshness] ?? 'bg-slate-500/20 text-slate-400',
          )}
        >
          {freshness}
        </span>
        <span
          className={cn(
            'rounded-full px-2 py-0.5',
            QUALITY_COLORS[quality] ?? 'bg-slate-500/20 text-slate-400',
          )}
        >
          {quality}
        </span>
        <span
          className={cn(
            'rounded-full px-2 py-0.5',
            QUARANTINE_COLORS[quarantine] ?? 'bg-slate-500/20 text-slate-400',
          )}
        >
          {quarantine}
        </span>
      </div>

      {/* Last completed */}
      <div className="mt-3 text-xs text-muted-foreground">
        {lastCompletedAt ? `Completed ${timeAgo(lastCompletedAt)}` : 'Never completed'}
      </div>
    </div>
  );
}

export function LaneHealthCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <div className="h-4 w-24 animate-pulse rounded bg-secondary" />
        <div className="h-4 w-16 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3 flex gap-2">
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
      </div>
      <div className="mt-3 h-3 w-28 animate-pulse rounded bg-secondary" />
    </div>
  );
}
