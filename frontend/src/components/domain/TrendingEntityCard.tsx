import { TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';

const ENTITY_COLORS: Record<string, string> = {
  TICKER: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
  COMPANY: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  PRODUCT: 'bg-violet-500/20 text-violet-300 border-violet-500/30',
  TECHNOLOGY: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  METRIC: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
};

function entityColor(type: string): string {
  return ENTITY_COLORS[type] ?? 'bg-slate-500/20 text-slate-300 border-slate-500/30';
}

interface TrendingEntityCardProps {
  type: string;
  normalized: string;
  recent_count: number;
  baseline_count: number;
  spike_ratio: number;
  onClick?: () => void;
}

export function TrendingEntityCard({
  type,
  normalized,
  recent_count,
  baseline_count,
  spike_ratio,
  onClick,
}: TrendingEntityCardProps) {
  return (
    <div
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={onClick ? (e) => { if (e.key === 'Enter') onClick(); } : undefined}
      className={cn(
        'rounded-lg border border-border bg-card p-4 transition-colors',
        onClick && 'cursor-pointer hover:border-border/80',
      )}
    >
      {/* Top row: spike badge + type badge */}
      <div className="flex items-center gap-2 text-xs">
        <span className="flex items-center gap-1 rounded-full bg-rose-500/20 px-2 py-0.5 font-semibold text-rose-300">
          <TrendingUp className="h-3 w-3" />
          {spike_ratio.toFixed(1)}x
        </span>
        <span className={cn('rounded-full border px-2 py-0.5 font-medium', entityColor(type))}>
          {type}
        </span>
      </div>

      {/* Entity name */}
      <div className="mt-2 font-medium text-foreground">{normalized}</div>

      {/* Recent vs baseline counts */}
      <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground">
        <span>
          Recent: <span className="font-medium text-foreground">{recent_count}</span>
        </span>
        <span>
          Baseline: <span className="font-medium text-foreground">{baseline_count}</span>
        </span>
      </div>
    </div>
  );
}

export function TrendingEntityCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="h-5 w-14 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-20 animate-pulse rounded-full bg-secondary" />
      </div>
      <div className="mt-3 h-4 w-2/3 animate-pulse rounded bg-secondary" />
      <div className="mt-3 flex gap-3">
        <div className="h-3 w-20 animate-pulse rounded bg-secondary" />
        <div className="h-3 w-24 animate-pulse rounded bg-secondary" />
      </div>
    </div>
  );
}
