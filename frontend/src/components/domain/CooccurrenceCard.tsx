import { Hash } from 'lucide-react';
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

interface CooccurrenceCardProps {
  type: string;
  normalized: string;
  cooccurrence_count: number;
  jaccard: number;
  onClick?: () => void;
}

export function CooccurrenceCard({
  type,
  normalized,
  cooccurrence_count,
  jaccard,
  onClick,
}: CooccurrenceCardProps) {
  const jaccardPct = Math.round(jaccard * 100);

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
      {/* Top row: type badge + count */}
      <div className="flex items-center gap-2 text-xs">
        <span className={cn('rounded-full border px-2 py-0.5 font-medium', entityColor(type))}>
          {type}
        </span>
        <span className="ml-auto flex items-center gap-1 text-muted-foreground">
          <Hash className="h-3 w-3" />
          {cooccurrence_count}
        </span>
      </div>

      {/* Entity name */}
      <div className="mt-2 font-medium text-foreground">{normalized}</div>

      {/* Jaccard similarity bar */}
      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
          <span>Jaccard similarity</span>
          <span className="font-medium text-foreground">{jaccardPct}%</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{ width: `${jaccardPct}%` }}
          />
        </div>
      </div>
    </div>
  );
}

export function CooccurrenceCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="h-5 w-20 animate-pulse rounded-full bg-secondary" />
        <div className="ml-auto h-4 w-10 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3 h-4 w-2/3 animate-pulse rounded bg-secondary" />
      <div className="mt-3 space-y-1">
        <div className="flex justify-between">
          <div className="h-3 w-24 animate-pulse rounded bg-secondary" />
          <div className="h-3 w-8 animate-pulse rounded bg-secondary" />
        </div>
        <div className="h-1.5 w-full animate-pulse rounded-full bg-secondary" />
      </div>
    </div>
  );
}
