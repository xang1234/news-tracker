import { cn } from '@/lib/utils';
import { pct, timeAgo } from '@/lib/formatters';

interface Assertion {
  assertion_id: string;
  subject_concept_id: string;
  predicate: string;
  object_concept_id: string;
  confidence: number;
  status: string;
  support_count: number;
  contradiction_count: number;
  source_diversity: number;
  first_seen_at: string;
  last_evidence_at: string;
}

interface AssertionCardProps {
  assertion: Assertion;
  isSelected?: boolean;
  onClick?: () => void;
}

const STATUS_COLORS: Record<string, string> = {
  active: 'bg-emerald-500/20 text-emerald-400',
  disputed: 'bg-amber-500/20 text-amber-400',
  retracted: 'bg-red-500/20 text-red-400',
  superseded: 'bg-slate-500/20 text-slate-400',
};

const PREDICATE_COLORS: Record<string, string> = {
  supplies_to: 'bg-sky-500/20 text-sky-400',
  depends_on: 'bg-violet-500/20 text-violet-400',
  competes_with: 'bg-red-500/20 text-red-400',
  enables: 'bg-emerald-500/20 text-emerald-400',
  disrupts: 'bg-amber-500/20 text-amber-400',
  acquires: 'bg-cyan-500/20 text-cyan-400',
  partners_with: 'bg-teal-500/20 text-teal-400',
  invests_in: 'bg-indigo-500/20 text-indigo-400',
};

export function AssertionCard({ assertion, isSelected, onClick }: AssertionCardProps) {
  const statusColor = STATUS_COLORS[assertion.status] ?? 'bg-slate-500/20 text-slate-400';
  const predicateColor = PREDICATE_COLORS[assertion.predicate] ?? 'bg-secondary text-muted-foreground';
  const confidencePct = Math.round(assertion.confidence * 100);

  return (
    <div
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={onClick ? (e) => { if (e.key === 'Enter') onClick(); } : undefined}
      className={cn(
        'rounded-lg border bg-card p-4 transition-colors',
        isSelected ? 'border-primary bg-primary/5' : 'border-border',
        onClick && 'cursor-pointer hover:border-border/80',
      )}
    >
      {/* Triple: [subject] --predicate--> [object] */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs text-foreground">
          {assertion.subject_concept_id}
        </span>
        <span className={cn('rounded-full px-2 py-0.5 text-xs', predicateColor)}>
          {assertion.predicate}
        </span>
        <span className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs text-foreground">
          {assertion.object_concept_id}
        </span>
      </div>

      {/* Confidence bar */}
      <div className="mt-3">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Confidence</span>
          <span>{pct(assertion.confidence, 0)}</span>
        </div>
        <div className="mt-1 h-1.5 w-full rounded-full bg-secondary">
          <div
            className="h-1.5 rounded-full bg-gradient-to-r from-emerald-600 to-emerald-400"
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {/* Evidence badges */}
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-emerald-400">
          {assertion.support_count} support
        </span>
        <span className="rounded-full bg-red-500/20 px-2 py-0.5 text-red-400">
          {assertion.contradiction_count} contradict
        </span>
        <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
          diversity: {assertion.source_diversity}
        </span>
      </div>

      {/* Status + time */}
      <div className="mt-3 flex items-center justify-between text-xs">
        <span className={cn('rounded-full px-2 py-0.5', statusColor)}>
          {assertion.status}
        </span>
        <span className="text-muted-foreground">
          first seen {timeAgo(assertion.first_seen_at)}
        </span>
      </div>
    </div>
  );
}

export function AssertionCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="h-5 w-20 animate-pulse rounded bg-secondary" />
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-20 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3">
        <div className="flex justify-between">
          <div className="h-3 w-16 animate-pulse rounded bg-secondary" />
          <div className="h-3 w-8 animate-pulse rounded bg-secondary" />
        </div>
        <div className="mt-1 h-1.5 w-full animate-pulse rounded-full bg-secondary" />
      </div>
      <div className="mt-3 flex gap-2">
        <div className="h-5 w-20 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-24 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-20 animate-pulse rounded-full bg-secondary" />
      </div>
      <div className="mt-3 flex items-center justify-between">
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="h-3 w-24 animate-pulse rounded bg-secondary" />
      </div>
    </div>
  );
}
