import { cn } from '@/lib/utils';

interface AdoptionBreakdownProps {
  sectionCoverage: number;
  sectionDepth: number;
  factAlignment: number;
  temporalConsistency: number;
}

interface BarProps {
  label: string;
  value: number;
}

function barColor(value: number): string {
  if (value > 0.6) return 'bg-emerald-500';
  if (value > 0.2) return 'bg-amber-500';
  return 'bg-red-500';
}

function Bar({ label, value }: BarProps) {
  const pctValue = Math.round(value * 100);

  return (
    <div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-foreground">{pctValue}%</span>
      </div>
      <div className="mt-1 h-2 w-full rounded-full bg-secondary">
        <div
          className={cn('h-2 rounded-full transition-all', barColor(value))}
          style={{ width: `${pctValue}%` }}
        />
      </div>
    </div>
  );
}

export function AdoptionBreakdown({
  sectionCoverage,
  sectionDepth,
  factAlignment,
  temporalConsistency,
}: AdoptionBreakdownProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="space-y-3">
        <Bar label="Section Coverage" value={sectionCoverage} />
        <Bar label="Section Depth" value={sectionDepth} />
        <Bar label="Fact Alignment" value={factAlignment} />
        <Bar label="Temporal Consistency" value={temporalConsistency} />
      </div>
    </div>
  );
}

export function AdoptionBreakdownSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="space-y-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i}>
            <div className="flex items-center justify-between">
              <div className="h-3 w-28 animate-pulse rounded bg-secondary" />
              <div className="h-3 w-8 animate-pulse rounded bg-secondary" />
            </div>
            <div className="mt-1 h-2 w-full animate-pulse rounded-full bg-secondary" />
          </div>
        ))}
      </div>
    </div>
  );
}
