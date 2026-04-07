import { cn } from '@/lib/utils';
import { timeAgo } from '@/lib/formatters';

interface Claim {
  claim_id: string;
  lane: string;
  source_id: string;
  source_type: string;
  subject_text: string;
  predicate: string;
  object_text: string;
  confidence: number;
  extraction_method: string;
  created_at: string;
}

interface ClaimLink {
  claim_id: string;
  link_type: string;
  contribution_weight: number;
  claim: Claim;
}

interface ClaimTableProps {
  links: ClaimLink[];
}

const LANE_COLORS: Record<string, string> = {
  narrative: 'bg-sky-500/20 text-sky-400',
  filing: 'bg-violet-500/20 text-violet-400',
  structural: 'bg-emerald-500/20 text-emerald-400',
  backtest: 'bg-amber-500/20 text-amber-400',
};

const LINK_TYPE_COLORS: Record<string, string> = {
  support: 'text-emerald-400',
  contradiction: 'text-red-400',
};

export function ClaimTable({ links }: ClaimTableProps) {
  if (links.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-6 text-center text-sm text-muted-foreground">
        No linked claims
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-card">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs text-muted-foreground">
            <th className="px-4 py-3 font-medium">Lane</th>
            <th className="px-4 py-3 font-medium">Source</th>
            <th className="px-4 py-3 font-medium">Method</th>
            <th className="px-4 py-3 font-medium">Confidence</th>
            <th className="px-4 py-3 font-medium">Weight</th>
            <th className="px-4 py-3 font-medium">Type</th>
            <th className="px-4 py-3 font-medium">Created</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {links.map((link) => {
            const laneColor = LANE_COLORS[link.claim.lane] ?? 'bg-slate-500/20 text-slate-400';
            const typeColor = LINK_TYPE_COLORS[link.link_type] ?? 'text-muted-foreground';
            const confidencePct = Math.round(link.claim.confidence * 100);
            const weightPct = Math.round(link.contribution_weight * 100);

            return (
              <tr key={link.claim_id} className="text-foreground">
                <td className="px-4 py-3">
                  <span className={cn('rounded-full px-2 py-0.5 text-xs', laneColor)}>
                    {link.claim.lane}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <span className="text-xs text-muted-foreground">
                    {link.claim.source_type}
                  </span>
                  <div className="font-mono text-xs">{link.claim.source_id}</div>
                </td>
                <td className="px-4 py-3">
                  <span className="rounded bg-secondary px-1.5 py-0.5 text-xs text-muted-foreground">
                    {link.claim.extraction_method}
                  </span>
                </td>
                <td className="px-4 py-3 font-mono text-xs">{confidencePct}%</td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 rounded-full bg-secondary">
                      <div
                        className="h-1.5 rounded-full bg-primary"
                        style={{ width: `${weightPct}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs text-muted-foreground">{weightPct}%</span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span className={cn('text-xs font-medium', typeColor)}>
                    {link.link_type}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-muted-foreground">
                  {timeAgo(link.claim.created_at)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ClaimTableSkeleton() {
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-card">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs text-muted-foreground">
            <th className="px-4 py-3 font-medium">Lane</th>
            <th className="px-4 py-3 font-medium">Source</th>
            <th className="px-4 py-3 font-medium">Method</th>
            <th className="px-4 py-3 font-medium">Confidence</th>
            <th className="px-4 py-3 font-medium">Weight</th>
            <th className="px-4 py-3 font-medium">Type</th>
            <th className="px-4 py-3 font-medium">Created</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {Array.from({ length: 3 }).map((_, i) => (
            <tr key={i}>
              <td className="px-4 py-3"><div className="h-5 w-16 animate-pulse rounded-full bg-secondary" /></td>
              <td className="px-4 py-3"><div className="h-4 w-20 animate-pulse rounded bg-secondary" /></td>
              <td className="px-4 py-3"><div className="h-5 w-14 animate-pulse rounded bg-secondary" /></td>
              <td className="px-4 py-3"><div className="h-4 w-10 animate-pulse rounded bg-secondary" /></td>
              <td className="px-4 py-3"><div className="h-1.5 w-16 animate-pulse rounded-full bg-secondary" /></td>
              <td className="px-4 py-3"><div className="h-4 w-16 animate-pulse rounded bg-secondary" /></td>
              <td className="px-4 py-3"><div className="h-4 w-20 animate-pulse rounded bg-secondary" /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
