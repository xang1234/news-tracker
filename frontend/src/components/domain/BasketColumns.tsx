import { AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BasketMember {
  concept_id: string;
  concept_name: string;
  role: string;
  best_score: number;
  best_sign: number;
  min_hops: number;
  path_count: number;
  has_mixed_signals: boolean;
}

interface BasketColumnsProps {
  members: BasketMember[];
  onMemberClick?: (conceptId: string) => void;
}

function MemberCard({
  member,
  onClick,
}: {
  member: BasketMember;
  onClick?: (conceptId: string) => void;
}) {
  const scorePct = Math.min(100, Math.round(member.best_score * 100));
  const isBeneficiary = member.role === 'beneficiary';

  return (
    <div
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick ? () => onClick(member.concept_id) : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter') onClick(member.concept_id);
            }
          : undefined
      }
      className={cn(
        'rounded-lg border border-border bg-card p-3 transition-colors',
        onClick && 'cursor-pointer hover:border-border/80',
      )}
    >
      {/* Name + mixed signals */}
      <div className="flex items-center gap-2">
        <span className="font-medium text-foreground">{member.concept_name}</span>
        {member.has_mixed_signals && (
          <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
        )}
      </div>

      {/* Score bar */}
      <div className="mt-2">
        <div className="h-1.5 w-full rounded-full bg-secondary">
          <div
            className={cn(
              'h-1.5 rounded-full transition-all',
              isBeneficiary ? 'bg-emerald-500' : 'bg-red-500',
            )}
            style={{ width: `${scorePct}%` }}
          />
        </div>
      </div>

      {/* Badges row */}
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
          {member.min_hops} hop{member.min_hops !== 1 && 's'}
        </span>
        <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
          {member.path_count} path{member.path_count !== 1 && 's'}
        </span>
        <span className="ml-auto font-mono text-muted-foreground">
          {member.best_sign >= 0 ? '+' : ''}{member.best_score.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export function BasketColumns({ members, onMemberClick }: BasketColumnsProps) {
  const beneficiaries = members.filter((m) => m.role === 'beneficiary');
  const atRisk = members.filter((m) => m.role === 'at_risk');

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {/* Beneficiaries column */}
      <div>
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-emerald-400">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" />
          Beneficiaries
        </div>
        <div className="space-y-2">
          {beneficiaries.length > 0 ? (
            beneficiaries.map((m) => (
              <MemberCard key={m.concept_id} member={m} onClick={onMemberClick} />
            ))
          ) : (
            <div className="rounded-lg border border-border bg-card p-4 text-center text-sm text-muted-foreground">
              No beneficiaries
            </div>
          )}
        </div>
      </div>

      {/* At Risk column */}
      <div>
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-red-400">
          <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
          At Risk
        </div>
        <div className="space-y-2">
          {atRisk.length > 0 ? (
            atRisk.map((m) => (
              <MemberCard key={m.concept_id} member={m} onClick={onMemberClick} />
            ))
          ) : (
            <div className="rounded-lg border border-border bg-card p-4 text-center text-sm text-muted-foreground">
              No at-risk companies
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function BasketColumnsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {[0, 1].map((col) => (
        <div key={col}>
          <div className="mb-3 h-4 w-24 animate-pulse rounded bg-secondary" />
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="rounded-lg border border-border bg-card p-3">
                <div className="h-4 w-28 animate-pulse rounded bg-secondary" />
                <div className="mt-2 h-1.5 w-full animate-pulse rounded-full bg-secondary" />
                <div className="mt-2 flex gap-2">
                  <div className="h-5 w-14 animate-pulse rounded-full bg-secondary" />
                  <div className="h-5 w-14 animate-pulse rounded-full bg-secondary" />
                  <div className="ml-auto h-4 w-10 animate-pulse rounded bg-secondary" />
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
