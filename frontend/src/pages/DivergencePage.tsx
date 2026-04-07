import { useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, ChevronLeft, ChevronRight, Info, ShieldAlert, Split } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { cn } from '@/lib/utils';
import { timeAgo, latency } from '@/lib/formatters';
import { SEVERITY_COLORS, DIVERGENCE_REASON_COLORS } from '@/lib/constants';
import { useDivergences, type DivergenceFilters, type DivergenceItem } from '@/api/hooks/useDivergence';

const PAGE_SIZE = 20;

export default function DivergencePage() {
  const [severity, setSeverity] = useState('');
  const [reasonCode, setReasonCode] = useState('');
  const [issuer, setIssuer] = useState('');
  const [theme, setTheme] = useState('');
  const [offset, setOffset] = useState(0);

  const filters: DivergenceFilters = {
    severity: severity || undefined,
    reason_code: reasonCode || undefined,
    issuer: issuer || undefined,
    theme: theme || undefined,
    limit: PAGE_SIZE,
    offset,
  };

  const { data, isLoading, isError, error } = useDivergences(filters);

  function handleFilterChange() {
    setOffset(0);
  }

  // Pagination
  const total = data?.total ?? 0;
  const showing = data ? data.divergences.length : 0;
  const rangeStart = total > 0 ? offset + 1 : 0;
  const rangeEnd = offset + showing;
  const hasNext = rangeEnd < total;
  const hasPrev = offset > 0;

  // Severity counts from API
  const criticalCount = data?.severity_counts?.critical ?? 0;
  const warningCount = data?.severity_counts?.warning ?? 0;
  const infoCount = data?.severity_counts?.info ?? 0;

  const activeFilterCount = [severity, reasonCode, issuer, theme].filter(Boolean).length;

  return (
    <>
      <Header title="Divergence" />
      <div className="mx-auto max-w-5xl p-6">
        {/* Severity summary strip */}
        <div className="grid grid-cols-3 gap-3">
          {isLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            <>
              <MetricCard
                label="Critical"
                value={criticalCount}
                icon={ShieldAlert}
                trend={criticalCount > 0 ? 'up' : 'neutral'}
              />
              <MetricCard
                label="Warning"
                value={warningCount}
                icon={AlertTriangle}
                trend={warningCount > 0 ? 'up' : 'neutral'}
              />
              <MetricCard
                label="Info"
                value={infoCount}
                icon={Info}
                trend="neutral"
              />
            </>
          )}
        </div>

        {/* Filters */}
        <div className="mt-6 flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Severity</label>
            <select
              value={severity}
              onChange={(e) => {
                setSeverity(e.target.value);
                handleFilterChange();
              }}
              className="h-8 w-32 rounded border border-border bg-card px-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="">All</option>
              <option value="critical">Critical</option>
              <option value="warning">Warning</option>
              <option value="info">Info</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Reason Code</label>
            <select
              value={reasonCode}
              onChange={(e) => {
                setReasonCode(e.target.value);
                handleFilterChange();
              }}
              className="h-8 w-52 rounded border border-border bg-card px-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="">All</option>
              <option value="narrative_without_filing">Narrative without filing</option>
              <option value="filing_without_narrative">Filing without narrative</option>
              <option value="adverse_drift">Adverse drift</option>
              <option value="contradictory_drift">Contradictory drift</option>
              <option value="lagging_adoption">Lagging adoption</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Issuer</label>
            <input
              type="text"
              placeholder="e.g. TSMC"
              value={issuer}
              onChange={(e) => {
                setIssuer(e.target.value);
                handleFilterChange();
              }}
              className="h-8 w-36 rounded border border-border bg-card px-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Theme</label>
            <input
              type="text"
              placeholder="e.g. AI chips"
              value={theme}
              onChange={(e) => {
                setTheme(e.target.value);
                handleFilterChange();
              }}
              className="h-8 w-36 rounded border border-border bg-card px-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>

        {/* Results meta */}
        {data && (
          <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
            <span>
              {total} divergence{total !== 1 && 's'}
            </span>
            <span>{latency(data.latency_ms)}</span>
            {activeFilterCount > 0 && (
              <span className="rounded-full bg-primary/20 px-2 py-0.5 text-primary">
                {activeFilterCount} filter{activeFilterCount !== 1 && 's'} active
              </span>
            )}
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load divergences{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Divergence cards */}
        <div className="mt-4">
          {isLoading && (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="rounded-lg border border-border bg-card p-4">
                  <div className="flex items-center gap-3">
                    <div className="h-4 w-32 animate-pulse rounded bg-secondary" />
                    <div className="h-4 w-20 animate-pulse rounded bg-secondary" />
                  </div>
                  <div className="mt-3 h-3 w-3/4 animate-pulse rounded bg-secondary" />
                  <div className="mt-2 h-2 w-full animate-pulse rounded bg-secondary" />
                </div>
              ))}
            </div>
          )}

          {data && data.divergences.length === 0 && (
            <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-card py-20 text-muted-foreground">
              <Split className="h-12 w-12" />
              <p className="mt-3 text-sm">No divergences detected</p>
              <p className="mt-1 max-w-md text-center text-xs">
                Divergences appear when narrative momentum disagrees with SEC filing evidence for the
                same issuer and theme.
              </p>
            </div>
          )}

          {data && data.divergences.length > 0 && (
            <div className="space-y-3">
              {data.divergences.map((d) => (
                <DivergenceCard key={d.id} divergence={d} />
              ))}
            </div>
          )}
        </div>

        {/* Pagination */}
        {data && total > 0 && (
          <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground">
            <span>
              Showing {rangeStart}–{rangeEnd} of {total}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={!hasPrev}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
                Previous
              </button>
              <button
                type="button"
                disabled={!hasNext}
                onClick={() => setOffset(offset + PAGE_SIZE)}
                className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40"
              >
                Next
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

// ── Divergence Card ──

function DivergenceCard({ divergence }: { divergence: DivergenceItem }) {
  const d = divergence;
  const reasonClass = DIVERGENCE_REASON_COLORS[d.reason] ?? 'bg-slate-500/20 text-slate-400';
  const severityClass = SEVERITY_COLORS[d.severity] ?? 'bg-slate-500/20 text-slate-400';

  const narrativeScore = d.narrative_score;
  const filingScore = d.filing_adoption_score;
  const hasScores = narrativeScore != null && filingScore != null;

  // Format reason code for display
  const reasonLabel = d.reason.replace(/_/g, ' ');

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Header row: issuer + theme names */}
      <div className="flex flex-wrap items-center gap-2">
        <Link
          to={`/entities/ORG/${encodeURIComponent(d.issuer_name)}`}
          className="text-sm font-medium text-foreground hover:text-primary"
        >
          {d.issuer_name}
        </Link>
        <span className="text-muted-foreground">·</span>
        <Link
          to={`/themes/${d.theme_concept_id}`}
          className="text-sm text-muted-foreground hover:text-primary"
        >
          {d.theme_name}
        </Link>
        <span className="ml-auto text-xs text-muted-foreground">{timeAgo(d.created_at)}</span>
      </div>

      {/* Title */}
      <p className="mt-1.5 text-sm font-medium text-foreground">{d.title}</p>

      {/* Badges */}
      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <span className={cn('rounded px-1.5 py-0.5 text-xs font-medium', reasonClass)}>
          {reasonLabel}
        </span>
        <span className={cn('rounded px-1.5 py-0.5 text-xs font-medium', severityClass)}>
          {d.severity}
        </span>
      </div>

      {/* Score comparison bar */}
      {hasScores && (
        <div className="mt-3 space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="w-20 text-xs text-muted-foreground">Narrative</span>
            <div className="h-1.5 flex-1 rounded-full bg-secondary">
              <div
                className="h-1.5 rounded-full bg-sky-400 transition-all"
                style={{ width: `${Math.min(100, Math.round(narrativeScore * 100))}%` }}
              />
            </div>
            <span className="w-10 text-right text-xs font-medium text-foreground">
              {Math.round(narrativeScore * 100)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-xs text-muted-foreground">Filing</span>
            <div className="h-1.5 flex-1 rounded-full bg-secondary">
              <div
                className="h-1.5 rounded-full bg-amber-400 transition-all"
                style={{ width: `${Math.min(100, Math.round(filingScore * 100))}%` }}
              />
            </div>
            <span className="w-10 text-right text-xs font-medium text-foreground">
              {Math.round(filingScore * 100)}
            </span>
          </div>
        </div>
      )}

      {/* Summary */}
      {d.summary && (
        <p className="mt-3 text-xs leading-relaxed text-muted-foreground">{d.summary}</p>
      )}
    </div>
  );
}
