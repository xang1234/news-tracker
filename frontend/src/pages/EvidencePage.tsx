import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronLeft, ChevronRight, FileSearch, Scale } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { cn } from '@/lib/utils';
import { timeAgo, pct, latency, formatDate } from '@/lib/formatters';
import { ASSERTION_STATUS_COLORS } from '@/lib/constants';
import {
  useAssertions,
  useAssertionDetail,
  type AssertionFilters,
  type AssertionItem,
} from '@/api/hooks/useEvidence';

const PAGE_SIZE = 20;

export default function EvidencePage() {
  const [conceptId, setConceptId] = useState('');
  const [predicate, setPredicate] = useState('');
  const [status, setStatus] = useState('');
  const [minConfidence, setMinConfidence] = useState('');
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState<string>();

  const filters: AssertionFilters = {
    concept_id: conceptId || undefined,
    predicate: predicate || undefined,
    status: status || undefined,
    min_confidence: minConfidence && !Number.isNaN(parseFloat(minConfidence)) ? parseFloat(minConfidence) : undefined,
    limit: PAGE_SIZE,
    offset,
  };

  const { data, isLoading, isError, error } = useAssertions(filters);
  const detail = useAssertionDetail(selectedId);

  function handleFilterChange() {
    setOffset(0);
  }

  // Pagination
  const total = data?.total ?? 0;
  const showing = data ? data.assertions.length : 0;
  const rangeStart = total > 0 ? offset + 1 : 0;
  const rangeEnd = offset + showing;
  const hasNext = rangeEnd < total;
  const hasPrev = offset > 0;

  return (
    <>
      <Header title="Evidence Explorer" />
      <div className="p-6">
        {/* Filters row */}
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Concept ID</label>
            <input
              type="text"
              placeholder="e.g. TSMC"
              value={conceptId}
              onChange={(e) => {
                setConceptId(e.target.value);
                handleFilterChange();
              }}
              className="h-8 w-40 rounded border border-border bg-card px-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Predicate</label>
            <input
              type="text"
              placeholder="e.g. supplies_to"
              value={predicate}
              onChange={(e) => {
                setPredicate(e.target.value);
                handleFilterChange();
              }}
              className="h-8 w-40 rounded border border-border bg-card px-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Status</label>
            <select
              value={status}
              onChange={(e) => {
                setStatus(e.target.value);
                handleFilterChange();
              }}
              className="h-8 w-32 rounded border border-border bg-card px-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="">All</option>
              <option value="active">Active</option>
              <option value="disputed">Disputed</option>
              <option value="retracted">Retracted</option>
              <option value="superseded">Superseded</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Min Confidence</label>
            <input
              type="number"
              placeholder="0.0 - 1.0"
              min="0"
              max="1"
              step="0.1"
              value={minConfidence}
              onChange={(e) => {
                setMinConfidence(e.target.value);
                handleFilterChange();
              }}
              className="h-8 w-28 rounded border border-border bg-card px-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>

        {/* Results meta */}
        {data && (
          <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
            <span>
              {total} assertion{total !== 1 && 's'}
            </span>
            <span>{latency(data.latency_ms)}</span>
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load assertions{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Two-panel layout */}
        <div className="mt-4 grid gap-6 lg:grid-cols-[1fr,1fr]">
          {/* Left: Assertions list */}
          <div>
            {isLoading && (
              <div className="space-y-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="rounded-lg border border-border bg-card p-4">
                    <div className="h-4 w-3/4 animate-pulse rounded bg-secondary" />
                    <div className="mt-3 h-3 w-1/2 animate-pulse rounded bg-secondary" />
                    <div className="mt-2 h-2 w-full animate-pulse rounded bg-secondary" />
                  </div>
                ))}
              </div>
            )}

            {data && data.assertions.length === 0 && (
              <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-card py-20 text-muted-foreground">
                <Scale className="h-12 w-12" />
                <p className="mt-3 text-sm">No assertions yet</p>
                <p className="mt-1 max-w-xs text-center text-xs">
                  Assertions are created when lane runs process evidence from news, filings, and
                  structural analysis.
                </p>
              </div>
            )}

            {data && data.assertions.length > 0 && (
              <div className="space-y-3">
                {data.assertions.map((a) => (
                  <AssertionCard
                    key={a.assertion_id}
                    assertion={a}
                    isSelected={selectedId === a.assertion_id}
                    onClick={() => setSelectedId(a.assertion_id)}
                  />
                ))}
              </div>
            )}

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

          {/* Right: Detail panel */}
          <div>
            {!selectedId && (
              <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-card py-20 text-muted-foreground">
                <FileSearch className="h-12 w-12" />
                <p className="mt-3 text-sm">Select an assertion</p>
                <p className="mt-1 text-xs">Click an assertion on the left to view its details and linked claims.</p>
              </div>
            )}

            {selectedId && detail.isLoading && (
              <div className="rounded-lg border border-border bg-card p-6">
                <div className="h-5 w-2/3 animate-pulse rounded bg-secondary" />
                <div className="mt-4 h-4 w-1/3 animate-pulse rounded bg-secondary" />
                <div className="mt-6 space-y-3">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="h-3 w-full animate-pulse rounded bg-secondary" />
                  ))}
                </div>
              </div>
            )}

            {selectedId && detail.isError && (
              <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                Failed to load assertion detail
                {detail.error instanceof Error && `: ${detail.error.message}`}
              </div>
            )}

            {selectedId && detail.data && <AssertionDetail data={detail.data} />}
          </div>
        </div>
      </div>
    </>
  );
}

// ── Assertion Card ──

function AssertionCard({
  assertion,
  isSelected,
  onClick,
}: {
  assertion: AssertionItem;
  isSelected: boolean;
  onClick: () => void;
}) {
  const statusClass = ASSERTION_STATUS_COLORS[assertion.status] ?? 'bg-slate-500/20 text-slate-400';

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full rounded-lg border bg-card p-4 text-left transition-colors hover:border-primary/50',
        isSelected ? 'border-primary ring-1 ring-primary/30' : 'border-border',
      )}
    >
      {/* Triple */}
      <div className="font-mono text-sm text-foreground">
        <span className="text-primary">[{assertion.subject_concept_id}]</span>
        <span className="text-muted-foreground"> —{assertion.predicate}→ </span>
        <span className="text-primary">[{assertion.object_concept_id}]</span>
      </div>

      {/* Confidence bar */}
      <div className="mt-3 flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Confidence</span>
        <div className="h-1.5 flex-1 rounded-full bg-secondary">
          <div
            className="h-1.5 rounded-full bg-primary transition-all"
            style={{ width: `${Math.round(assertion.confidence * 100)}%` }}
          />
        </div>
        <span className="text-xs font-medium text-foreground">
          {pct(assertion.confidence, 0)}
        </span>
      </div>

      {/* Badges row */}
      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <span className={cn('rounded px-1.5 py-0.5 text-xs font-medium', statusClass)}>
          {assertion.status}
        </span>
        <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-xs text-emerald-400">
          {assertion.support_count} support
        </span>
        {assertion.contradiction_count > 0 && (
          <span className="rounded bg-red-500/20 px-1.5 py-0.5 text-xs text-red-400">
            {assertion.contradiction_count} contradiction
          </span>
        )}
        <span className="ml-auto text-xs text-muted-foreground">
          {timeAgo(assertion.last_evidence_at)}
        </span>
      </div>
    </button>
  );
}

// ── Assertion Detail Panel ──

function AssertionDetail({
  data,
}: {
  data: import('@/api/hooks/useEvidence').AssertionDetailResponse;
}) {
  const a = data.assertion;
  const statusClass = ASSERTION_STATUS_COLORS[a.status] ?? 'bg-slate-500/20 text-slate-400';

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      {/* Full triple */}
      <div className="font-mono text-sm text-foreground">
        <span className="text-primary">[{a.subject_concept_id}]</span>
        <span className="text-muted-foreground"> —{a.predicate}→ </span>
        <span className="text-primary">[{a.object_concept_id}]</span>
      </div>

      {/* Confidence */}
      <div className="mt-4 flex items-center gap-3">
        <span className="text-sm text-muted-foreground">Confidence:</span>
        <span className="text-lg font-semibold text-foreground">{pct(a.confidence)}</span>
        <span className={cn('rounded px-1.5 py-0.5 text-xs font-medium', statusClass)}>
          {a.status}
        </span>
      </div>

      {/* Source diversity */}
      <div className="mt-2 text-xs text-muted-foreground">
        Source diversity: <span className="text-foreground">{a.source_diversity}</span>
      </div>

      {/* Temporal validity */}
      <div className="mt-4 rounded border border-border bg-card/50 p-3">
        <h4 className="text-xs font-medium text-muted-foreground">Temporal Validity</h4>
        <div className="mt-1.5 flex items-center gap-2 text-sm text-foreground">
          <span>{a.valid_from ? formatDate(a.valid_from) : 'unbounded'}</span>
          <span className="text-muted-foreground">→</span>
          <span>{a.valid_to ? formatDate(a.valid_to) : 'present'}</span>
        </div>
        <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
          <span>First seen: {timeAgo(a.first_seen_at)}</span>
          <span>Last evidence: {timeAgo(a.last_evidence_at)}</span>
        </div>
      </div>

      {/* Evidence counts summary */}
      <div className="mt-4 flex gap-4">
        <div className="rounded border border-border bg-card/50 px-3 py-2 text-center">
          <div className="text-lg font-semibold text-emerald-400">{a.support_count}</div>
          <div className="text-xs text-muted-foreground">Supporting</div>
        </div>
        <div className="rounded border border-border bg-card/50 px-3 py-2 text-center">
          <div className="text-lg font-semibold text-red-400">{a.contradiction_count}</div>
          <div className="text-xs text-muted-foreground">Contradicting</div>
        </div>
      </div>

      {/* Claims table */}
      <div className="mt-6">
        <h4 className="text-sm font-medium text-foreground">
          Linked Claims ({data.claim_links.length})
        </h4>

        {data.claim_links.length === 0 && (
          <p className="mt-2 text-xs text-muted-foreground">No claims linked to this assertion.</p>
        )}

        {data.claim_links.length > 0 && (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border text-muted-foreground">
                  <th className="pb-2 pr-3 font-medium">Lane</th>
                  <th className="pb-2 pr-3 font-medium">Source</th>
                  <th className="pb-2 pr-3 font-medium">Method</th>
                  <th className="pb-2 pr-3 font-medium">Type</th>
                  <th className="pb-2 pr-3 text-right font-medium">Confidence</th>
                  <th className="pb-2 pr-3 text-right font-medium">Weight</th>
                  <th className="pb-2 font-medium">Created</th>
                </tr>
              </thead>
              <tbody>
                {data.claim_links.filter((link) => link.claim != null).map((link) => {
                  const c = link.claim!;
                  const isDocSource =
                    c.source_type === 'document' ||
                    (c.source_id && (c.source_id.length === 36 || c.source_id.startsWith('doc_')));
                  return (
                    <tr
                      key={link.claim_id}
                      className="border-b border-border/50 last:border-0"
                    >
                      <td className="py-2 pr-3">
                        <span className="rounded bg-primary/10 px-1.5 py-0.5 text-primary">
                          {c.lane}
                        </span>
                      </td>
                      <td className="py-2 pr-3 text-foreground">
                        {isDocSource ? (
                          <Link
                            to={`/documents/${c.source_id}`}
                            className="text-primary hover:underline"
                          >
                            {c.source_id.slice(0, 8)}...
                          </Link>
                        ) : (
                          <span title={c.source_id}>
                            {c.source_type}
                          </span>
                        )}
                      </td>
                      <td className="py-2 pr-3 text-muted-foreground">{c.extraction_method}</td>
                      <td className="py-2 pr-3">
                        <span
                          className={cn(
                            'rounded px-1.5 py-0.5 text-xs',
                            link.link_type === 'support'
                              ? 'bg-emerald-500/20 text-emerald-400'
                              : 'bg-red-500/20 text-red-400',
                          )}
                        >
                          {link.link_type}
                        </span>
                      </td>
                      <td className="py-2 pr-3 text-right text-foreground">
                        {pct(c.confidence, 0)}
                      </td>
                      <td className="py-2 pr-3 text-right text-foreground">
                        {link.contribution_weight.toFixed(2)}
                      </td>
                      <td className="py-2 text-muted-foreground">{timeAgo(c.created_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
