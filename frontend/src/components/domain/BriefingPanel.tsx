import { Link } from 'react-router-dom';
import { FileText, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';
import { generatedByBadge, timeAgo } from '@/lib/formatters';
import type {
  ClaimCitationItem,
  ThemeBriefingResponse,
} from '@/api/hooks/useThemeBriefing';

interface BriefingPanelProps {
  data?: ThemeBriefingResponse;
  isLoading: boolean;
  isError: boolean;
  error?: unknown;
}

/** A small superscript citation marker that links to its evidence source. */
function CitationChip({ n, citation }: { n: number; citation?: ClaimCitationItem }) {
  const label = citation
    ? `${citation.subject_text} ${citation.predicate.replace(/_/g, ' ')}${
        citation.object_text ? ` ${citation.object_text}` : ''
      }`
    : 'source';
  const chipClass =
    'ml-0.5 inline-flex items-center rounded bg-primary/15 px-1 text-[10px] font-semibold text-primary align-super hover:bg-primary/30';

  // Document-backed citations deep-link to the source document; other source
  // types (filing sections, graph edges) have no document page, so render a
  // non-link marker with the claim as a tooltip.
  if (citation && citation.source_type === 'document' && citation.source_id) {
    return (
      <Link to={`/documents/${citation.source_id}`} title={label} className={chipClass}>
        [{n}]
      </Link>
    );
  }
  return (
    <span title={label} className={cn(chipClass, 'cursor-default hover:bg-primary/15')}>
      [{n}]
    </span>
  );
}

export function BriefingPanel({ data, isLoading, isError, error }: BriefingPanelProps) {
  if (isLoading) return <BriefingPanelSkeleton />;

  if (isError) {
    return (
      <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
        Failed to load briefing{error instanceof Error && `: ${error.message}`}
      </div>
    );
  }

  if (!data || data.clauses.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-card py-16 text-muted-foreground">
        <Sparkles className="h-10 w-10" />
        <p className="mt-3 text-sm">No briefing available yet</p>
        <p className="mt-1 text-xs">
          Needs indexed claims for this theme (and{' '}
          <code className="text-foreground">theme_briefing_enabled</code>).
        </p>
      </div>
    );
  }

  // Assign each cited claim a stable 1-based index in first-appearance order.
  const citationByClaim = new Map(data.citations.map((c) => [c.claim_id, c]));
  const indexByClaim = new Map<string, number>();
  for (const clause of data.clauses) {
    for (const cid of clause.claim_ids) {
      if (!indexByClaim.has(cid)) indexByClaim.set(cid, indexByClaim.size + 1);
    }
  }
  const orderedCitations = [...indexByClaim.entries()].map(([cid, n]) => ({
    n,
    cid,
    citation: citationByClaim.get(cid),
  }));

  const badge = generatedByBadge(data.generated_by);

  return (
    <div className="space-y-5">
      {/* Meta */}
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span className={cn('rounded-full px-2 py-0.5 font-medium', badge.className)}>
          {badge.label}
        </span>
        <span>
          {data.claim_count} grounding claim{data.claim_count !== 1 ? 's' : ''}
        </span>
        {data.model && <span>· {data.model}</span>}
        {data.generated_at && <span>· {timeAgo(data.generated_at)}</span>}
      </div>

      {/* Prose with inline citations */}
      <div className="rounded-lg border border-border bg-card p-4 leading-relaxed text-foreground">
        {data.clauses.map((clause, i) => (
          <span key={i}>
            {clause.text}
            {clause.claim_ids.map((cid) => (
              <CitationChip key={cid} n={indexByClaim.get(cid) ?? 0} citation={citationByClaim.get(cid)} />
            ))}{' '}
          </span>
        ))}
      </div>

      {/* Sources */}
      {orderedCitations.length > 0 && (
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Sources
          </h4>
          <ol className="space-y-2">
            {orderedCitations.map(({ n, cid, citation }) => (
              <li key={cid} className="flex gap-2 text-sm">
                <span className="font-semibold text-primary">[{n}]</span>
                <div className="min-w-0">
                  {citation ? (
                    <>
                      <span className="text-foreground">
                        {citation.subject_text} {citation.predicate.replace(/_/g, ' ')}
                        {citation.object_text ? ` ${citation.object_text}` : ''}
                      </span>
                      {citation.snippet && (
                        <p className="mt-0.5 truncate text-xs text-muted-foreground">
                          “{citation.snippet}”
                        </p>
                      )}
                      {citation.source_type === 'document' && citation.source_id && (
                        <Link
                          to={`/documents/${citation.source_id}`}
                          className="mt-0.5 inline-flex items-center gap-1 text-xs text-primary hover:underline"
                        >
                          <FileText className="h-3 w-3" /> View source
                        </Link>
                      )}
                    </>
                  ) : (
                    <span className="text-muted-foreground">{cid}</span>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

export function BriefingPanelSkeleton() {
  return (
    <div className="space-y-5">
      <div className="h-4 w-48 animate-pulse rounded bg-secondary" />
      <div className="space-y-2 rounded-lg border border-border bg-card p-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-4 w-full animate-pulse rounded bg-secondary" />
        ))}
      </div>
    </div>
  );
}
