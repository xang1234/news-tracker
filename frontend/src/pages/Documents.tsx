import { useState } from 'react';
import { FileText, Database, Cpu, BarChart3, Clock, ChevronLeft, ChevronRight } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { DocumentCard, DocumentCardSkeleton } from '@/components/domain/DocumentCard';
import {
  DocumentFilters,
  DEFAULT_DOCUMENT_FILTERS,
  type DocumentFilterValues,
} from '@/components/domain/DocumentFilters';
import {
  useDocuments,
  useDocumentStats,
  type DocumentFilters as ApiFilters,
} from '@/api/hooks/useDocuments';
import { latency, timeAgo, pct } from '@/lib/formatters';

function buildApiFilters(f: DocumentFilterValues, offset: number): ApiFilters {
  return {
    platform: f.platform || undefined,
    content_type: f.contentType || undefined,
    ticker: f.ticker || undefined,
    q: f.searchText || undefined,
    max_spam: f.maxSpam ?? undefined,
    min_authority: f.minAuthority ?? undefined,
    sort: f.sort,
    order: f.order,
    limit: f.limit,
    offset,
  };
}

export default function Documents() {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filters, setFilters] = useState<DocumentFilterValues>(DEFAULT_DOCUMENT_FILTERS);
  const [offset, setOffset] = useState(0);

  const apiFilters = buildApiFilters(filters, offset);
  const { data, isLoading, isError, error } = useDocuments(apiFilters);
  const { data: stats, isLoading: statsLoading } = useDocumentStats();

  function handleFilterChange(next: DocumentFilterValues) {
    setFilters(next);
    setOffset(0);
  }

  // Pagination helpers
  const total = data?.total ?? 0;
  const showing = data ? Math.min(data.documents.length, total) : 0;
  const rangeStart = total > 0 ? offset + 1 : 0;
  const rangeEnd = offset + showing;
  const hasNext = rangeEnd < total;
  const hasPrev = offset > 0;

  // Stats helpers
  const topPlatform = stats?.platform_counts?.length
    ? stats.platform_counts.reduce((a, b) => (b.count > a.count ? b : a))
    : null;

  // Active filter count for indicator
  const activeFilterCount = [
    filters.platform,
    filters.contentType,
    filters.ticker,
    filters.searchText,
    filters.maxSpam != null,
    filters.minAuthority != null,
  ].filter(Boolean).length;

  return (
    <>
      <Header title="Documents" />
      <div className="mx-auto max-w-4xl p-6">
        {/* Stats row */}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {statsLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : stats ? (
            <>
              <MetricCard
                label="Total Documents"
                value={stats.total_count.toLocaleString()}
                icon={Database}
              />
              <MetricCard
                label="Top Platform"
                value={topPlatform?.platform ?? '—'}
                subtitle={topPlatform ? `${topPlatform.count.toLocaleString()} docs` : undefined}
                icon={BarChart3}
              />
              <MetricCard
                label="Embedding Coverage"
                value={pct(stats.embedding_coverage.finbert_pct)}
                subtitle={`MiniLM: ${pct(stats.embedding_coverage.minilm_pct)}`}
                icon={Cpu}
              />
              <MetricCard
                label="Latest Ingestion"
                value={timeAgo(stats.latest_fetched_at ?? stats.latest_document)}
                icon={Clock}
              />
            </>
          ) : (
            <>
              <MetricCard label="Total Documents" value="—" icon={Database} />
              <MetricCard label="Top Platform" value="—" icon={BarChart3} />
              <MetricCard label="Embedding Coverage" value="—" icon={Cpu} />
              <MetricCard label="Latest Ingestion" value="—" icon={Clock} />
            </>
          )}
        </div>

        {/* Filter panel */}
        <div className="mt-4">
          <DocumentFilters
            isOpen={filtersOpen}
            onToggle={() => setFiltersOpen(!filtersOpen)}
            filters={filters}
            onChange={handleFilterChange}
          />
        </div>

        {/* Results meta bar */}
        {data && (
          <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
            <span>
              {total.toLocaleString()} document{total !== 1 && 's'}
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
            Failed to load documents{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Results / states */}
        <div className="mt-4">
          {/* Loading */}
          {isLoading && (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <DocumentCardSkeleton key={i} />
              ))}
            </div>
          )}

          {/* Empty */}
          {data && data.documents.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <FileText className="h-12 w-12" />
              <p className="mt-3 text-sm">No documents found</p>
              <p className="mt-1 text-xs">
                Try adjusting filters or check that the backend endpoints are available
              </p>
            </div>
          )}

          {/* Document list */}
          {data && data.documents.length > 0 && (
            <div className="space-y-3">
              {data.documents.map((doc) => (
                <DocumentCard key={doc.document_id} document={doc} />
              ))}
            </div>
          )}
        </div>

        {/* Pagination */}
        {data && total > 0 && (
          <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground">
            <span>
              Showing {rangeStart}–{rangeEnd} of {total.toLocaleString()}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={!hasPrev}
                onClick={() => setOffset(Math.max(0, offset - filters.limit))}
                className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
                Previous
              </button>
              <button
                type="button"
                disabled={!hasNext}
                onClick={() => setOffset(offset + filters.limit)}
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
