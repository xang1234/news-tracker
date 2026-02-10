import { useState } from 'react';
import { Radio, Plus, ChevronLeft, ChevronRight } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { SourcesTable, SourcesTableSkeleton } from '@/components/domain/SourcesTable';
import {
  SourcesFilters,
  DEFAULT_SOURCES_FILTERS,
  type SourcesFilterValues,
} from '@/components/domain/SourcesFilters';
import { SourceFormModal, type SourceFormData } from '@/components/domain/SourceFormModal';
import {
  useSources,
  useCreateSource,
  useUpdateSource,
  useDeactivateSource,
  type SourceFilters as ApiSourceFilters,
  type SourceItem,
} from '@/api/hooks/useSources';
import { latency } from '@/lib/formatters';

function buildApiFilters(f: SourcesFilterValues, offset: number): ApiSourceFilters {
  return {
    platform: f.platform || undefined,
    search: f.search || undefined,
    active_only: f.activeOnly || undefined,
    limit: f.limit,
    offset,
  };
}

export default function Settings() {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filters, setFilters] = useState<SourcesFilterValues>(DEFAULT_SOURCES_FILTERS);
  const [offset, setOffset] = useState(0);
  const [addOpen, setAddOpen] = useState(false);
  const [editItem, setEditItem] = useState<SourceItem | null>(null);

  const apiFilters = buildApiFilters(filters, offset);
  const { data, isLoading, isError, error } = useSources(apiFilters);
  const createMutation = useCreateSource();
  const updateMutation = useUpdateSource();
  const deactivateMutation = useDeactivateSource();

  function handleFilterChange(next: SourcesFilterValues) {
    setFilters(next);
    setOffset(0);
  }

  // Pagination
  const total = data?.total ?? 0;
  const showing = data ? data.sources.length : 0;
  const rangeStart = total > 0 ? offset + 1 : 0;
  const rangeEnd = offset + showing;
  const hasNext = data?.has_more ?? false;
  const hasPrev = offset > 0;

  function handleEdit(platform: string, identifier: string) {
    const item = data?.sources.find(
      (s) => s.platform === platform && s.identifier === identifier,
    );
    if (item) setEditItem(item);
  }

  function handleDeactivate(platform: string, identifier: string) {
    if (confirm(`Deactivate ${platform}/${identifier}?`)) {
      deactivateMutation.mutate({ platform, identifier });
    }
  }

  // Compute per-platform counts from current page data
  const twitterCount = data?.sources.filter((s) => s.platform === 'twitter').length ?? 0;
  const redditCount = data?.sources.filter((s) => s.platform === 'reddit').length ?? 0;
  const substackCount = data?.sources.filter((s) => s.platform === 'substack').length ?? 0;

  return (
    <>
      <Header title="Settings" />
      <div className="mx-auto max-w-5xl p-6">
        <p className="text-sm text-muted-foreground">
          Manage ingestion sources for Twitter, Reddit, and Substack.
        </p>

        {/* Header row */}
        <div className="mt-4 flex items-center justify-between">
          <div />
          <button
            onClick={() => setAddOpen(true)}
            className="flex items-center gap-1.5 rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Plus className="h-3.5 w-3.5" />
            Add Source
          </button>
        </div>

        {/* Stats */}
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          {isLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            <>
              <MetricCard label="Total Sources" value={total} icon={Radio} />
              <MetricCard label="Twitter" value={twitterCount} />
              <MetricCard label="Reddit" value={redditCount} />
              <MetricCard label="Substack" value={substackCount} />
            </>
          )}
        </div>

        {/* Filters */}
        <div className="mt-4">
          <SourcesFilters
            isOpen={filtersOpen}
            onToggle={() => setFiltersOpen(!filtersOpen)}
            filters={filters}
            onChange={handleFilterChange}
          />
        </div>

        {/* Meta bar */}
        {data && (
          <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
            <span>
              {total} source{total !== 1 ? 's' : ''}
            </span>
            <span>{latency(data.latency_ms)}</span>
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load sources{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Table */}
        <div className="mt-4">
          {isLoading && <SourcesTableSkeleton />}

          {data && data.sources.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <Radio className="h-12 w-12" />
              <p className="mt-3 text-sm">No sources found</p>
              <p className="mt-1 text-xs">Click &quot;Add Source&quot; to create one.</p>
            </div>
          )}

          {data && data.sources.length > 0 && (
            <SourcesTable
              sources={data.sources}
              onEdit={handleEdit}
              onDeactivate={handleDeactivate}
            />
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

        {/* Add modal */}
        <SourceFormModal
          isOpen={addOpen}
          onClose={() => setAddOpen(false)}
          mode="create"
          isPending={createMutation.isPending}
          onSubmit={(vals: SourceFormData) => {
            createMutation.mutate(vals, {
              onSuccess: () => setAddOpen(false),
            });
          }}
        />

        {/* Edit modal */}
        {editItem && (
          <SourceFormModal
            isOpen={!!editItem}
            onClose={() => setEditItem(null)}
            mode="edit"
            initialValues={{
              platform: editItem.platform,
              identifier: editItem.identifier,
              display_name: editItem.display_name,
              description: editItem.description,
            }}
            isPending={updateMutation.isPending}
            onSubmit={(vals: SourceFormData) => {
              updateMutation.mutate(
                {
                  platform: editItem.platform,
                  identifier: editItem.identifier,
                  display_name: vals.display_name,
                  description: vals.description,
                },
                { onSuccess: () => setEditItem(null) },
              );
            }}
          />
        )}
      </div>
    </>
  );
}
