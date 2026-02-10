import { useState } from 'react';
import { Shield, Plus, ChevronLeft, ChevronRight } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { SecuritiesTable, SecuritiesTableSkeleton } from '@/components/domain/SecuritiesTable';
import {
  SecuritiesFilters,
  DEFAULT_SECURITIES_FILTERS,
  type SecuritiesFilterValues,
} from '@/components/domain/SecuritiesFilters';
import { SecurityFormModal } from '@/components/domain/SecurityFormModal';
import {
  useSecurities,
  useCreateSecurity,
  useUpdateSecurity,
  useDeactivateSecurity,
  type SecurityFilters as ApiSecurityFilters,
  type SecurityItem,
} from '@/api/hooks/useSecurities';
import { latency } from '@/lib/formatters';

function buildApiFilters(f: SecuritiesFilterValues, offset: number): ApiSecurityFilters {
  return {
    search: f.search || undefined,
    active_only: f.activeOnly || undefined,
    exchange: f.exchange || undefined,
    limit: f.limit,
    offset,
  };
}

export default function Securities() {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filters, setFilters] = useState<SecuritiesFilterValues>(DEFAULT_SECURITIES_FILTERS);
  const [offset, setOffset] = useState(0);
  const [addOpen, setAddOpen] = useState(false);
  const [editItem, setEditItem] = useState<SecurityItem | null>(null);

  const apiFilters = buildApiFilters(filters, offset);
  const { data, isLoading, isError, error } = useSecurities(apiFilters);
  const createMutation = useCreateSecurity();
  const updateMutation = useUpdateSecurity();
  const deactivateMutation = useDeactivateSecurity();

  function handleFilterChange(next: SecuritiesFilterValues) {
    setFilters(next);
    setOffset(0);
  }

  // Pagination
  const total = data?.total ?? 0;
  const showing = data ? data.securities.length : 0;
  const rangeStart = total > 0 ? offset + 1 : 0;
  const rangeEnd = offset + showing;
  const hasNext = data?.has_more ?? false;
  const hasPrev = offset > 0;

  function handleEdit(ticker: string, exchange: string) {
    const item = data?.securities.find((s) => s.ticker === ticker && s.exchange === exchange);
    if (item) setEditItem(item);
  }

  function handleDeactivate(ticker: string, exchange: string) {
    if (confirm(`Deactivate ${ticker} (${exchange})?`)) {
      deactivateMutation.mutate({ ticker, exchange });
    }
  }

  return (
    <>
      <Header title="Security Master" />
      <div className="mx-auto max-w-5xl p-6">
        {/* Header row */}
        <div className="flex items-center justify-between">
          <div />
          <button
            onClick={() => setAddOpen(true)}
            className="flex items-center gap-1.5 rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Plus className="h-3.5 w-3.5" />
            Add Security
          </button>
        </div>

        {/* Stats */}
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3">
          {isLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            <>
              <MetricCard label="Total Securities" value={total} icon={Shield} />
              <MetricCard
                label="Active"
                value={data?.securities.filter((s) => s.is_active).length ?? 0}
              />
            </>
          )}
        </div>

        {/* Filters */}
        <div className="mt-4">
          <SecuritiesFilters
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
              {total} securit{total !== 1 ? 'ies' : 'y'}
            </span>
            <span>{latency(data.latency_ms)}</span>
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load securities{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Table */}
        <div className="mt-4">
          {isLoading && <SecuritiesTableSkeleton />}

          {data && data.securities.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <Shield className="h-12 w-12" />
              <p className="mt-3 text-sm">No securities found</p>
              <p className="mt-1 text-xs">Click "Add Security" to create one.</p>
            </div>
          )}

          {data && data.securities.length > 0 && (
            <SecuritiesTable
              securities={data.securities}
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
        <SecurityFormModal
          isOpen={addOpen}
          onClose={() => setAddOpen(false)}
          mode="create"
          onSubmit={(vals) => {
            createMutation.mutate(vals, {
              onSuccess: () => setAddOpen(false),
            });
          }}
        />

        {/* Edit modal */}
        {editItem && (
          <SecurityFormModal
            isOpen={!!editItem}
            onClose={() => setEditItem(null)}
            mode="edit"
            initialValues={editItem}
            onSubmit={(vals) => {
              updateMutation.mutate(
                { ticker: editItem.ticker, exchange: editItem.exchange, ...vals },
                { onSuccess: () => setEditItem(null) },
              );
            }}
          />
        )}
      </div>
    </>
  );
}
