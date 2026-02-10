import { useState } from 'react';
import { Bell, AlertTriangle, CheckCircle, ChevronLeft, ChevronRight } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { AlertCard, AlertCardSkeleton } from '@/components/domain/AlertCard';
import {
  AlertFilters,
  DEFAULT_ALERT_FILTERS,
  type AlertFilterValues,
} from '@/components/domain/AlertFilters';
import {
  useAlerts,
  useAcknowledgeAlert,
  type AlertFilters as ApiAlertFilters,
} from '@/api/hooks/useAlerts';
import { latency } from '@/lib/formatters';

function buildApiFilters(f: AlertFilterValues, offset: number): ApiAlertFilters {
  return {
    severity: f.severity || undefined,
    trigger_type: f.triggerType || undefined,
    acknowledged: f.acknowledged === '' ? undefined : f.acknowledged === 'true',
    limit: f.limit,
    offset,
  };
}

export default function Alerts() {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filters, setFilters] = useState<AlertFilterValues>(DEFAULT_ALERT_FILTERS);
  const [offset, setOffset] = useState(0);

  const apiFilters = buildApiFilters(filters, offset);
  const { data, isLoading, isError, error } = useAlerts(apiFilters);
  const acknowledge = useAcknowledgeAlert();

  function handleFilterChange(next: AlertFilterValues) {
    setFilters(next);
    setOffset(0);
  }

  // Pagination
  const total = data?.total ?? 0;
  const showing = data ? data.alerts.length : 0;
  const rangeStart = total > 0 ? offset + 1 : 0;
  const rangeEnd = offset + showing;
  const hasNext = rangeEnd < total;
  const hasPrev = offset > 0;

  // Stats
  const criticalCount = data?.alerts.filter((a) => a.severity === 'critical').length ?? 0;
  const unackCount = data?.alerts.filter((a) => !a.acknowledged).length ?? 0;

  const activeFilterCount = [
    filters.severity,
    filters.triggerType,
    filters.acknowledged,
  ].filter(Boolean).length;

  return (
    <>
      <Header title="Alerts" />
      <div className="mx-auto max-w-4xl p-6">
        {/* Stats row */}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {isLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            <>
              <MetricCard label="Total Alerts" value={total} icon={Bell} />
              <MetricCard
                label="Critical"
                value={criticalCount}
                icon={AlertTriangle}
                trend={criticalCount > 0 ? 'up' : 'neutral'}
              />
              <MetricCard
                label="Unacknowledged"
                value={unackCount}
                icon={CheckCircle}
                trend={unackCount > 0 ? 'up' : 'neutral'}
              />
            </>
          )}
        </div>

        {/* Filter panel */}
        <div className="mt-4">
          <AlertFilters
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
              {total} alert{total !== 1 && 's'}
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
            Failed to load alerts{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Results / states */}
        <div className="mt-4">
          {isLoading && (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <AlertCardSkeleton key={i} />
              ))}
            </div>
          )}

          {data && data.alerts.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <Bell className="h-12 w-12" />
              <p className="mt-3 text-sm">No alerts found</p>
              <p className="mt-1 text-xs">
                Alerts are triggered by theme volume spikes, sentiment shifts, and other signals.
              </p>
            </div>
          )}

          {data && data.alerts.length > 0 && (
            <div className="space-y-3">
              {data.alerts.map((alert) => (
                <AlertCard
                  key={alert.alert_id}
                  alert={alert}
                  onAcknowledge={(id) => acknowledge.mutate(id)}
                  isAcknowledging={acknowledge.isPending}
                />
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
