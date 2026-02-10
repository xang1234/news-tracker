import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, TrendingUp, Hash, ChevronLeft, ChevronRight } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { EntityCard, EntityCardSkeleton } from '@/components/domain/EntityCard';
import { TrendingEntityCard, TrendingEntityCardSkeleton } from '@/components/domain/TrendingEntityCard';
import {
  EntityFilters,
  DEFAULT_ENTITY_FILTERS,
  type EntityFilterValues,
} from '@/components/domain/EntityFilters';
import {
  useEntities,
  useEntityStats,
  useTrendingEntities,
  type EntityFilters as ApiEntityFilters,
} from '@/api/hooks/useEntities';
import { latency } from '@/lib/formatters';
import { cn } from '@/lib/utils';

type Tab = 'all' | 'trending';

function buildApiFilters(f: EntityFilterValues, offset: number): ApiEntityFilters {
  return {
    entity_type: f.entity_type || undefined,
    search: f.search || undefined,
    sort: f.sort,
    limit: f.limit,
    offset,
  };
}

export default function Entities() {
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>('all');
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filters, setFilters] = useState<EntityFilterValues>(DEFAULT_ENTITY_FILTERS);
  const [offset, setOffset] = useState(0);

  const apiFilters = buildApiFilters(filters, offset);
  const { data, isLoading, isError, error } = useEntities(apiFilters);
  const stats = useEntityStats();
  const trending = useTrendingEntities();

  function handleFilterChange(next: EntityFilterValues) {
    setFilters(next);
    setOffset(0);
  }

  // Pagination
  const total = data?.total ?? 0;
  const showing = data ? data.entities.length : 0;
  const rangeStart = total > 0 ? offset + 1 : 0;
  const rangeEnd = offset + showing;
  const hasNext = data?.has_more ?? false;
  const hasPrev = offset > 0;

  const activeFilterCount = [filters.entity_type, filters.search].filter(Boolean).length;

  // Top type from stats
  const topType = stats.data
    ? Object.entries(stats.data.by_type).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—'
    : '—';

  return (
    <>
      <Header title="Entity Explorer" />
      <div className="mx-auto max-w-4xl p-6">
        {/* Stats row */}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {stats.isLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            <>
              <MetricCard
                label="Total Entities"
                value={stats.data?.total_entities ?? 0}
                icon={Users}
              />
              <MetricCard
                label="Docs with Entities"
                value={stats.data?.documents_with_entities ?? 0}
                icon={Hash}
              />
              <MetricCard label="Top Type" value={topType} icon={TrendingUp} />
            </>
          )}
        </div>

        {/* Tab toggle */}
        <div className="mt-6 flex gap-1 rounded-lg border border-border bg-card p-1">
          {(['all', 'trending'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                'flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                tab === t
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {t === 'all' ? 'All' : 'Trending'}
            </button>
          ))}
        </div>

        {/* All tab */}
        {tab === 'all' && (
          <>
            <div className="mt-4">
              <EntityFilters
                isOpen={filtersOpen}
                onToggle={() => setFiltersOpen(!filtersOpen)}
                filters={filters}
                onChange={handleFilterChange}
              />
            </div>

            {data && (
              <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
                <span>
                  {total} entit{total !== 1 ? 'ies' : 'y'}
                </span>
                <span>{latency(data.latency_ms)}</span>
                {activeFilterCount > 0 && (
                  <span className="rounded-full bg-primary/20 px-2 py-0.5 text-primary">
                    {activeFilterCount} filter{activeFilterCount !== 1 && 's'} active
                  </span>
                )}
              </div>
            )}

            {isError && (
              <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                Failed to load entities{error instanceof Error && `: ${error.message}`}
              </div>
            )}

            <div className="mt-4">
              {isLoading && (
                <div className="space-y-3">
                  {Array.from({ length: 8 }).map((_, i) => (
                    <EntityCardSkeleton key={i} />
                  ))}
                </div>
              )}

              {data && data.entities.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                  <Users className="h-12 w-12" />
                  <p className="mt-3 text-sm">No entities found</p>
                  <p className="mt-1 text-xs">
                    Entities are extracted by the NER pipeline from ingested documents.
                  </p>
                </div>
              )}

              {data && data.entities.length > 0 && (
                <div className="space-y-3">
                  {data.entities.map((e) => (
                    <EntityCard
                      key={`${e.type}:${e.normalized}`}
                      type={e.type}
                      normalized={e.normalized}
                      mention_count={e.mention_count}
                      first_seen={e.first_seen}
                      last_seen={e.last_seen}
                      onClick={() =>
                        navigate(
                          `/entities/${encodeURIComponent(e.type)}/${encodeURIComponent(e.normalized)}`,
                        )
                      }
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
          </>
        )}

        {/* Trending tab */}
        {tab === 'trending' && (
          <div className="mt-4">
            {trending.isLoading && (
              <div className="space-y-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <TrendingEntityCardSkeleton key={i} />
                ))}
              </div>
            )}

            {trending.isError && (
              <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                Failed to load trending entities
              </div>
            )}

            {trending.data && trending.data.trending.length === 0 && (
              <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                <TrendingUp className="h-12 w-12" />
                <p className="mt-3 text-sm">No trending entities</p>
                <p className="mt-1 text-xs">
                  Entities spike when their mention rate increases vs. the baseline window.
                </p>
              </div>
            )}

            {trending.data && trending.data.trending.length > 0 && (
              <div className="space-y-3">
                {trending.data.trending.map((e) => (
                  <TrendingEntityCard
                    key={`${e.type}:${e.normalized}`}
                    type={e.type}
                    normalized={e.normalized}
                    recent_count={e.recent_count}
                    baseline_count={e.baseline_count}
                    spike_ratio={e.spike_ratio}
                    onClick={() =>
                      navigate(
                        `/entities/${encodeURIComponent(e.type)}/${encodeURIComponent(e.normalized)}`,
                      )
                    }
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}
