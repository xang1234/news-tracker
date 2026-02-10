import { useState } from 'react';
import { Layers, TrendingUp, FileText, ChevronLeft, ChevronRight } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { ThemeCard, ThemeCardSkeleton } from '@/components/domain/ThemeCard';
import {
  ThemeFilters,
  DEFAULT_THEME_FILTERS,
  type ThemeFilterValues,
} from '@/components/domain/ThemeFilters';
import {
  useThemes,
  useRankedThemes,
  type ThemeFilters as ApiThemeFilters,
} from '@/api/hooks/useThemes';
import { latency } from '@/lib/formatters';

type ViewMode = 'all' | 'ranked';

function buildApiFilters(f: ThemeFilterValues, offset: number): ApiThemeFilters {
  return {
    lifecycle_stage: f.lifecycleStage || undefined,
    limit: f.limit,
    offset,
  };
}

export default function Themes() {
  const [viewMode, setViewMode] = useState<ViewMode>('all');
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filters, setFilters] = useState<ThemeFilterValues>(DEFAULT_THEME_FILTERS);
  const [offset, setOffset] = useState(0);

  const apiFilters = buildApiFilters(filters, offset);
  const allThemes = useThemes(viewMode === 'all' ? apiFilters : undefined);
  const rankedThemes = useRankedThemes(
    viewMode === 'ranked' ? 'swing' : undefined,
    undefined,
    viewMode === 'ranked' ? 50 : undefined,
  );

  const activeQuery = viewMode === 'all' ? allThemes : rankedThemes;
  const isLoading = activeQuery.isLoading;
  const isError = activeQuery.isError;
  const error = activeQuery.error;

  function handleFilterChange(next: ThemeFilterValues) {
    setFilters(next);
    setOffset(0);
  }

  // Pagination (all themes view only)
  const total = viewMode === 'all' ? (allThemes.data?.total ?? 0) : (rankedThemes.data?.total ?? 0);
  const items = viewMode === 'all'
    ? (allThemes.data?.themes ?? [])
    : (rankedThemes.data?.themes ?? []);
  const showing = items.length;
  const rangeStart = total > 0 ? offset + 1 : 0;
  const rangeEnd = offset + showing;
  const hasNext = viewMode === 'all' && rangeEnd < total;
  const hasPrev = viewMode === 'all' && offset > 0;

  // Stats from all themes
  const themeCount = allThemes.data?.total ?? 0;
  const topStage = allThemes.data?.themes?.length
    ? getMostCommonStage(allThemes.data.themes.map((t) => t.lifecycle_stage))
    : '—';
  const avgDocs = allThemes.data?.themes?.length
    ? Math.round(allThemes.data.themes.reduce((s, t) => s + t.document_count, 0) / allThemes.data.themes.length)
    : 0;

  const activeFilterCount = [filters.lifecycleStage].filter(Boolean).length;

  return (
    <>
      <Header title="Themes" />
      <div className="mx-auto max-w-4xl p-6">
        {/* Stats row */}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {allThemes.isLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            <>
              <MetricCard label="Total Themes" value={themeCount} icon={Layers} />
              <MetricCard label="Top Stage" value={topStage} icon={TrendingUp} />
              <MetricCard label="Avg Documents" value={avgDocs} icon={FileText} />
            </>
          )}
        </div>

        {/* View mode toggle */}
        <div className="mt-4 flex items-center gap-2">
          <button
            type="button"
            onClick={() => { setViewMode('all'); setOffset(0); }}
            className={`rounded border px-3 py-1.5 text-sm ${
              viewMode === 'all'
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border text-muted-foreground hover:bg-secondary/50'
            }`}
          >
            All Themes
          </button>
          <button
            type="button"
            onClick={() => setViewMode('ranked')}
            className={`rounded border px-3 py-1.5 text-sm ${
              viewMode === 'ranked'
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border text-muted-foreground hover:bg-secondary/50'
            }`}
          >
            Ranked
          </button>
        </div>

        {/* Filter panel (all view only) */}
        {viewMode === 'all' && (
          <div className="mt-4">
            <ThemeFilters
              isOpen={filtersOpen}
              onToggle={() => setFiltersOpen(!filtersOpen)}
              filters={filters}
              onChange={handleFilterChange}
            />
          </div>
        )}

        {/* Results meta bar */}
        {activeQuery.data && (
          <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
            <span>
              {total} theme{total !== 1 && 's'}
            </span>
            <span>{latency(activeQuery.data.latency_ms)}</span>
            {viewMode === 'ranked' && rankedThemes.data && (
              <span className="rounded-full bg-primary/20 px-2 py-0.5 text-primary">
                Strategy: {rankedThemes.data.strategy}
              </span>
            )}
            {viewMode === 'all' && activeFilterCount > 0 && (
              <span className="rounded-full bg-primary/20 px-2 py-0.5 text-primary">
                {activeFilterCount} filter{activeFilterCount !== 1 && 's'} active
              </span>
            )}
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load themes{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Results / states */}
        <div className="mt-4">
          {isLoading && (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <ThemeCardSkeleton key={i} />
              ))}
            </div>
          )}

          {activeQuery.data && items.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <Layers className="h-12 w-12" />
              <p className="mt-3 text-sm">No themes found</p>
              <p className="mt-1 text-xs">
                Themes are created during clustering. Run daily clustering or check backend.
              </p>
            </div>
          )}

          {activeQuery.data && items.length > 0 && (
            <div className="space-y-3">
              {viewMode === 'all'
                ? allThemes.data!.themes.map((theme) => (
                    <ThemeCard key={theme.theme_id} theme={theme} />
                  ))
                : rankedThemes.data!.themes.map((rt) => (
                    <ThemeCard
                      key={rt.theme.theme_id}
                      theme={rt.theme}
                      score={rt.score}
                      tier={rt.tier}
                    />
                  ))}
            </div>
          )}
        </div>

        {/* Pagination (all view) */}
        {viewMode === 'all' && allThemes.data && total > 0 && (
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

function getMostCommonStage(stages: string[]): string {
  const counts: Record<string, number> = {};
  for (const s of stages) {
    counts[s] = (counts[s] ?? 0) + 1;
  }
  let best = '—';
  let max = 0;
  for (const [stage, count] of Object.entries(counts)) {
    if (count > max) {
      max = count;
      best = stage;
    }
  }
  return best;
}
