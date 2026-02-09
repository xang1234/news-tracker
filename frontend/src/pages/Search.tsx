import { useState, type FormEvent } from 'react';
import { Search as SearchIcon, SearchX } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { useSearch, type SearchFilters as SearchFilterParams } from '@/api/hooks/useSearch';
import { SearchResultCard, SearchResultSkeleton } from '@/components/domain/SearchResultCard';
import {
  SearchFilters,
  type SearchFilterValues,
} from '@/components/domain/SearchFilters';
import { latency } from '@/lib/formatters';

const DEFAULT_FILTERS: SearchFilterValues = {
  platforms: [],
  tickers: '',
  threshold: 0.7,
  minAuthority: null,
  limit: 10,
};

function buildApiFilters(f: SearchFilterValues): SearchFilterParams {
  const tickers = f.tickers
    .split(',')
    .map((t) => t.trim().toUpperCase())
    .filter(Boolean);

  return {
    platforms: f.platforms.length ? f.platforms : undefined,
    tickers: tickers.length ? tickers : undefined,
    threshold: f.threshold,
    min_authority_score: f.minAuthority ?? undefined,
    limit: f.limit,
  };
}

export default function Search() {
  const [queryText, setQueryText] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filters, setFilters] = useState<SearchFilterValues>(DEFAULT_FILTERS);

  const apiFilters = buildApiFilters(filters);
  const { data, isLoading, isError, error } = useSearch(submittedQuery, apiFilters);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmittedQuery(queryText.trim());
  }

  return (
    <>
      <Header title="Semantic Search" />
      <div className="mx-auto max-w-4xl p-6">
        {/* Search bar */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          <div className="relative flex-1">
            <SearchIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="Search semiconductor news..."
              autoFocus
              className="w-full rounded-lg border border-border bg-background py-2.5 pl-10 pr-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <button
            type="submit"
            disabled={!queryText.trim()}
            className="rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            Search
          </button>
        </form>

        {/* Filter panel */}
        <div className="mt-4">
          <SearchFilters
            isOpen={filtersOpen}
            onToggle={() => setFiltersOpen(!filtersOpen)}
            filters={filters}
            onChange={setFilters}
          />
        </div>

        {/* Metadata bar */}
        {data && (
          <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
            <span>
              {data.total} result{data.total !== 1 && 's'}
            </span>
            <span>{latency(data.latency_ms)}</span>
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Search failed{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Results / states */}
        <div className="mt-4">
          {/* Idle — no query submitted yet */}
          {!submittedQuery && !isLoading && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <SearchIcon className="h-12 w-12" />
              <p className="mt-3 text-sm">Enter a query to search semiconductor news</p>
            </div>
          )}

          {/* Loading */}
          {isLoading && (
            <div className="space-y-3">
              <SearchResultSkeleton />
              <SearchResultSkeleton />
              <SearchResultSkeleton />
            </div>
          )}

          {/* No results */}
          {data && data.results.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <SearchX className="h-12 w-12" />
              <p className="mt-3 text-sm">No results found</p>
              <p className="mt-1 text-xs">
                Try adjusting the similarity threshold or broadening your query
              </p>
            </div>
          )}

          {/* Results */}
          {data && data.results.length > 0 && (
            <div className="space-y-3">
              {data.results.map((result) => (
                <SearchResultCard key={result.document_id} result={result} />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
