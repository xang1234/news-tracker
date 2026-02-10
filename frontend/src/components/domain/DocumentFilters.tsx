import { SlidersHorizontal, ChevronDown, ChevronUp, ArrowUpDown } from 'lucide-react';
import { PLATFORMS } from '@/lib/constants';
import { pct } from '@/lib/formatters';

export interface DocumentFilterValues {
  platform: string;
  contentType: string;
  ticker: string;
  searchText: string;
  maxSpam: number | null;
  minAuthority: number | null;
  sort: string;
  order: string;
  limit: number;
}

export const DEFAULT_DOCUMENT_FILTERS: DocumentFilterValues = {
  platform: '',
  contentType: '',
  ticker: '',
  searchText: '',
  maxSpam: null,
  minAuthority: null,
  sort: 'timestamp',
  order: 'desc',
  limit: 25,
};

interface DocumentFiltersProps {
  isOpen: boolean;
  onToggle: () => void;
  filters: DocumentFilterValues;
  onChange: (filters: DocumentFilterValues) => void;
}

const PLATFORM_KEYS = Object.keys(PLATFORMS);
const CONTENT_TYPES = ['post', 'comment', 'article'];
const SORT_OPTIONS = [
  { value: 'timestamp', label: 'Newest' },
  { value: 'authority_score', label: 'Authority' },
  { value: 'spam_score', label: 'Spam' },
];
const LIMIT_OPTIONS = [25, 50, 100];

export function DocumentFilters({
  isOpen,
  onToggle,
  filters,
  onChange,
}: DocumentFiltersProps) {
  const Chevron = isOpen ? ChevronUp : ChevronDown;

  return (
    <div className="rounded-lg border border-border bg-card">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-2 px-4 py-3 text-sm font-medium text-foreground hover:bg-secondary/50"
      >
        <SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
        Filters
        <Chevron className="ml-auto h-4 w-4 text-muted-foreground" />
      </button>

      {isOpen && (
        <div className="grid grid-cols-1 gap-4 border-t border-border px-4 py-4 sm:grid-cols-2 lg:grid-cols-3">
          {/* Platform */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Platform
            </label>
            <select
              value={filters.platform}
              onChange={(e) => onChange({ ...filters, platform: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All Platforms</option>
              {PLATFORM_KEYS.map((key) => (
                <option key={key} value={key}>
                  {PLATFORMS[key].label}
                </option>
              ))}
            </select>
          </div>

          {/* Content type */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Content Type
            </label>
            <select
              value={filters.contentType}
              onChange={(e) => onChange({ ...filters, contentType: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All Types</option>
              {CONTENT_TYPES.map((ct) => (
                <option key={ct} value={ct}>
                  {ct.charAt(0).toUpperCase() + ct.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Ticker */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Ticker
            </label>
            <input
              type="text"
              value={filters.ticker}
              onChange={(e) =>
                onChange({ ...filters, ticker: e.target.value.toUpperCase() })
              }
              placeholder="e.g. NVDA"
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>

          {/* Full-text search */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Search Text
            </label>
            <input
              type="text"
              value={filters.searchText}
              onChange={(e) => onChange({ ...filters, searchText: e.target.value })}
              placeholder="Full-text search..."
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>

          {/* Max spam */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Max spam score: {filters.maxSpam != null ? pct(filters.maxSpam) : 'Any'}
            </label>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={filters.maxSpam ?? 1}
                onChange={(e) =>
                  onChange({ ...filters, maxSpam: parseFloat(e.target.value) })
                }
                className="flex-1 accent-primary"
              />
              <button
                type="button"
                onClick={() => onChange({ ...filters, maxSpam: null })}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Min authority */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Min authority: {filters.minAuthority != null ? pct(filters.minAuthority) : 'Any'}
            </label>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={filters.minAuthority ?? 0}
                onChange={(e) =>
                  onChange({ ...filters, minAuthority: parseFloat(e.target.value) })
                }
                className="flex-1 accent-primary"
              />
              <button
                type="button"
                onClick={() => onChange({ ...filters, minAuthority: null })}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Sort */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Sort by
            </label>
            <select
              value={filters.sort}
              onChange={(e) => onChange({ ...filters, sort: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              {SORT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Order toggle */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Order
            </label>
            <button
              type="button"
              onClick={() =>
                onChange({ ...filters, order: filters.order === 'desc' ? 'asc' : 'desc' })
              }
              className="flex items-center gap-1.5 rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground hover:bg-secondary/50"
            >
              <ArrowUpDown className="h-3.5 w-3.5" />
              {filters.order === 'desc' ? 'Descending' : 'Ascending'}
            </button>
          </div>

          {/* Limit */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Results per page
            </label>
            <select
              value={filters.limit}
              onChange={(e) =>
                onChange({ ...filters, limit: parseInt(e.target.value, 10) })
              }
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              {LIMIT_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n} documents
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
