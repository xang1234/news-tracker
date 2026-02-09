import { SlidersHorizontal, ChevronDown, ChevronUp } from 'lucide-react';
import { PLATFORMS } from '@/lib/constants';
import { pct } from '@/lib/formatters';

export interface SearchFilterValues {
  platforms: string[];
  tickers: string;
  threshold: number;
  minAuthority: number | null;
  limit: number;
}

interface SearchFiltersProps {
  isOpen: boolean;
  onToggle: () => void;
  filters: SearchFilterValues;
  onChange: (filters: SearchFilterValues) => void;
}

const PLATFORM_KEYS = Object.keys(PLATFORMS);
const LIMIT_OPTIONS = [5, 10, 25, 50, 100];

export function SearchFilters({
  isOpen,
  onToggle,
  filters,
  onChange,
}: SearchFiltersProps) {
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
          {/* Platforms */}
          <fieldset>
            <legend className="mb-2 text-xs font-medium text-muted-foreground">
              Platforms
            </legend>
            <div className="flex flex-wrap gap-2">
              {PLATFORM_KEYS.map((key) => (
                <label key={key} className="flex items-center gap-1.5 text-xs text-foreground">
                  <input
                    type="checkbox"
                    checked={filters.platforms.includes(key)}
                    onChange={(e) => {
                      const next = e.target.checked
                        ? [...filters.platforms, key]
                        : filters.platforms.filter((p) => p !== key);
                      onChange({ ...filters, platforms: next });
                    }}
                    className="rounded border-border"
                  />
                  {PLATFORMS[key].label}
                </label>
              ))}
            </div>
          </fieldset>

          {/* Tickers */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Tickers (comma-separated)
            </label>
            <input
              type="text"
              value={filters.tickers}
              onChange={(e) => onChange({ ...filters, tickers: e.target.value })}
              placeholder="NVDA, TSM, INTC"
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>

          {/* Threshold */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Min similarity: {pct(filters.threshold)}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={filters.threshold}
              onChange={(e) =>
                onChange({ ...filters, threshold: parseFloat(e.target.value) })
              }
              className="w-full accent-primary"
            />
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

          {/* Limit */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Results limit
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
                  {n} results
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
