import { SlidersHorizontal, ChevronDown, ChevronUp, Search } from 'lucide-react';

export interface SecuritiesFilterValues {
  search: string;
  activeOnly: boolean;
  exchange: string;
  limit: number;
}

export const DEFAULT_SECURITIES_FILTERS: SecuritiesFilterValues = {
  search: '',
  activeOnly: false,
  exchange: '',
  limit: 25,
};

interface SecuritiesFiltersProps {
  isOpen: boolean;
  onToggle: () => void;
  filters: SecuritiesFilterValues;
  onChange: (filters: SecuritiesFilterValues) => void;
}

const EXCHANGES = ['US', 'KRX', 'TSE', 'HKEX', 'SSE', 'LSE', 'XETRA'];
const LIMIT_OPTIONS = [25, 50, 100];

export function SecuritiesFilters({
  isOpen,
  onToggle,
  filters,
  onChange,
}: SecuritiesFiltersProps) {
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
        <div className="grid grid-cols-1 gap-4 border-t border-border px-4 py-4 sm:grid-cols-2 lg:grid-cols-4">
          {/* Search */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Search
            </label>
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                value={filters.search}
                onChange={(e) => onChange({ ...filters, search: e.target.value })}
                placeholder="Ticker or name..."
                className="w-full rounded border border-border bg-background py-1.5 pl-8 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>
          </div>

          {/* Exchange */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Exchange
            </label>
            <select
              value={filters.exchange}
              onChange={(e) => onChange({ ...filters, exchange: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All Exchanges</option>
              {EXCHANGES.map((ex) => (
                <option key={ex} value={ex}>
                  {ex}
                </option>
              ))}
            </select>
          </div>

          {/* Active only */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Status
            </label>
            <label className="flex items-center gap-2 rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground">
              <input
                type="checkbox"
                checked={filters.activeOnly}
                onChange={(e) => onChange({ ...filters, activeOnly: e.target.checked })}
                className="h-3.5 w-3.5 rounded border-border accent-primary"
              />
              Active only
            </label>
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
                  {n} securities
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
