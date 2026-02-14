import { SlidersHorizontal, ChevronDown, ChevronUp, Search } from 'lucide-react';
import { SOURCE_PLATFORMS } from '@/lib/constants';

export interface SourcesFilterValues {
  search: string;
  platform: string;
  activeOnly: boolean;
  limit: number;
}

export const DEFAULT_SOURCES_FILTERS: SourcesFilterValues = {
  search: '',
  platform: '',
  activeOnly: true,
  limit: 50,
};

interface SourcesFiltersProps {
  isOpen: boolean;
  onToggle: () => void;
  filters: SourcesFilterValues;
  onChange: (filters: SourcesFilterValues) => void;
}

const PLATFORM_OPTIONS = Object.entries(SOURCE_PLATFORMS).map(([value, info]) => ({
  value,
  label: info.label,
}));

const LIMIT_OPTIONS = [25, 50, 100];

export function SourcesFilters({
  isOpen,
  onToggle,
  filters,
  onChange,
}: SourcesFiltersProps) {
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
                placeholder="Identifier or name..."
                className="w-full rounded border border-border bg-background py-1.5 pl-8 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>
          </div>

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
              {PLATFORM_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
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
                  {n} sources
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
