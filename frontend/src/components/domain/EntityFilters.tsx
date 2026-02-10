import { SlidersHorizontal, Search, ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface EntityFilterValues {
  entity_type: string;
  search: string;
  sort: string;
  limit: number;
}

export const DEFAULT_ENTITY_FILTERS: EntityFilterValues = {
  entity_type: '',
  search: '',
  sort: 'count',
  limit: 25,
};

interface EntityFiltersProps {
  isOpen: boolean;
  onToggle: () => void;
  filters: EntityFilterValues;
  onChange: (filters: EntityFilterValues) => void;
}

const ENTITY_TYPES = ['COMPANY', 'PRODUCT', 'TECHNOLOGY', 'TICKER', 'METRIC'];
const SORT_OPTIONS = [
  { value: 'count', label: 'Mention Count' },
  { value: 'recent', label: 'Most Recent' },
];
const LIMIT_OPTIONS = [25, 50, 100];

function activeFilterCount(filters: EntityFilterValues): number {
  let count = 0;
  if (filters.entity_type) count++;
  if (filters.search) count++;
  if (filters.sort !== DEFAULT_ENTITY_FILTERS.sort) count++;
  if (filters.limit !== DEFAULT_ENTITY_FILTERS.limit) count++;
  return count;
}

export function EntityFilters({
  isOpen,
  onToggle,
  filters,
  onChange,
}: EntityFiltersProps) {
  const Chevron = isOpen ? ChevronUp : ChevronDown;
  const filterCount = activeFilterCount(filters);

  return (
    <div className="rounded-lg border border-border bg-card">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-2 px-4 py-3 text-sm font-medium text-foreground hover:bg-secondary/50"
      >
        <SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
        Filters
        {filterCount > 0 && (
          <span className="rounded-full bg-primary/20 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
            {filterCount}
          </span>
        )}
        <Chevron className="ml-auto h-4 w-4 text-muted-foreground" />
      </button>

      {isOpen && (
        <div className="grid grid-cols-1 gap-4 border-t border-border px-4 py-4 sm:grid-cols-2 lg:grid-cols-4">
          {/* Entity type */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Entity Type
            </label>
            <select
              value={filters.entity_type}
              onChange={(e) => onChange({ ...filters, entity_type: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All Types</option>
              {ENTITY_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t.charAt(0) + t.slice(1).toLowerCase()}
                </option>
              ))}
            </select>
          </div>

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
                placeholder="Entity name..."
                className={cn(
                  'w-full rounded border border-border bg-background py-1.5 pl-8 pr-3 text-sm text-foreground',
                  'placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring',
                )}
              />
            </div>
          </div>

          {/* Sort */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Sort By
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
                  {n} entities
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
