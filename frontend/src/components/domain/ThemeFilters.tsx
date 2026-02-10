import { SlidersHorizontal, ChevronDown, ChevronUp } from 'lucide-react';

export interface ThemeFilterValues {
  lifecycleStage: string;
  limit: number;
}

export const DEFAULT_THEME_FILTERS: ThemeFilterValues = {
  lifecycleStage: '',
  limit: 25,
};

interface ThemeFiltersProps {
  isOpen: boolean;
  onToggle: () => void;
  filters: ThemeFilterValues;
  onChange: (filters: ThemeFilterValues) => void;
}

const LIFECYCLE_STAGES = ['emerging', 'accelerating', 'mature', 'fading'];
const LIMIT_OPTIONS = [25, 50, 100];

export function ThemeFilters({
  isOpen,
  onToggle,
  filters,
  onChange,
}: ThemeFiltersProps) {
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
        <div className="grid grid-cols-1 gap-4 border-t border-border px-4 py-4 sm:grid-cols-2">
          {/* Lifecycle stage */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Lifecycle Stage
            </label>
            <select
              value={filters.lifecycleStage}
              onChange={(e) => onChange({ ...filters, lifecycleStage: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All Stages</option>
              {LIFECYCLE_STAGES.map((stage) => (
                <option key={stage} value={stage}>
                  {stage.charAt(0).toUpperCase() + stage.slice(1)}
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
                  {n} themes
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
