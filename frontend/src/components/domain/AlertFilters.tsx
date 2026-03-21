import { SlidersHorizontal, ChevronDown, ChevronUp } from 'lucide-react';
import { TRIGGER_TYPE_LABELS } from '@/lib/constants';
import {
  LIMIT_OPTIONS,
  SEVERITIES,
  SUBJECT_TYPES,
  TRIGGER_KEYS,
  type AlertFilterValues,
} from '@/components/domain/alertFilterState';

interface AlertFiltersProps {
  isOpen: boolean;
  onToggle: () => void;
  filters: AlertFilterValues;
  onChange: (filters: AlertFilterValues) => void;
}

export function AlertFilters({
  isOpen,
  onToggle,
  filters,
  onChange,
}: AlertFiltersProps) {
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
        <div className="grid grid-cols-1 gap-4 border-t border-border px-4 py-4 sm:grid-cols-2 lg:grid-cols-5">
          {/* Severity */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Severity
            </label>
            <select
              value={filters.severity}
              onChange={(e) => onChange({ ...filters, severity: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All Severities</option>
              {SEVERITIES.map((s) => (
                <option key={s} value={s}>
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Trigger type */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Trigger Type
            </label>
            <select
              value={filters.triggerType}
              onChange={(e) => onChange({ ...filters, triggerType: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All Types</option>
              {TRIGGER_KEYS.map((key) => (
                <option key={key} value={key}>
                  {TRIGGER_TYPE_LABELS[key]}
                </option>
              ))}
            </select>
          </div>

          {/* Acknowledged */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Subject
            </label>
            <select
              value={filters.subjectType}
              onChange={(e) => onChange({ ...filters, subjectType: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All Subjects</option>
              {SUBJECT_TYPES.map((subjectType) => (
                <option key={subjectType.value} value={subjectType.value}>
                  {subjectType.label}
                </option>
              ))}
            </select>
          </div>

          {/* Acknowledged */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Status
            </label>
            <select
              value={filters.acknowledged}
              onChange={(e) => onChange({ ...filters, acknowledged: e.target.value })}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              <option value="">All</option>
              <option value="false">Unacknowledged</option>
              <option value="true">Acknowledged</option>
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
                  {n} alerts
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
