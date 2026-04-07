import { Link } from 'react-router-dom';
import { Bell, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { SEVERITY_COLORS, TRIGGER_TYPE_LABELS } from '@/lib/constants';
import { timeAgo, truncate } from '@/lib/formatters';
import type { AlertItem } from '@/api/hooks/useAlerts';

interface AlertCardProps {
  alert: AlertItem;
  onAcknowledge?: (alertId: string) => void;
  isAcknowledging?: boolean;
}

export function AlertCard({ alert, onAcknowledge, isAcknowledging }: AlertCardProps) {
  const severityColor = SEVERITY_COLORS[alert.severity] ?? 'bg-slate-500/20 text-slate-400';
  const triggerLabel = TRIGGER_TYPE_LABELS[alert.trigger_type] ?? alert.trigger_type;
  const subjectLabel = {
    theme: 'Theme',
    narrative_run: 'Narrative Run',
    graph_node: 'Graph Node',
  }[alert.subject_type] ?? alert.subject_type;
  const isDivergence = alert.trigger_type === 'divergence';
  const destination = isDivergence
    ? '/divergence'
    : alert.subject_type === 'narrative_run'
      ? `/themes/${alert.theme_id}?tab=narratives&run=${alert.subject_id}`
      : `/themes/${alert.theme_id}`;
  const destinationLabel = isDivergence
    ? 'View Divergence'
    : alert.subject_type === 'narrative_run'
      ? 'View Run'
      : 'View Theme';

  return (
    <div
      className={cn(
        'rounded-lg border border-border bg-card p-4 transition-colors',
        alert.acknowledged && 'opacity-60',
      )}
    >
      {/* Top row: severity + trigger type + time */}
      <div className="flex items-center gap-2 text-xs">
        <span className={cn('rounded-full px-2 py-0.5 font-medium', severityColor)}>
          {alert.severity}
        </span>
        <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
          {triggerLabel}
        </span>
        <span className="rounded-full border border-border px-2 py-0.5 text-muted-foreground">
          {subjectLabel}
        </span>
        {alert.conviction_score != null && (
          <span className="rounded-full bg-primary/15 px-2 py-0.5 text-primary">
            {Math.round(alert.conviction_score)} conviction
          </span>
        )}
        {alert.acknowledged && (
          <span className="flex items-center gap-1 text-emerald-400">
            <CheckCircle className="h-3 w-3" />
            Acknowledged
          </span>
        )}
        <span className="ml-auto text-muted-foreground">
          {timeAgo(alert.created_at)}
        </span>
      </div>

      {/* Title + message */}
      <div className="mt-2">
        <span className="font-medium text-foreground">{alert.title}</span>
        <p className="mt-1 text-sm text-muted-foreground">
          {truncate(alert.message, 200)}
        </p>
      </div>

      {/* Bottom row: theme link + acknowledge button */}
      <div className="mt-3 flex items-center gap-2 text-xs">
        <Link
          to={destination}
          className="flex items-center gap-1 text-primary hover:underline"
          onClick={(e) => e.stopPropagation()}
        >
          {destinationLabel}
        </Link>
        {!alert.acknowledged && onAcknowledge && (
          <button
            type="button"
            disabled={isAcknowledging}
            onClick={() => onAcknowledge(alert.alert_id)}
            className="ml-auto flex items-center gap-1 rounded border border-border px-2.5 py-1 text-xs text-muted-foreground hover:bg-secondary/50 hover:text-foreground disabled:opacity-40"
          >
            <Bell className="h-3 w-3" />
            Acknowledge
          </button>
        )}
      </div>
    </div>
  );
}

export function AlertCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-24 animate-pulse rounded-full bg-secondary" />
        <div className="ml-auto h-4 w-20 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3 h-4 w-3/4 animate-pulse rounded bg-secondary" />
      <div className="mt-2 h-3 w-full animate-pulse rounded bg-secondary" />
      <div className="mt-3 flex items-center">
        <div className="h-3 w-20 animate-pulse rounded bg-secondary" />
        <div className="ml-auto h-6 w-24 animate-pulse rounded bg-secondary" />
      </div>
    </div>
  );
}
