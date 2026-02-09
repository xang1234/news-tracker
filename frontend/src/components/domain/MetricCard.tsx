import type { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  trendLabel?: string;
  className?: string;
}

export function MetricCard({
  label,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendLabel,
  className,
}: MetricCardProps) {
  return (
    <div className={cn('rounded-lg border border-border bg-card p-4', className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">{label}</span>
        {Icon && <Icon className="h-4 w-4 text-muted-foreground" />}
      </div>
      <div className="mt-2 text-2xl font-semibold text-foreground">{value}</div>
      <div className="mt-1 flex items-center gap-1.5">
        {trend && (
          <span
            className={cn(
              'text-xs font-medium',
              trend === 'up' && 'text-emerald-400',
              trend === 'down' && 'text-red-400',
              trend === 'neutral' && 'text-muted-foreground',
            )}
          >
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '–'}
            {trendLabel && ` ${trendLabel}`}
          </span>
        )}
        {subtitle && <span className="text-xs text-muted-foreground">{subtitle}</span>}
      </div>
    </div>
  );
}

/** Loading skeleton variant */
export function MetricCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="h-3 w-20 animate-pulse rounded bg-secondary" />
      <div className="mt-3 h-7 w-16 animate-pulse rounded bg-secondary" />
      <div className="mt-2 h-3 w-24 animate-pulse rounded bg-secondary" />
    </div>
  );
}
