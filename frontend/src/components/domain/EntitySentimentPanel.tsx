import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

interface EntitySentimentPanelProps {
  avg_score: number | null;
  pos_count: number;
  neg_count: number;
  neu_count: number;
  trend: 'improving' | 'declining' | 'stable';
}

function trendConfig(trend: EntitySentimentPanelProps['trend']) {
  switch (trend) {
    case 'improving':
      return { icon: TrendingUp, label: 'Improving', color: 'text-emerald-400 bg-emerald-500/20' };
    case 'declining':
      return { icon: TrendingDown, label: 'Declining', color: 'text-red-400 bg-red-500/20' };
    case 'stable':
      return { icon: Minus, label: 'Stable', color: 'text-slate-400 bg-slate-500/20' };
  }
}

export function EntitySentimentPanel({
  avg_score,
  pos_count,
  neg_count,
  neu_count,
  trend,
}: EntitySentimentPanelProps) {
  const total = pos_count + neg_count + neu_count;
  const posPct = total > 0 ? Math.round((pos_count / total) * 100) : 0;
  const neuPct = total > 0 ? Math.round((neu_count / total) * 100) : 0;
  const negPct = total > 0 ? Math.round((neg_count / total) * 100) : 0;

  const { icon: TrendIcon, label: trendLabel, color: trendColor } = trendConfig(trend);

  return (
    <div className="space-y-4">
      {/* Average score + trend */}
      <div className="flex items-center gap-3">
        <div>
          <div className="text-xs text-muted-foreground">Average Score</div>
          <div className="mt-0.5 text-lg font-semibold text-foreground">
            {avg_score != null ? avg_score.toFixed(3) : '--'}
          </div>
        </div>
        <span className={cn('ml-auto flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium', trendColor)}>
          <TrendIcon className="h-3 w-3" />
          {trendLabel}
        </span>
      </div>

      {/* Horizontal sentiment bar */}
      <div>
        <div className="mb-2 text-xs font-medium text-muted-foreground">
          Distribution ({total} document{total !== 1 ? 's' : ''})
        </div>
        <div className="flex h-5 w-full overflow-hidden rounded-full">
          {posPct > 0 && (
            <div
              className="bg-emerald-500 transition-all"
              style={{ width: `${posPct}%` }}
              title={`Positive: ${posPct}%`}
            />
          )}
          {neuPct > 0 && (
            <div
              className="bg-slate-500 transition-all"
              style={{ width: `${neuPct}%` }}
              title={`Neutral: ${neuPct}%`}
            />
          )}
          {negPct > 0 && (
            <div
              className="bg-red-500 transition-all"
              style={{ width: `${negPct}%` }}
              title={`Negative: ${negPct}%`}
            />
          )}
        </div>
        <div className="mt-2 flex justify-between text-xs">
          <span className="text-emerald-400">Positive {posPct}%</span>
          <span className="text-slate-400">Neutral {neuPct}%</span>
          <span className="text-red-400">Negative {negPct}%</span>
        </div>
      </div>

      {/* Count grid */}
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-border bg-background p-3 text-center">
          <div className="text-xs text-muted-foreground">Positive</div>
          <div className="mt-1 text-lg font-semibold text-emerald-400">{pos_count}</div>
        </div>
        <div className="rounded-lg border border-border bg-background p-3 text-center">
          <div className="text-xs text-muted-foreground">Neutral</div>
          <div className="mt-1 text-lg font-semibold text-slate-400">{neu_count}</div>
        </div>
        <div className="rounded-lg border border-border bg-background p-3 text-center">
          <div className="text-xs text-muted-foreground">Negative</div>
          <div className="mt-1 text-lg font-semibold text-red-400">{neg_count}</div>
        </div>
      </div>
    </div>
  );
}

export function EntitySentimentPanelSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div>
          <div className="h-3 w-20 animate-pulse rounded bg-secondary" />
          <div className="mt-2 h-6 w-16 animate-pulse rounded bg-secondary" />
        </div>
        <div className="ml-auto h-6 w-24 animate-pulse rounded-full bg-secondary" />
      </div>
      <div>
        <div className="mb-2 h-3 w-32 animate-pulse rounded bg-secondary" />
        <div className="h-5 w-full animate-pulse rounded-full bg-secondary" />
        <div className="mt-2 flex justify-between">
          <div className="h-3 w-16 animate-pulse rounded bg-secondary" />
          <div className="h-3 w-16 animate-pulse rounded bg-secondary" />
          <div className="h-3 w-16 animate-pulse rounded bg-secondary" />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="rounded-lg border border-border bg-background p-3 text-center">
            <div className="mx-auto h-3 w-12 animate-pulse rounded bg-secondary" />
            <div className="mx-auto mt-2 h-6 w-8 animate-pulse rounded bg-secondary" />
          </div>
        ))}
      </div>
    </div>
  );
}
