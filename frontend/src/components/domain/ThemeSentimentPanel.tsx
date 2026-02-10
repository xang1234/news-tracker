import { TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { pct } from '@/lib/formatters';
import type { ThemeSentimentResponse } from '@/api/hooks/useThemes';

interface ThemeSentimentPanelProps {
  sentiment: ThemeSentimentResponse;
}

export function ThemeSentimentPanel({ sentiment }: ThemeSentimentPanelProps) {
  const bullishPct = Math.round(sentiment.bullish_ratio * 100);
  const bearishPct = Math.round(sentiment.bearish_ratio * 100);
  const neutralPct = Math.round(sentiment.neutral_ratio * 100);

  return (
    <div className="space-y-4">
      {/* Stacked horizontal bar */}
      <div>
        <div className="mb-2 text-xs font-medium text-muted-foreground">
          Sentiment Distribution ({sentiment.document_count} documents)
        </div>
        <div className="flex h-6 w-full overflow-hidden rounded-full">
          {bullishPct > 0 && (
            <div
              className="bg-emerald-500 transition-all"
              style={{ width: `${bullishPct}%` }}
              title={`Bullish: ${bullishPct}%`}
            />
          )}
          {neutralPct > 0 && (
            <div
              className="bg-slate-500 transition-all"
              style={{ width: `${neutralPct}%` }}
              title={`Neutral: ${neutralPct}%`}
            />
          )}
          {bearishPct > 0 && (
            <div
              className="bg-red-500 transition-all"
              style={{ width: `${bearishPct}%` }}
              title={`Bearish: ${bearishPct}%`}
            />
          )}
        </div>
        <div className="mt-2 flex justify-between text-xs">
          <span className="text-emerald-400">Bullish {bullishPct}%</span>
          <span className="text-slate-400">Neutral {neutralPct}%</span>
          <span className="text-red-400">Bearish {bearishPct}%</span>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-border bg-background p-3">
          <div className="text-xs text-muted-foreground">Avg Confidence</div>
          <div className="mt-1 text-lg font-semibold text-foreground">
            {pct(sentiment.avg_confidence)}
          </div>
        </div>
        <div className="rounded-lg border border-border bg-background p-3">
          <div className="text-xs text-muted-foreground">Avg Authority</div>
          <div className="mt-1 text-lg font-semibold text-foreground">
            {sentiment.avg_authority != null ? pct(sentiment.avg_authority) : '—'}
          </div>
        </div>
      </div>

      {/* Velocity indicator */}
      {sentiment.sentiment_velocity != null && (
        <div className="flex items-center gap-2 rounded-lg border border-border bg-background p-3">
          {sentiment.sentiment_velocity > 0 ? (
            <TrendingUp className="h-4 w-4 text-emerald-400" />
          ) : (
            <TrendingDown className="h-4 w-4 text-red-400" />
          )}
          <div>
            <div className="text-xs text-muted-foreground">Sentiment Velocity</div>
            <div className={cn(
              'text-sm font-medium',
              sentiment.sentiment_velocity > 0 ? 'text-emerald-400' : 'text-red-400',
            )}>
              {sentiment.sentiment_velocity > 0 ? '+' : ''}
              {sentiment.sentiment_velocity.toFixed(3)}
            </div>
          </div>
        </div>
      )}

      {/* Extreme sentiment flag */}
      {sentiment.extreme_sentiment && (
        <div className={cn(
          'flex items-center gap-2 rounded-lg border p-3',
          sentiment.extreme_sentiment === 'extreme_bullish'
            ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
            : 'border-red-500/30 bg-red-500/10 text-red-400',
        )}>
          <AlertTriangle className="h-4 w-4" />
          <span className="text-sm font-medium">
            {sentiment.extreme_sentiment === 'extreme_bullish'
              ? 'Extreme Bullish Sentiment'
              : 'Extreme Bearish Sentiment'}
          </span>
        </div>
      )}
    </div>
  );
}
