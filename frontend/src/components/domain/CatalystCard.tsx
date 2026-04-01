import { Link } from 'react-router-dom';
import { ArrowUpRight, ExternalLink, FileText, Network, RadioTower, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';
import { LIFECYCLE_COLORS, PLATFORMS, SENTIMENT_COLORS } from '@/lib/constants';
import { timeAgo } from '@/lib/formatters';
import type { MarketCatalystItem } from '@/api/hooks/useThemes';

const BIAS_COLORS: Record<string, string> = {
  bullish: 'bg-emerald-500/20 text-emerald-400',
  bearish: 'bg-red-500/20 text-red-400',
  mixed: 'bg-amber-500/20 text-amber-400',
};

function titleize(value: string): string {
  return value.split('_').join(' ');
}

function signed(value: number): string {
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}`;
}

export function CatalystCard({ catalyst }: { catalyst: MarketCatalystItem }) {
  const biasColor = BIAS_COLORS[catalyst.bias] ?? 'bg-slate-500/20 text-slate-400';
  const lifecycleColor = LIFECYCLE_COLORS[catalyst.lifecycle_stage] ?? 'bg-slate-500/20 text-slate-400';

  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className={cn('rounded-full px-2 py-0.5 font-medium', biasColor)}>
          {catalyst.bias}
        </span>
        <span className={cn('rounded-full px-2 py-0.5', lifecycleColor)}>
          {catalyst.lifecycle_stage}
        </span>
        {catalyst.investment_signal && (
          <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
            {titleize(catalyst.investment_signal)}
          </span>
        )}
        <span className="ml-auto text-muted-foreground">
          Updated {timeAgo(catalyst.last_document_at)}
        </span>
      </div>

      <div className="mt-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-foreground">{catalyst.theme_name}</h3>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{catalyst.summary}</p>
        </div>
        <div className="rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-right">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Impact Score</div>
          <div className="mt-1 text-2xl font-semibold text-primary">
            {Math.round(catalyst.market_impact_score)}
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 xl:grid-cols-5">
        <div className="rounded-md border border-border bg-secondary/20 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Conviction</div>
          <div className="mt-1 text-sm font-medium text-foreground">
            {Math.round(catalyst.conviction_score)}
          </div>
        </div>
        <div className="rounded-md border border-border bg-secondary/20 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Rate</div>
          <div className="mt-1 text-sm font-medium text-foreground">
            {catalyst.current_rate_per_hour.toFixed(1)}/hr
          </div>
        </div>
        <div className="rounded-md border border-border bg-secondary/20 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Accel</div>
          <div className="mt-1 text-sm font-medium text-foreground">
            {signed(catalyst.current_acceleration)}
          </div>
        </div>
        <div className="rounded-md border border-border bg-secondary/20 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Volume Z</div>
          <div className="mt-1 text-sm font-medium text-foreground">
            {catalyst.volume_zscore == null ? '—' : signed(catalyst.volume_zscore)}
          </div>
        </div>
        <div className="rounded-md border border-border bg-secondary/20 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Platforms</div>
          <div className="mt-1 text-sm font-medium text-foreground">{catalyst.platform_count}</div>
        </div>
      </div>

      {catalyst.primary_tickers.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <Zap className="h-3.5 w-3.5" />
            Direct Tickers
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {catalyst.primary_tickers.map((ticker) => (
              <span
                key={ticker.ticker}
                className="rounded-md border border-border bg-background/70 px-2.5 py-1 text-xs font-mono text-foreground"
              >
                ${ticker.ticker} · {ticker.mention_count}
              </span>
            ))}
          </div>
        </div>
      )}

      {catalyst.related_tickers.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <Network className="h-3.5 w-3.5" />
            Follow-On Names
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {catalyst.related_tickers.map((ticker) => (
              <span
                key={`${ticker.source_ticker}-${ticker.ticker}`}
                className="rounded-md border border-border bg-background/70 px-2.5 py-1 text-xs text-foreground"
              >
                <span className="font-mono">${ticker.ticker}</span>
                <span className="text-muted-foreground"> via ${ticker.source_ticker}</span>
                <span className={cn('ml-1 font-medium', ticker.impact >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                  {signed(ticker.impact)}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {catalyst.dominant_event_types.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <RadioTower className="h-3.5 w-3.5" />
            Corroborating Events
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {catalyst.dominant_event_types.map((eventType) => (
              <span
                key={eventType}
                className="rounded-full bg-secondary px-2 py-0.5 text-[11px] text-muted-foreground"
              >
                {titleize(eventType)}
              </span>
            ))}
          </div>
        </div>
      )}

      {catalyst.evidence.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <FileText className="h-3.5 w-3.5" />
            Supporting Headlines
          </div>
          <div className="mt-2 space-y-2">
            {catalyst.evidence.map((doc) => {
              const platform = doc.platform ? PLATFORMS[doc.platform] : null;
              const sentimentColor = doc.sentiment_label
                ? (SENTIMENT_COLORS[doc.sentiment_label] ?? 'bg-slate-500/20 text-slate-400')
                : null;

              return (
                <div
                  key={doc.document_id}
                  className="rounded-md border border-border bg-background/50 px-3 py-2"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm text-foreground">
                        {doc.title ?? 'Untitled supporting document'}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                        {platform && (
                          <span className={cn('rounded-full px-2 py-0.5', platform.color)}>
                            {platform.label}
                          </span>
                        )}
                        {doc.sentiment_label && sentimentColor && (
                          <span className={cn('rounded-full px-2 py-0.5', sentimentColor)}>
                            {doc.sentiment_label}
                          </span>
                        )}
                        {doc.authority_score != null && (
                          <span>Authority {(doc.authority_score * 100).toFixed(0)}</span>
                        )}
                        <span>{timeAgo(doc.timestamp)}</span>
                      </div>
                    </div>
                    {doc.url && (
                      <a
                        href={doc.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-muted-foreground hover:text-foreground"
                      >
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="mt-4 flex items-center justify-between border-t border-border pt-4">
        <div className="text-xs text-muted-foreground">
          Started {timeAgo(catalyst.started_at)}
        </div>
        <Link
          to={`/themes/${catalyst.theme_id}?tab=narratives&run=${catalyst.run_id}`}
          className="inline-flex items-center gap-1 text-sm text-primary hover:text-primary/80"
        >
          Open theme
          <ArrowUpRight className="h-4 w-4" />
        </Link>
      </div>
    </div>
  );
}

export function CatalystCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2">
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-20 animate-pulse rounded-full bg-secondary" />
        <div className="ml-auto h-4 w-28 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3 h-5 w-1/3 animate-pulse rounded bg-secondary" />
      <div className="mt-2 h-4 w-full animate-pulse rounded bg-secondary" />
      <div className="mt-1 h-4 w-5/6 animate-pulse rounded bg-secondary" />
      <div className="mt-4 grid grid-cols-2 gap-3 xl:grid-cols-5">
        {Array.from({ length: 5 }).map((_, idx) => (
          <div key={idx} className="h-16 animate-pulse rounded-md bg-secondary" />
        ))}
      </div>
      <div className="mt-4 h-4 w-32 animate-pulse rounded bg-secondary" />
      <div className="mt-2 flex gap-2">
        <div className="h-7 w-20 animate-pulse rounded-md bg-secondary" />
        <div className="h-7 w-20 animate-pulse rounded-md bg-secondary" />
      </div>
    </div>
  );
}
