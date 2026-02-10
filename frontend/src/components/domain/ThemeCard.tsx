import { Link } from 'react-router-dom';
import { FileText, Tag } from 'lucide-react';
import { cn } from '@/lib/utils';
import { LIFECYCLE_COLORS } from '@/lib/constants';
import { timeAgo } from '@/lib/formatters';
import type { ThemeItem } from '@/api/hooks/useThemes';

interface ThemeCardProps {
  theme: ThemeItem;
  score?: number;
  tier?: number;
}

const MAX_KEYWORDS = 6;
const MAX_TICKERS = 5;

export function ThemeCard({ theme, score, tier }: ThemeCardProps) {
  const lifecycleColor = LIFECYCLE_COLORS[theme.lifecycle_stage] ?? 'bg-slate-500/20 text-slate-400';
  const keywordOverflow = theme.top_keywords.length - MAX_KEYWORDS;
  const tickerOverflow = theme.top_tickers.length - MAX_TICKERS;

  return (
    <Link
      to={`/themes/${theme.theme_id}`}
      className="block rounded-lg border border-border bg-card p-4 transition-colors hover:border-border/80"
    >
      {/* Top row: lifecycle + score/tier + time */}
      <div className="flex items-center gap-2 text-xs">
        <span className={cn('rounded-full px-2 py-0.5', lifecycleColor)}>
          {theme.lifecycle_stage}
        </span>
        {tier != null && (
          <span className={cn(
            'rounded-full px-2 py-0.5',
            tier === 1 ? 'bg-emerald-500/20 text-emerald-400' :
            tier === 2 ? 'bg-amber-500/20 text-amber-400' :
            'bg-slate-500/20 text-slate-400',
          )}>
            Tier {tier}
          </span>
        )}
        {score != null && (
          <span className="text-muted-foreground">
            Score: {score.toFixed(2)}
          </span>
        )}
        <span className="ml-auto text-muted-foreground">
          {timeAgo(theme.updated_at)}
        </span>
      </div>

      {/* Name + description */}
      <div className="mt-2">
        <span className="font-medium text-foreground">{theme.name}</span>
        {theme.description && (
          <p className="mt-1 text-sm text-muted-foreground">{theme.description}</p>
        )}
      </div>

      {/* Keywords */}
      {theme.top_keywords.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {theme.top_keywords.slice(0, MAX_KEYWORDS).map((kw) => (
            <span
              key={kw}
              className="inline-flex items-center gap-1 rounded-full bg-secondary px-2 py-0.5 text-[10px] text-muted-foreground"
            >
              <Tag className="h-2.5 w-2.5" />
              {kw}
            </span>
          ))}
          {keywordOverflow > 0 && (
            <span className="text-[10px] text-muted-foreground">+{keywordOverflow}</span>
          )}
        </div>
      )}

      {/* Bottom row: tickers + doc count */}
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        {theme.top_tickers.length > 0 && (
          <div className="flex items-center gap-1">
            {theme.top_tickers.slice(0, MAX_TICKERS).map((t) => (
              <span
                key={t}
                className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-foreground"
              >
                ${t}
              </span>
            ))}
            {tickerOverflow > 0 && (
              <span className="text-[10px] text-muted-foreground">+{tickerOverflow}</span>
            )}
          </div>
        )}
        <span className="ml-auto flex items-center gap-1">
          <FileText className="h-3 w-3" />
          {theme.document_count} doc{theme.document_count !== 1 && 's'}
        </span>
      </div>
    </Link>
  );
}

export function ThemeCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="h-5 w-20 animate-pulse rounded-full bg-secondary" />
        <div className="ml-auto h-4 w-20 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3 h-4 w-2/3 animate-pulse rounded bg-secondary" />
      <div className="mt-2 h-3 w-full animate-pulse rounded bg-secondary" />
      <div className="mt-2 flex gap-1">
        <div className="h-4 w-14 animate-pulse rounded-full bg-secondary" />
        <div className="h-4 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="h-4 w-12 animate-pulse rounded-full bg-secondary" />
      </div>
      <div className="mt-3 flex items-center gap-2">
        <div className="h-3 w-12 animate-pulse rounded bg-secondary" />
        <div className="h-3 w-12 animate-pulse rounded bg-secondary" />
        <div className="ml-auto h-3 w-16 animate-pulse rounded bg-secondary" />
      </div>
    </div>
  );
}
