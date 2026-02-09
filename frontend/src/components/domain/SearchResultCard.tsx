import { ExternalLink, User, BadgeCheck, Shield } from 'lucide-react';
import { cn } from '@/lib/utils';
import { PLATFORMS } from '@/lib/constants';
import { similarityColor } from '@/lib/constants';
import { timeAgo, pct, truncate } from '@/lib/formatters';
import type { SearchResultItem } from '@/api/hooks/useSearch';

interface SearchResultCardProps {
  result: SearchResultItem;
}

export function SearchResultCard({ result }: SearchResultCardProps) {
  const platform = result.platform ? PLATFORMS[result.platform] : null;

  return (
    <div className="rounded-lg border border-border bg-card p-4 transition-colors hover:border-border/80">
      {/* Top row: score + platform + time */}
      <div className="flex items-center gap-2 text-xs">
        <span
          className={cn(
            'rounded-full px-2 py-0.5 font-medium',
            similarityColor(result.score),
          )}
        >
          {pct(result.score)}
        </span>
        {platform && (
          <span className={cn('rounded-full px-2 py-0.5', platform.color)}>
            {platform.label}
          </span>
        )}
        <span className="ml-auto text-muted-foreground">
          {timeAgo(result.timestamp)}
        </span>
      </div>

      {/* Title / content */}
      <div className="mt-2">
        {result.title && (
          <div className="flex items-start gap-1.5">
            {result.url ? (
              <a
                href={result.url}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-foreground hover:underline"
              >
                {result.title}
              </a>
            ) : (
              <span className="font-medium text-foreground">{result.title}</span>
            )}
            {result.url && (
              <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            )}
          </div>
        )}
        {result.content_preview && (
          <p className="mt-1 text-sm text-muted-foreground">
            {truncate(result.content_preview, 200)}
          </p>
        )}
      </div>

      {/* Bottom row: author + tickers + authority */}
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        {result.author_name && (
          <span className="flex items-center gap-1">
            <User className="h-3 w-3" />
            {result.author_name}
            {result.author_verified && (
              <BadgeCheck className="h-3 w-3 text-sky-400" />
            )}
          </span>
        )}
        {result.tickers.length > 0 && (
          <div className="flex items-center gap-1">
            {result.tickers.map((t) => (
              <span
                key={t}
                className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-foreground"
              >
                ${t}
              </span>
            ))}
          </div>
        )}
        {result.authority_score != null && (
          <span className="ml-auto flex items-center gap-1">
            <Shield className="h-3 w-3" />
            {pct(result.authority_score)}
          </span>
        )}
      </div>
    </div>
  );
}

export function SearchResultSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="h-5 w-14 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="ml-auto h-4 w-20 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3 h-4 w-3/4 animate-pulse rounded bg-secondary" />
      <div className="mt-2 h-3 w-full animate-pulse rounded bg-secondary" />
      <div className="mt-1 h-3 w-2/3 animate-pulse rounded bg-secondary" />
      <div className="mt-3 flex items-center gap-2">
        <div className="h-3 w-24 animate-pulse rounded bg-secondary" />
        <div className="ml-auto h-3 w-12 animate-pulse rounded bg-secondary" />
      </div>
    </div>
  );
}
