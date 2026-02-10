import { Link } from 'react-router-dom';
import { User, BadgeCheck, Shield, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { PLATFORMS, SENTIMENT_COLORS } from '@/lib/constants';
import { timeAgo, pct, truncate } from '@/lib/formatters';
import type { DocumentListItem } from '@/api/hooks/useDocuments';

interface DocumentCardProps {
  document: DocumentListItem;
}

const MAX_TICKERS = 5;

export function DocumentCard({ document: doc }: DocumentCardProps) {
  const platform = doc.platform ? PLATFORMS[doc.platform] : null;
  const overflowCount = doc.tickers.length - MAX_TICKERS;

  return (
    <Link
      to={`/documents/${doc.document_id}`}
      className="block rounded-lg border border-border bg-card p-4 transition-colors hover:border-border/80"
    >
      {/* Top row: platform + content_type + sentiment + time */}
      <div className="flex items-center gap-2 text-xs">
        {platform && (
          <span className={cn('rounded-full px-2 py-0.5', platform.color)}>
            {platform.label}
          </span>
        )}
        {doc.content_type && (
          <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
            {doc.content_type}
          </span>
        )}
        {doc.sentiment_label && (
          <span
            className={cn(
              'rounded-full px-2 py-0.5',
              SENTIMENT_COLORS[doc.sentiment_label] ?? 'bg-slate-500/20 text-slate-400',
            )}
          >
            {doc.sentiment_label}
          </span>
        )}
        <span className="ml-auto text-muted-foreground">
          {timeAgo(doc.timestamp)}
        </span>
      </div>

      {/* Title / content preview */}
      <div className="mt-2">
        {doc.title && (
          <span className="font-medium text-foreground">
            {truncate(doc.title, 120)}
          </span>
        )}
        {doc.content_preview && (
          <p className="mt-1 text-sm text-muted-foreground">
            {truncate(doc.content_preview, 200)}
          </p>
        )}
      </div>

      {/* Bottom row: author + tickers + spam warning + authority */}
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        {doc.author_name && (
          <span className="flex items-center gap-1">
            <User className="h-3 w-3" />
            {doc.author_name}
            {doc.author_verified && (
              <BadgeCheck className="h-3 w-3 text-sky-400" />
            )}
          </span>
        )}
        {doc.tickers.length > 0 && (
          <div className="flex items-center gap-1">
            {doc.tickers.slice(0, MAX_TICKERS).map((t) => (
              <span
                key={t}
                className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-foreground"
              >
                ${t}
              </span>
            ))}
            {overflowCount > 0 && (
              <span className="text-[10px] text-muted-foreground">
                +{overflowCount}
              </span>
            )}
          </div>
        )}
        {doc.spam_score != null && doc.spam_score > 0.5 && (
          <span className="flex items-center gap-1 text-amber-400">
            <AlertTriangle className="h-3 w-3" />
            Spam
          </span>
        )}
        {doc.authority_score != null && (
          <span className="ml-auto flex items-center gap-1">
            <Shield className="h-3 w-3" />
            {pct(doc.authority_score)}
          </span>
        )}
      </div>
    </Link>
  );
}

export function DocumentCardSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-14 animate-pulse rounded-full bg-secondary" />
        <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
        <div className="ml-auto h-4 w-20 animate-pulse rounded bg-secondary" />
      </div>
      <div className="mt-3 h-4 w-3/4 animate-pulse rounded bg-secondary" />
      <div className="mt-2 h-3 w-full animate-pulse rounded bg-secondary" />
      <div className="mt-1 h-3 w-2/3 animate-pulse rounded bg-secondary" />
      <div className="mt-3 flex items-center gap-2">
        <div className="h-3 w-24 animate-pulse rounded bg-secondary" />
        <div className="h-3 w-12 animate-pulse rounded bg-secondary" />
        <div className="ml-auto h-3 w-12 animate-pulse rounded bg-secondary" />
      </div>
    </div>
  );
}
