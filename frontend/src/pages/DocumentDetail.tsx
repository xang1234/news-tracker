import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  ExternalLink,
  User,
  BadgeCheck,
  Check,
  X,
  Shield,
  AlertTriangle,
  Bot,
} from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { useDocument } from '@/api/hooks/useDocuments';
import { cn } from '@/lib/utils';
import { PLATFORMS, SENTIMENT_COLORS } from '@/lib/constants';
import { timeAgo, formatDate, pct } from '@/lib/formatters';

function ScoreBar({
  label,
  value,
  thresholds,
}: {
  label: string;
  value: number;
  thresholds: { warn: number; danger: number; invert?: boolean };
}) {
  const pctVal = Math.round(value * 100);
  const isGood = thresholds.invert ? value >= thresholds.warn : value <= thresholds.warn;
  const isBad = thresholds.invert ? value <= thresholds.danger : value >= thresholds.danger;
  const color = isBad ? 'bg-red-500' : isGood ? 'bg-emerald-500' : 'bg-amber-500';

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-foreground">{pct(value)}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-secondary">
        <div className={cn('h-2 rounded-full', color)} style={{ width: `${pctVal}%` }} />
      </div>
    </div>
  );
}

export default function DocumentDetail() {
  const { documentId } = useParams();
  const { data: doc, isLoading, isError, error } = useDocument(documentId);

  const platform = doc?.platform ? PLATFORMS[doc.platform] : null;

  return (
    <>
      <Header title="Document Detail" />
      <div className="mx-auto max-w-4xl p-6">
        {/* Back link */}
        <Link
          to="/documents"
          className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Documents
        </Link>

        {/* Loading */}
        {isLoading && (
          <div className="space-y-4">
            <div className="h-6 w-2/3 animate-pulse rounded bg-secondary" />
            <div className="flex gap-2">
              <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
              <div className="h-5 w-20 animate-pulse rounded-full bg-secondary" />
            </div>
            <div className="h-40 w-full animate-pulse rounded-lg bg-secondary" />
            <div className="h-20 w-full animate-pulse rounded-lg bg-secondary" />
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load document{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Not found (loaded but no data) */}
        {!isLoading && !isError && !doc && (
          <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
            <p className="text-sm">Document not found</p>
          </div>
        )}

        {/* Detail content */}
        {doc && (
          <div className="space-y-6">
            {/* Main content card */}
            <div className="rounded-lg border border-border bg-card p-5">
              {/* Title + external link */}
              <div className="flex items-start gap-2">
                <h2 className="text-lg font-semibold text-foreground">
                  {doc.title ?? 'Untitled Document'}
                </h2>
                {doc.url && (
                  <a
                    href={doc.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-1 shrink-0 text-muted-foreground hover:text-foreground"
                  >
                    <ExternalLink className="h-4 w-4" />
                  </a>
                )}
              </div>

              {/* Badges row */}
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                {platform && (
                  <span className={cn('rounded-full px-2 py-0.5', platform.color)}>
                    {platform.label}
                  </span>
                )}
                {doc.source_name && (
                  <span className="rounded-full bg-sky-500/20 px-2 py-0.5 text-sky-400">
                    {doc.source_name}
                  </span>
                )}
                {doc.content_type && (
                  <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
                    {doc.content_type}
                  </span>
                )}
                <span className="text-muted-foreground">
                  Published: {formatDate(doc.timestamp)}
                </span>
                <span className="text-muted-foreground">
                  Ingested: {timeAgo(doc.fetched_at)}
                </span>
              </div>

              {/* Full content */}
              {doc.content && (
                <p className="mt-4 whitespace-pre-wrap text-sm text-foreground">
                  {doc.content}
                </p>
              )}
            </div>

            {/* Author section */}
            {doc.author_name && (
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Author</h3>
                <div className="flex items-center gap-2 text-sm text-foreground">
                  <User className="h-4 w-4 text-muted-foreground" />
                  <span>{doc.author_name}</span>
                  {doc.author_verified && (
                    <BadgeCheck className="h-4 w-4 text-sky-400" />
                  )}
                  {doc.author_followers != null && (
                    <span className="text-xs text-muted-foreground">
                      {doc.author_followers.toLocaleString()} followers
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Engagement row */}
            {doc.engagement && Object.keys(doc.engagement).length > 0 && (
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Engagement</h3>
                <div className="flex flex-wrap gap-4 text-sm">
                  {Object.entries(doc.engagement)
                    .filter(([, val]) => val != null)
                    .map(([key, val]) => (
                    <div key={key} className="text-center">
                      <div className="text-lg font-semibold text-foreground">
                        {typeof val === 'number' ? val.toLocaleString() : String(val)}
                      </div>
                      <div className="text-xs text-muted-foreground">{key}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Quality signals */}
            <div className="rounded-lg border border-border bg-card p-4">
              <h3 className="mb-3 text-xs font-medium text-muted-foreground">Quality Signals</h3>
              <div className="space-y-3">
                {doc.spam_score != null && (
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <div className="flex-1">
                      <ScoreBar
                        label="Spam Score"
                        value={doc.spam_score}
                        thresholds={{ warn: 0.3, danger: 0.7 }}
                      />
                    </div>
                  </div>
                )}
                {doc.bot_probability != null && (
                  <div className="flex items-center gap-2">
                    <Bot className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <div className="flex-1">
                      <ScoreBar
                        label="Bot Probability"
                        value={doc.bot_probability}
                        thresholds={{ warn: 0.3, danger: 0.7 }}
                      />
                    </div>
                  </div>
                )}
                {doc.authority_score != null && (
                  <div className="flex items-center gap-2">
                    <Shield className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <div className="flex-1">
                      <ScoreBar
                        label="Authority Score"
                        value={doc.authority_score}
                        thresholds={{ warn: 0.3, danger: 0.2, invert: true }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Sentiment */}
            {doc.sentiment_label && (
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Sentiment</h3>
                <div className="flex items-center gap-2 text-sm">
                  <span
                    className={cn(
                      'rounded-full px-2 py-0.5 text-xs',
                      SENTIMENT_COLORS[doc.sentiment_label] ?? 'bg-slate-500/20 text-slate-400',
                    )}
                  >
                    {doc.sentiment_label}
                  </span>
                  {doc.sentiment_confidence != null && (
                    <span className="text-xs text-muted-foreground">
                      Confidence: {pct(doc.sentiment_confidence)}
                    </span>
                  )}
                </div>
                {doc.sentiment && typeof doc.sentiment === 'object' && (
                  <div className="mt-3 space-y-2">
                    {['positive', 'neutral', 'negative'].map((key) => {
                      const val = doc.sentiment?.[key];
                      if (typeof val !== 'number') return null;
                      return (
                        <div key={key}>
                          <div className="mb-1 flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">{key}</span>
                            <span className="font-mono text-foreground">{pct(val)}</span>
                          </div>
                          <div className="h-1.5 w-full rounded-full bg-secondary">
                            <div
                              className={cn(
                                'h-1.5 rounded-full',
                                key === 'positive' && 'bg-emerald-500',
                                key === 'neutral' && 'bg-slate-400',
                                key === 'negative' && 'bg-red-500',
                              )}
                              style={{ width: `${Math.round(val * 100)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Tickers */}
            {doc.tickers.length > 0 && (
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Tickers</h3>
                <div className="flex flex-wrap gap-1.5">
                  {doc.tickers.map((t) => (
                    <Link
                      key={t}
                      to={`/documents?ticker=${t}`}
                      className="rounded bg-secondary px-2 py-0.5 font-mono text-xs text-foreground hover:bg-secondary/80"
                    >
                      ${t}
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Entities */}
            {doc.entities.length > 0 && (
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Entities</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs text-muted-foreground">
                        <th className="pb-2 font-medium">Type</th>
                        <th className="pb-2 font-medium">Name</th>
                      </tr>
                    </thead>
                    <tbody>
                      {doc.entities.map((e) => (
                        <tr key={`${e.type}:${e.name}`} className="border-b border-border/50 last:border-0">
                          <td className="py-1.5 text-xs text-muted-foreground">{e.type}</td>
                          <td className="py-1.5 text-foreground">{e.name}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Keywords */}
            {doc.keywords.length > 0 && (
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Keywords</h3>
                <div className="flex flex-wrap gap-1.5">
                  {doc.keywords.map((kw) => (
                    <span
                      key={kw.word}
                      className="rounded bg-secondary px-2 py-0.5 text-xs text-foreground"
                    >
                      {kw.word}
                      <span className="ml-1 text-[10px] text-muted-foreground">
                        {kw.score.toFixed(2)}
                      </span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Events */}
            {doc.events.length > 0 && (
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Events</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs text-muted-foreground">
                        <th className="pb-2 pr-4 font-medium">Type</th>
                        <th className="pb-2 pr-4 font-medium">Actor</th>
                        <th className="pb-2 pr-4 font-medium">Action</th>
                        <th className="pb-2 pr-4 font-medium">Object</th>
                        <th className="pb-2 font-medium">Time Ref</th>
                      </tr>
                    </thead>
                    <tbody>
                      {doc.events.map((ev) => (
                        <tr key={`${ev.type}:${ev.actor}:${ev.action}`} className="border-b border-border/50 last:border-0">
                          <td className="py-1.5 pr-4 text-xs text-muted-foreground">{ev.type}</td>
                          <td className="py-1.5 pr-4 text-foreground">{ev.actor}</td>
                          <td className="py-1.5 pr-4 text-foreground">{ev.action}</td>
                          <td className="py-1.5 pr-4 text-foreground">{ev.object}</td>
                          <td className="py-1.5 text-xs text-muted-foreground">{ev.time_ref}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Theme IDs */}
            {doc.theme_ids.length > 0 && (
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Themes</h3>
                <div className="flex flex-wrap gap-1.5">
                  {doc.theme_ids.map((id) => (
                    <Link
                      key={id}
                      to={`/themes/${id}`}
                      className="rounded bg-secondary px-2 py-0.5 font-mono text-xs text-foreground hover:bg-secondary/80"
                    >
                      {id}
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Embedding status */}
            <div className="rounded-lg border border-border bg-card p-4">
              <h3 className="mb-2 text-xs font-medium text-muted-foreground">Embeddings</h3>
              <div className="flex gap-4 text-sm">
                <span className="flex items-center gap-1.5 text-foreground">
                  {doc.has_embedding ? (
                    <Check className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <X className="h-4 w-4 text-red-400" />
                  )}
                  FinBERT
                </span>
                <span className="flex items-center gap-1.5 text-foreground">
                  {doc.has_embedding_minilm ? (
                    <Check className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <X className="h-4 w-4 text-red-400" />
                  )}
                  MiniLM
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
