import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, FileText, Tag, Activity } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { cn } from '@/lib/utils';
import { LIFECYCLE_COLORS, SENTIMENT_COLORS } from '@/lib/constants';
import { timeAgo, pct, truncate, latency as fmtLatency } from '@/lib/formatters';
import { ThemeMetricsChart } from '@/components/domain/ThemeMetricsChart';
import { ThemeSentimentPanel } from '@/components/domain/ThemeSentimentPanel';
import {
  useThemeDetail,
  useThemeMetrics,
  useThemeSentiment,
  useThemeDocuments,
  useThemeEvents,
} from '@/api/hooks/useThemes';

type Tab = 'metrics' | 'sentiment' | 'documents' | 'events';

export default function ThemeDetail() {
  const { themeId } = useParams();
  const [activeTab, setActiveTab] = useState<Tab>('metrics');
  const [docOffset, setDocOffset] = useState(0);

  const { data: detail, isLoading, isError } = useThemeDetail(themeId);
  const metrics = useThemeMetrics(activeTab === 'metrics' ? themeId : undefined);
  const sentiment = useThemeSentiment(activeTab === 'sentiment' ? themeId : undefined);
  const documents = useThemeDocuments(
    activeTab === 'documents' ? themeId : undefined,
    { limit: 20, offset: docOffset },
  );
  const events = useThemeEvents(activeTab === 'events' ? themeId : undefined);

  const theme = detail?.theme;

  const tabs: { key: Tab; label: string }[] = [
    { key: 'metrics', label: 'Metrics' },
    { key: 'sentiment', label: 'Sentiment' },
    { key: 'documents', label: 'Documents' },
    { key: 'events', label: 'Events' },
  ];

  return (
    <>
      <Header title="Theme Detail" />
      <div className="mx-auto max-w-4xl p-6">
        {/* Back link */}
        <Link
          to="/themes"
          className="mb-4 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3 w-3" />
          Back to Themes
        </Link>

        {/* Loading */}
        {isLoading && (
          <div className="space-y-4">
            <div className="h-6 w-2/3 animate-pulse rounded bg-secondary" />
            <div className="h-4 w-full animate-pulse rounded bg-secondary" />
            <div className="h-4 w-3/4 animate-pulse rounded bg-secondary" />
          </div>
        )}

        {/* Error */}
        {isError && (
          <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load theme details
          </div>
        )}

        {/* Theme header */}
        {theme && (
          <>
            <div className="mb-6">
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-semibold text-foreground">{theme.name}</h2>
                <span className={cn(
                  'rounded-full px-2.5 py-0.5 text-xs',
                  LIFECYCLE_COLORS[theme.lifecycle_stage] ?? 'bg-slate-500/20 text-slate-400',
                )}>
                  {theme.lifecycle_stage}
                </span>
              </div>
              {theme.description && (
                <p className="mt-2 text-sm text-muted-foreground">{theme.description}</p>
              )}

              {/* Meta grid */}
              <div className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                <div className="rounded-lg border border-border bg-card p-3">
                  <div className="text-muted-foreground">Documents</div>
                  <div className="mt-1 text-lg font-semibold text-foreground">{theme.document_count}</div>
                </div>
                <div className="rounded-lg border border-border bg-card p-3">
                  <div className="text-muted-foreground">Created</div>
                  <div className="mt-1 text-sm font-medium text-foreground">{timeAgo(theme.created_at)}</div>
                </div>
                <div className="rounded-lg border border-border bg-card p-3">
                  <div className="text-muted-foreground">Updated</div>
                  <div className="mt-1 text-sm font-medium text-foreground">{timeAgo(theme.updated_at)}</div>
                </div>
                <div className="rounded-lg border border-border bg-card p-3">
                  <div className="text-muted-foreground">Latency</div>
                  <div className="mt-1 text-sm font-medium text-foreground">{fmtLatency(detail.latency_ms)}</div>
                </div>
              </div>

              {/* Keywords */}
              {theme.top_keywords.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {theme.top_keywords.map((kw) => (
                    <span
                      key={kw}
                      className="inline-flex items-center gap-1 rounded-full bg-secondary px-2.5 py-0.5 text-xs text-muted-foreground"
                    >
                      <Tag className="h-2.5 w-2.5" />
                      {kw}
                    </span>
                  ))}
                </div>
              )}

              {/* Tickers */}
              {theme.top_tickers.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {theme.top_tickers.map((t) => (
                    <span
                      key={t}
                      className="rounded bg-secondary px-2 py-0.5 font-mono text-xs text-foreground"
                    >
                      ${t}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Tab navigation */}
            <div className="border-b border-border">
              <div className="flex gap-0">
                {tabs.map((tab) => (
                  <button
                    key={tab.key}
                    type="button"
                    onClick={() => { setActiveTab(tab.key); setDocOffset(0); }}
                    className={cn(
                      'border-b-2 px-4 py-2.5 text-sm font-medium transition-colors',
                      activeTab === tab.key
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground',
                    )}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Tab content */}
            <div className="mt-6">
              {/* Metrics tab */}
              {activeTab === 'metrics' && (
                <>
                  {metrics.isLoading && (
                    <div className="h-[350px] animate-pulse rounded bg-secondary" />
                  )}
                  {metrics.data && (
                    <ThemeMetricsChart metrics={metrics.data.metrics} />
                  )}
                </>
              )}

              {/* Sentiment tab */}
              {activeTab === 'sentiment' && (
                <>
                  {sentiment.isLoading && (
                    <div className="space-y-4">
                      <div className="h-6 animate-pulse rounded bg-secondary" />
                      <div className="h-24 animate-pulse rounded bg-secondary" />
                    </div>
                  )}
                  {sentiment.data && (
                    <ThemeSentimentPanel sentiment={sentiment.data} />
                  )}
                </>
              )}

              {/* Documents tab */}
              {activeTab === 'documents' && (
                <>
                  {documents.isLoading && (
                    <div className="space-y-3">
                      {Array.from({ length: 5 }).map((_, i) => (
                        <div key={i} className="h-20 animate-pulse rounded-lg bg-secondary" />
                      ))}
                    </div>
                  )}
                  {documents.data && documents.data.documents.length === 0 && (
                    <div className="flex flex-col items-center py-12 text-muted-foreground">
                      <FileText className="h-10 w-10" />
                      <p className="mt-2 text-sm">No documents in this theme</p>
                    </div>
                  )}
                  {documents.data && documents.data.documents.length > 0 && (
                    <div className="space-y-2">
                      {documents.data.documents.map((doc) => (
                        <Link
                          key={doc.document_id}
                          to={`/documents/${doc.document_id}`}
                          className="block rounded-lg border border-border bg-card p-3 transition-colors hover:border-border/80"
                        >
                          <div className="flex items-center gap-2 text-xs">
                            {doc.platform && (
                              <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
                                {doc.platform}
                              </span>
                            )}
                            {doc.sentiment_label && (
                              <span className={cn(
                                'rounded-full px-2 py-0.5',
                                SENTIMENT_COLORS[doc.sentiment_label] ?? 'bg-slate-500/20 text-slate-400',
                              )}>
                                {doc.sentiment_label}
                              </span>
                            )}
                            {doc.authority_score != null && (
                              <span className="text-muted-foreground">
                                Auth: {pct(doc.authority_score)}
                              </span>
                            )}
                            <span className="ml-auto text-muted-foreground">{timeAgo(doc.timestamp)}</span>
                          </div>
                          {doc.title && (
                            <div className="mt-1 text-sm font-medium text-foreground">
                              {truncate(doc.title, 120)}
                            </div>
                          )}
                          {doc.content_preview && (
                            <p className="mt-1 text-xs text-muted-foreground">
                              {truncate(doc.content_preview, 200)}
                            </p>
                          )}
                        </Link>
                      ))}
                      {/* Simple pagination */}
                      <div className="flex items-center justify-between pt-3 text-xs text-muted-foreground">
                        <span>{documents.data.total} document{documents.data.total !== 1 && 's'}</span>
                        <div className="flex gap-2">
                          <button
                            type="button"
                            disabled={docOffset === 0}
                            onClick={() => setDocOffset(Math.max(0, docOffset - 20))}
                            className="rounded border border-border px-3 py-1 hover:bg-secondary/50 disabled:opacity-40"
                          >
                            Previous
                          </button>
                          <button
                            type="button"
                            disabled={docOffset + 20 >= documents.data.total}
                            onClick={() => setDocOffset(docOffset + 20)}
                            className="rounded border border-border px-3 py-1 hover:bg-secondary/50 disabled:opacity-40"
                          >
                            Next
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Events tab */}
              {activeTab === 'events' && (
                <>
                  {events.isLoading && (
                    <div className="space-y-3">
                      {Array.from({ length: 3 }).map((_, i) => (
                        <div key={i} className="h-24 animate-pulse rounded-lg bg-secondary" />
                      ))}
                    </div>
                  )}
                  {events.data && events.data.events.length === 0 && (
                    <div className="flex flex-col items-center py-12 text-muted-foreground">
                      <Activity className="h-10 w-10" />
                      <p className="mt-2 text-sm">No events linked to this theme</p>
                    </div>
                  )}
                  {events.data && events.data.events.length > 0 && (
                    <div>
                      {events.data.investment_signal && (
                        <div className="mb-4 rounded-lg border border-primary/30 bg-primary/10 px-4 py-2 text-sm text-primary">
                          Investment Signal: <strong>{events.data.investment_signal.replaceAll('_', ' ')}</strong>
                        </div>
                      )}
                      <div className="space-y-2">
                        {events.data.events.map((evt) => (
                          <div
                            key={evt.event_id}
                            className="rounded-lg border border-border bg-card p-3"
                          >
                            <div className="flex items-center gap-2 text-xs">
                              <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
                                {evt.event_type.replaceAll('_', ' ')}
                              </span>
                              <span className="text-muted-foreground">
                                confidence: {pct(evt.confidence, 0)}
                              </span>
                              {evt.time_ref && (
                                <span className="ml-auto text-muted-foreground">{evt.time_ref}</span>
                              )}
                            </div>
                            <div className="mt-2 text-sm text-foreground">
                              {evt.actor && <span className="font-medium">{evt.actor}</span>}
                              {evt.actor && ' '}
                              <span className="text-primary">{evt.action}</span>
                              {evt.object && ' '}
                              {evt.object && <span className="text-muted-foreground">{evt.object}</span>}
                            </div>
                            {evt.tickers.length > 0 && (
                              <div className="mt-2 flex gap-1">
                                {evt.tickers.map((t) => (
                                  <span
                                    key={t}
                                    className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-foreground"
                                  >
                                    ${t}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
}
