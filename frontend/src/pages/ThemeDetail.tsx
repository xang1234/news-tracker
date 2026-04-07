import { useState } from 'react';
import { useParams, Link, useSearchParams } from 'react-router-dom';
import { ArrowLeft, FileText, Tag, Activity, Radar, Building2, GitBranch } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { cn } from '@/lib/utils';
import { LIFECYCLE_COLORS, SENTIMENT_COLORS } from '@/lib/constants';
import { timeAgo, pct, truncate, latency as fmtLatency } from '@/lib/formatters';
import { ThemeMetricsChart } from '@/components/domain/ThemeMetricsChart';
import { ThemeSentimentPanel } from '@/components/domain/ThemeSentimentPanel';
import { BasketColumns } from '@/components/domain/BasketColumns';
import { PathExplanation } from '@/components/domain/PathExplanation';
import {
  useThemeDetail,
  useThemeMetrics,
  useThemeSentiment,
  useThemeDocuments,
  useThemeEvents,
  useThemeNarratives,
  useThemeNarrativeDetail,
} from '@/api/hooks/useThemes';
import { useDivergences } from '@/api/hooks/useDivergence';
import { useThemeBasket, useBasketPaths } from '@/api/hooks/useThemeIntelligence';

type Tab = 'metrics' | 'sentiment' | 'documents' | 'events' | 'narratives' | 'filing' | 'structural';

export default function ThemeDetail() {
  const { themeId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTab = searchParams.get('tab') === 'narratives' ? 'narratives' : 'metrics';
  const [manualTab, setManualTab] = useState<Tab>(initialTab);
  const [docOffset, setDocOffset] = useState(0);
  const [manualSelectedRunId, setManualSelectedRunId] = useState<string | null>(searchParams.get('run'));

  const activeTab: Tab = searchParams.get('tab') === 'narratives' ? 'narratives' : manualTab;

  const { data: detail, isLoading, isError } = useThemeDetail(themeId);
  const metrics = useThemeMetrics(activeTab === 'metrics' ? themeId : undefined);
  const sentiment = useThemeSentiment(activeTab === 'sentiment' ? themeId : undefined);
  const documents = useThemeDocuments(
    activeTab === 'documents' ? themeId : undefined,
    { limit: 20, offset: docOffset },
  );
  const events = useThemeEvents(activeTab === 'events' ? themeId : undefined);
  const narratives = useThemeNarratives(themeId);
  const themeBasket = useThemeBasket(activeTab === 'structural' ? themeId : undefined);
  const themeDivergences = useDivergences(activeTab === 'filing' ? { theme: themeId, limit: 20 } : undefined);
  const [selectedConceptId, setSelectedConceptId] = useState<string | null>(null);
  const basketPaths = useBasketPaths(
    activeTab === 'structural' && selectedConceptId ? themeId : undefined,
    selectedConceptId ?? undefined,
  );
  const selectedRunId =
    searchParams.get('run')
    ?? manualSelectedRunId
    ?? narratives.data?.runs[0]?.run_id
    ?? null;
  const narrativeDetail = useThemeNarrativeDetail(themeId, selectedRunId ?? undefined);

  const theme = detail?.theme;

  function updateNarrativeSearch(runId: string | null) {
    const next = new URLSearchParams(searchParams);
    next.set('tab', 'narratives');
    if (runId) {
      next.set('run', runId);
    } else {
      next.delete('run');
    }
    setSearchParams(next, { replace: true });
  }

  function handleTabChange(tab: Tab) {
    setManualTab(tab);
    setDocOffset(0);

    const next = new URLSearchParams(searchParams);
    if (tab === 'narratives') {
      next.set('tab', 'narratives');
      if (selectedRunId) {
        next.set('run', selectedRunId);
      }
    } else {
      next.delete('tab');
      next.delete('run');
    }
    setSearchParams(next, { replace: true });
  }

  const tabs: { key: Tab; label: string }[] = [
    { key: 'metrics', label: 'Metrics' },
    { key: 'sentiment', label: 'Sentiment' },
    { key: 'documents', label: 'Documents' },
    { key: 'events', label: 'Events' },
    { key: 'narratives', label: 'Narratives' },
    { key: 'filing', label: 'Filing Evidence' },
    { key: 'structural', label: 'Structural' },
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
              <div className="flex gap-0" role="tablist" aria-label="Theme tabs">
                {tabs.map((tab) => (
                  <button
                    key={tab.key}
                    type="button"
                    role="tab"
                    aria-selected={activeTab === tab.key}
                    aria-controls={`tabpanel-${tab.key}`}
                    id={`tab-${tab.key}`}
                    onClick={() => { handleTabChange(tab.key); }}
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
            <div className="mt-6" role="tabpanel" id={`tabpanel-${activeTab}`} aria-labelledby={`tab-${activeTab}`}>
              {/* Metrics tab */}
              {activeTab === 'metrics' && (
                <>
                  {metrics.isLoading && (
                    <div className="h-[350px] animate-pulse rounded bg-secondary" />
                  )}
                  {metrics.isError && (
                    <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                      Failed to load metrics
                    </div>
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
                  {sentiment.isError && (
                    <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                      Failed to load sentiment
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
                  {documents.isError && (
                    <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                      Failed to load documents
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

              {/* Narrative tab */}
              {activeTab === 'narratives' && (
                <>
                  {narratives.isLoading && (
                    <div className="grid gap-4 lg:grid-cols-[320px,1fr]">
                      <div className="space-y-3">
                        {Array.from({ length: 4 }).map((_, i) => (
                          <div key={i} className="h-24 animate-pulse rounded-lg bg-secondary" />
                        ))}
                      </div>
                      <div className="h-[420px] animate-pulse rounded-lg bg-secondary" />
                    </div>
                  )}
                  {narratives.isError && (
                    <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                      Failed to load narrative runs
                    </div>
                  )}
                  {narratives.data && narratives.data.runs.length === 0 && (
                    <div className="flex flex-col items-center py-12 text-muted-foreground">
                      <Radar className="h-10 w-10" />
                      <p className="mt-2 text-sm">No narrative runs under this theme yet</p>
                    </div>
                  )}
                  {narratives.data && narratives.data.runs.length > 0 && (
                    <div className="grid gap-4 lg:grid-cols-[320px,1fr]">
                      <div className="space-y-3">
                        {narratives.data.runs.map((run) => (
                          <button
                            key={run.run_id}
                            type="button"
                            onClick={() => {
                              setManualSelectedRunId(run.run_id);
                              updateNarrativeSearch(run.run_id);
                            }}
                            className={cn(
                              'w-full rounded-lg border p-4 text-left transition-colors',
                              selectedRunId === run.run_id
                                ? 'border-primary bg-primary/8'
                                : 'border-border bg-card hover:border-border/80',
                            )}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <div className="text-sm font-medium text-foreground">{run.label}</div>
                                <div className="mt-1 text-xs text-muted-foreground">
                                  {run.status} · {timeAgo(run.last_document_at)}
                                </div>
                              </div>
                              <div className="rounded-full bg-primary/15 px-2 py-0.5 text-[11px] text-primary">
                                {Math.round(run.conviction_score)}
                              </div>
                            </div>
                            <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-muted-foreground">
                              <div className="rounded border border-border bg-background/60 px-2 py-2">
                                <div>Rate</div>
                                <div className="mt-1 text-sm font-medium text-foreground">
                                  {run.current_rate_per_hour.toFixed(1)}
                                </div>
                              </div>
                              <div className="rounded border border-border bg-background/60 px-2 py-2">
                                <div>Accel</div>
                                <div className="mt-1 text-sm font-medium text-foreground">
                                  {run.current_acceleration > 0 ? '+' : ''}{run.current_acceleration.toFixed(1)}
                                </div>
                              </div>
                              <div className="rounded border border-border bg-background/60 px-2 py-2">
                                <div>Platforms</div>
                                <div className="mt-1 text-sm font-medium text-foreground">{run.platform_count}</div>
                              </div>
                            </div>
                            <div className="mt-3 flex flex-wrap gap-1.5">
                              {run.top_tickers.slice(0, 3).map((tickerCount) => (
                                <span
                                  key={tickerCount.ticker}
                                  className="rounded bg-secondary px-2 py-0.5 font-mono text-[10px] text-foreground"
                                >
                                  ${tickerCount.ticker} · {tickerCount.count}
                                </span>
                              ))}
                            </div>
                          </button>
                        ))}
                      </div>

                      <div className="rounded-lg border border-border bg-card p-5">
                        {narrativeDetail.isLoading && (
                          <div className="space-y-4">
                            <div className="h-6 w-1/3 animate-pulse rounded bg-secondary" />
                            <div className="h-24 animate-pulse rounded bg-secondary" />
                            <div className="h-48 animate-pulse rounded bg-secondary" />
                          </div>
                        )}
                        {narrativeDetail.isError && (
                          <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                            Failed to load narrative run detail
                          </div>
                        )}
                        {narrativeDetail.data && (
                          <>
                            <div className="flex items-center justify-between gap-3">
                              <div>
                                <h3 className="text-lg font-semibold text-foreground">
                                  {narrativeDetail.data.run.label}
                                </h3>
                                <p className="mt-1 text-sm text-muted-foreground">
                                  Started {timeAgo(narrativeDetail.data.run.started_at)} · last update {timeAgo(narrativeDetail.data.run.last_document_at)}
                                </p>
                              </div>
                              <div className="rounded-full bg-primary/15 px-3 py-1 text-sm text-primary">
                                {Math.round(narrativeDetail.data.run.conviction_score)} conviction
                              </div>
                            </div>

                            <div className="mt-4 grid gap-3 sm:grid-cols-3">
                              <div className="rounded-lg border border-border bg-secondary/20 p-3">
                                <div className="text-xs text-muted-foreground">Current Rate</div>
                                <div className="mt-1 text-lg font-semibold text-foreground">
                                  {narrativeDetail.data.run.current_rate_per_hour.toFixed(1)}/hr
                                </div>
                              </div>
                              <div className="rounded-lg border border-border bg-secondary/20 p-3">
                                <div className="text-xs text-muted-foreground">Acceleration</div>
                                <div className="mt-1 text-lg font-semibold text-foreground">
                                  {narrativeDetail.data.run.current_acceleration > 0 ? '+' : ''}
                                  {narrativeDetail.data.run.current_acceleration.toFixed(1)}
                                </div>
                              </div>
                              <div className="rounded-lg border border-border bg-secondary/20 p-3">
                                <div className="text-xs text-muted-foreground">Platforms</div>
                                <div className="mt-1 text-lg font-semibold text-foreground">
                                  {narrativeDetail.data.run.platform_count}
                                </div>
                              </div>
                            </div>

                            <div className="mt-5">
                              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                                Platform Sequence
                              </div>
                              <div className="mt-2 flex flex-wrap gap-2">
                                {Object.entries(narrativeDetail.data.platform_timeline).map(([platform, firstSeen]) => (
                                  <span
                                    key={platform}
                                    className="rounded-full border border-border px-3 py-1 text-xs text-foreground"
                                  >
                                    {platform} · {timeAgo(firstSeen)}
                                  </span>
                                ))}
                              </div>
                            </div>

                            <div className="mt-5">
                              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                                Ticker Concentration
                              </div>
                              <div className="mt-2 flex flex-wrap gap-2">
                                {Object.entries(narrativeDetail.data.ticker_counts)
                                  .sort((a, b) => b[1] - a[1])
                                  .slice(0, 6)
                                  .map(([ticker, count]) => (
                                    <span
                                      key={ticker}
                                      className="rounded bg-secondary px-2 py-1 font-mono text-xs text-foreground"
                                    >
                                      ${ticker} · {count}
                                    </span>
                                  ))}
                              </div>
                            </div>

                            <div className="mt-5">
                              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                                Recent Alerts
                              </div>
                              {narrativeDetail.data.run.recent_alerts.length === 0 ? (
                                <p className="mt-2 text-sm text-muted-foreground">No active narrative alerts.</p>
                              ) : (
                                <div className="mt-2 space-y-2">
                                  {narrativeDetail.data.run.recent_alerts.map((alert) => (
                                    <div
                                      key={alert.alert_id}
                                      className="rounded-lg border border-border bg-background/50 px-3 py-2"
                                    >
                                      <div className="flex items-center gap-2 text-xs">
                                        <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
                                          {alert.trigger_type.replaceAll('_', ' ')}
                                        </span>
                                        <span className="rounded-full bg-primary/15 px-2 py-0.5 text-primary">
                                          {alert.severity}
                                        </span>
                                        <span className="ml-auto text-muted-foreground">
                                          {timeAgo(alert.created_at)}
                                        </span>
                                      </div>
                                      <div className="mt-1 text-sm text-foreground">{alert.title}</div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>

                            <div className="mt-5">
                              <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                                Top Documents
                              </div>
                              {narrativeDetail.data.documents.length === 0 ? (
                                <p className="mt-2 text-sm text-muted-foreground">No documents assigned yet.</p>
                              ) : (
                                <div className="mt-2 space-y-2">
                                  {narrativeDetail.data.documents.slice(0, 8).map((doc) => (
                                    <Link
                                      key={doc.document_id}
                                      to={`/documents/${doc.document_id}`}
                                      className="block rounded-lg border border-border bg-background/50 p-3 transition-colors hover:border-border/80"
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
                                        <span className="ml-auto text-muted-foreground">
                                          {doc.timestamp ? timeAgo(doc.timestamp) : 'Unknown time'}
                                        </span>
                                      </div>
                                      {doc.title && (
                                        <div className="mt-1 text-sm font-medium text-foreground">
                                          {truncate(doc.title, 110)}
                                        </div>
                                      )}
                                      {doc.content_preview && (
                                        <p className="mt-1 text-xs text-muted-foreground">
                                          {truncate(doc.content_preview, 180)}
                                        </p>
                                      )}
                                    </Link>
                                  ))}
                                </div>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Filing Evidence tab */}
              {activeTab === 'filing' && (
                <>
                  {themeDivergences.isLoading && (
                    <div className="space-y-3">
                      {Array.from({ length: 3 }).map((_, i) => (
                        <div key={i} className="h-24 animate-pulse rounded-lg bg-secondary" />
                      ))}
                    </div>
                  )}
                  {themeDivergences.isError && (
                    <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                      Failed to load filing evidence
                    </div>
                  )}
                  {themeDivergences.data && themeDivergences.data.divergences.length === 0 && (
                    <div className="flex flex-col items-center py-12 text-muted-foreground">
                      <Building2 className="h-10 w-10" />
                      <p className="mt-2 text-sm">No filing evidence for this theme</p>
                      <p className="mt-1 text-xs">Filing evidence appears when the filing lane processes SEC filings related to this theme.</p>
                    </div>
                  )}
                  {themeDivergences.data && themeDivergences.data.divergences.length > 0 && (
                    <div className="space-y-3">
                      {themeDivergences.data.divergences.map((d) => (
                        <Link
                          key={d.id}
                          to="/divergence"
                          className="block rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/30"
                        >
                          <div className="flex items-center gap-2 text-xs">
                            <span className="font-medium text-foreground">{d.issuer_name}</span>
                            <span className={cn(
                              'rounded-full px-2 py-0.5',
                              d.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                              d.severity === 'warning' ? 'bg-amber-500/20 text-amber-400' :
                              'bg-sky-500/20 text-sky-400',
                            )}>
                              {d.severity}
                            </span>
                            <span className="rounded-full bg-secondary px-2 py-0.5 text-muted-foreground">
                              {d.reason.replaceAll('_', ' ')}
                            </span>
                          </div>
                          <p className="mt-1 text-sm text-foreground">{d.title}</p>
                          {d.narrative_score != null && d.filing_adoption_score != null && (
                            <div className="mt-2 flex gap-4 text-xs text-muted-foreground">
                              <span>Narrative: {pct(d.narrative_score, 0)}</span>
                              <span>Filing: {pct(d.filing_adoption_score, 0)}</span>
                            </div>
                          )}
                        </Link>
                      ))}
                    </div>
                  )}
                </>
              )}

              {/* Structural tab */}
              {activeTab === 'structural' && (
                <>
                  {themeBasket.isLoading && (
                    <div className="space-y-3">
                      {Array.from({ length: 3 }).map((_, i) => (
                        <div key={i} className="h-24 animate-pulse rounded-lg bg-secondary" />
                      ))}
                    </div>
                  )}
                  {themeBasket.isError && (
                    <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                      Failed to load structural data
                    </div>
                  )}
                  {themeBasket.data && themeBasket.data.members.length === 0 && (
                    <div className="flex flex-col items-center py-12 text-muted-foreground">
                      <GitBranch className="h-10 w-10" />
                      <p className="mt-2 text-sm">No structural relationships for this theme</p>
                      <p className="mt-1 text-xs">Structural data appears when the graph lane builds causal paths connecting entities to this theme.</p>
                    </div>
                  )}
                  {themeBasket.data && themeBasket.data.members.length > 0 && (
                    <div className="space-y-6">
                      <BasketColumns
                        members={themeBasket.data.members}
                        onMemberClick={(conceptId) => setSelectedConceptId(conceptId === selectedConceptId ? null : conceptId)}
                      />
                      {selectedConceptId && basketPaths.data && (
                        <div className="rounded-lg border border-border bg-card p-4">
                          <h4 className="mb-3 text-sm font-medium text-foreground">Path Explanation</h4>
                          <PathExplanation paths={basketPaths.data.paths} />
                        </div>
                      )}
                      {selectedConceptId && basketPaths.isLoading && (
                        <div className="h-24 animate-pulse rounded-lg bg-secondary" />
                      )}
                      {selectedConceptId && basketPaths.isError && (
                        <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                          Failed to load path explanation
                        </div>
                      )}
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
                  {events.isError && (
                    <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                      Failed to load events
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
