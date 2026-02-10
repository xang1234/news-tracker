import { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeft,
  FileText,
  GitBranch,
  Activity,
  Users,
  ChevronLeft,
  ChevronRight,
  Merge,
} from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { CooccurrenceCard, CooccurrenceCardSkeleton } from '@/components/domain/CooccurrenceCard';
import {
  EntitySentimentPanel,
  EntitySentimentPanelSkeleton,
} from '@/components/domain/EntitySentimentPanel';
import { MergeEntityModal } from '@/components/domain/MergeEntityModal';
import {
  useEntityDetail,
  useEntityDocuments,
  useEntityCooccurrence,
  useEntitySentiment,
  useMergeEntity,
} from '@/api/hooks/useEntities';
import { timeAgo, latency } from '@/lib/formatters';
import { cn } from '@/lib/utils';

const ENTITY_COLORS: Record<string, string> = {
  TICKER: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
  COMPANY: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  PRODUCT: 'bg-violet-500/20 text-violet-300 border-violet-500/30',
  TECHNOLOGY: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  METRIC: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
};

type Tab = 'documents' | 'cooccurrence' | 'sentiment' | 'graph';

export default function EntityDetail() {
  const { type = '', normalized = '' } = useParams<{ type: string; normalized: string }>();
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>('documents');
  const [docOffset, setDocOffset] = useState(0);
  const [mergeOpen, setMergeOpen] = useState(false);

  const detail = useEntityDetail(type, normalized);
  const docs = useEntityDocuments(type, normalized, { limit: 50, offset: docOffset });
  const cooccurrence = useEntityCooccurrence(type, normalized);
  const sentiment = useEntitySentiment(type, normalized);
  const merge = useMergeEntity();

  const badgeColor = ENTITY_COLORS[type] ?? 'bg-slate-500/20 text-slate-300 border-slate-500/30';

  // Pagination
  const docTotal = docs.data?.total ?? 0;
  const docShowing = docs.data?.documents.length ?? 0;
  const docHasNext = docOffset + docShowing < docTotal;
  const docHasPrev = docOffset > 0;

  // Top platform from detail
  const topPlatform = detail.data
    ? Object.entries(detail.data.platforms).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—'
    : '—';

  function handleMerge(toType: string, toNormalized: string) {
    merge.mutate(
      { fromType: type, fromNormalized: normalized, toType, toNormalized },
      {
        onSuccess: () => {
          setMergeOpen(false);
          navigate('/entities');
        },
      },
    );
  }

  const tabs: { key: Tab; label: string; icon: typeof FileText }[] = [
    { key: 'documents', label: 'Documents', icon: FileText },
    { key: 'cooccurrence', label: 'Co-occurrence', icon: Users },
    { key: 'sentiment', label: 'Sentiment', icon: Activity },
    ...(detail.data?.graph_node_id
      ? [{ key: 'graph' as const, label: 'Graph', icon: GitBranch }]
      : []),
  ];

  return (
    <>
      <Header title={decodeURIComponent(normalized)} />
      <div className="mx-auto max-w-4xl p-6">
        {/* Back link */}
        <Link
          to="/entities"
          className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Entities
        </Link>

        {/* Header row */}
        <div className="flex items-center gap-3">
          <span className={cn('rounded border px-2 py-0.5 text-xs font-medium', badgeColor)}>
            {type}
          </span>
          <h2 className="text-xl font-semibold text-foreground">{decodeURIComponent(normalized)}</h2>
          <button
            onClick={() => setMergeOpen(true)}
            className="ml-auto flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
          >
            <Merge className="h-3.5 w-3.5" />
            Merge
          </button>
        </div>

        {/* Stats row */}
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          {detail.isLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            <>
              <MetricCard label="Total Mentions" value={detail.data?.mention_count ?? 0} />
              <MetricCard label="First Seen" value={timeAgo(detail.data?.first_seen)} />
              <MetricCard label="Last Seen" value={timeAgo(detail.data?.last_seen)} />
              <MetricCard label="Top Platform" value={topPlatform} />
            </>
          )}
        </div>

        {/* Tabs */}
        <div className="mt-6 flex gap-1 rounded-lg border border-border bg-card p-1" role="tablist" aria-label="Entity tabs">
          {tabs.map((t) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={tab === t.key}
              aria-controls={`entity-tabpanel-${t.key}`}
              id={`entity-tab-${t.key}`}
              onClick={() => setTab(t.key)}
              className={cn(
                'flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                tab === t.key
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              <t.icon className="h-3.5 w-3.5" />
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="mt-4" role="tabpanel" id={`entity-tabpanel-${tab}`} aria-labelledby={`entity-tab-${tab}`}>
          {/* Documents tab */}
          {tab === 'documents' && (
            <>
              {docs.isLoading && (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="rounded-lg border border-border bg-card p-4">
                      <div className="h-4 w-3/4 animate-pulse rounded bg-secondary" />
                      <div className="mt-2 h-3 w-full animate-pulse rounded bg-secondary" />
                      <div className="mt-2 h-3 w-1/2 animate-pulse rounded bg-secondary" />
                    </div>
                  ))}
                </div>
              )}
              {docs.isError && (
                <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                  Failed to load documents
                </div>
              )}

              {docs.data && docs.data.documents.length === 0 && (
                <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                  <FileText className="h-10 w-10" />
                  <p className="mt-3 text-sm">No documents found</p>
                </div>
              )}

              {docs.data && docs.data.documents.length > 0 && (
                <>
                  <div className="space-y-3">
                    {docs.data.documents.map((d) => (
                      <Link
                        key={d.document_id}
                        to={`/documents/${d.document_id}`}
                        className="block rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/30"
                      >
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          {d.platform && (
                            <span className="rounded bg-secondary px-1.5 py-0.5">{d.platform}</span>
                          )}
                          {d.author_name && <span>{d.author_name}</span>}
                          {d.timestamp && <span className="ml-auto">{timeAgo(d.timestamp)}</span>}
                        </div>
                        {d.title && (
                          <p className="mt-1.5 text-sm font-medium text-foreground">{d.title}</p>
                        )}
                        {d.content_preview && (
                          <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                            {d.content_preview}
                          </p>
                        )}
                        <div className="mt-2 flex items-center gap-2">
                          {d.tickers.map((t) => (
                            <span
                              key={t}
                              className="rounded bg-sky-500/10 px-1.5 py-0.5 text-xs text-sky-300"
                            >
                              {t}
                            </span>
                          ))}
                          {d.sentiment_label && (
                            <span
                              className={cn(
                                'ml-auto rounded px-1.5 py-0.5 text-xs',
                                d.sentiment_label === 'positive' && 'bg-emerald-500/10 text-emerald-300',
                                d.sentiment_label === 'negative' && 'bg-red-500/10 text-red-300',
                                d.sentiment_label === 'neutral' && 'bg-slate-500/10 text-slate-300',
                              )}
                            >
                              {d.sentiment_label}
                            </span>
                          )}
                        </div>
                      </Link>
                    ))}
                  </div>

                  {/* Pagination */}
                  <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground">
                    <span>
                      Showing {docOffset + 1}–{docOffset + docShowing} of {docTotal}
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        disabled={!docHasPrev}
                        onClick={() => setDocOffset(Math.max(0, docOffset - 50))}
                        className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40"
                      >
                        <ChevronLeft className="h-3.5 w-3.5" />
                        Previous
                      </button>
                      <button
                        type="button"
                        disabled={!docHasNext}
                        onClick={() => setDocOffset(docOffset + 50)}
                        className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40"
                      >
                        Next
                        <ChevronRight className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                </>
              )}
            </>
          )}

          {/* Co-occurrence tab */}
          {tab === 'cooccurrence' && (
            <>
              {cooccurrence.isLoading && (
                <div className="space-y-3">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <CooccurrenceCardSkeleton key={i} />
                  ))}
                </div>
              )}
              {cooccurrence.isError && (
                <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                  Failed to load co-occurrence data
                </div>
              )}

              {cooccurrence.data && cooccurrence.data.entities.length === 0 && (
                <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                  <Users className="h-10 w-10" />
                  <p className="mt-3 text-sm">No co-occurring entities found</p>
                </div>
              )}

              {cooccurrence.data && cooccurrence.data.entities.length > 0 && (
                <div className="space-y-3">
                  {cooccurrence.data.entities.map((e) => (
                    <CooccurrenceCard
                      key={`${e.type}:${e.normalized}`}
                      type={e.type}
                      normalized={e.normalized}
                      cooccurrence_count={e.cooccurrence_count}
                      jaccard={e.jaccard}
                      onClick={() =>
                        navigate(
                          `/entities/${encodeURIComponent(e.type)}/${encodeURIComponent(e.normalized)}`,
                        )
                      }
                    />
                  ))}
                </div>
              )}
            </>
          )}

          {/* Sentiment tab */}
          {tab === 'sentiment' && (
            <>
              {sentiment.isLoading && <EntitySentimentPanelSkeleton />}
              {sentiment.isError && (
                <div className="rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                  Failed to load sentiment data
                </div>
              )}

              {sentiment.data && (
                <EntitySentimentPanel
                  avg_score={sentiment.data.avg_score}
                  pos_count={sentiment.data.pos_count}
                  neg_count={sentiment.data.neg_count}
                  neu_count={sentiment.data.neu_count}
                  trend={sentiment.data.trend}
                />
              )}
            </>
          )}

          {/* Graph tab */}
          {tab === 'graph' && detail.data?.graph_node_id && (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <GitBranch className="h-10 w-10" />
              <p className="mt-3 text-sm">This entity is linked to graph node</p>
              <Link
                to={`/graph?node=${detail.data.graph_node_id}`}
                className="mt-2 rounded bg-primary/10 px-3 py-1.5 text-sm text-primary hover:bg-primary/20"
              >
                View in Graph →
              </Link>
            </div>
          )}
        </div>

        {/* Merge modal */}
        <MergeEntityModal
          isOpen={mergeOpen}
          onClose={() => setMergeOpen(false)}
          fromType={type}
          fromNormalized={normalized}
          onMerge={handleMerge}
        />
      </div>
    </>
  );
}
