import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { useHealth } from '@/api/hooks/useHealth';
import { useDocumentStats } from '@/api/hooks/useDocuments';
import { timeAgo, pct } from '@/lib/formatters';
import { Wifi, WifiOff, Cpu, HardDrive, Database, BarChart3, Clock, Layers, CheckCircle2 } from 'lucide-react';

const QUEUE_LABELS: Record<string, string> = {
  embedding_queue: 'Embedding',
  sentiment_queue: 'Sentiment',
  clustering_queue: 'Clustering',
};

export default function Dashboard() {
  const { data: health, isLoading: healthLoading, isError: healthError, error: healthErr } = useHealth();
  const { data: stats, isLoading: statsLoading } = useDocumentStats({ refetchInterval: 30_000 });

  const topPlatform = stats?.platform_counts?.length
    ? stats.platform_counts.reduce((a, b) => (b.count > a.count ? b : a))
    : null;

  return (
    <>
      <Header title="Dashboard" />
      <div className="p-6">
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-foreground">System Health</h2>
          <p className="text-xs text-muted-foreground">
            Service status and resource availability
          </p>
        </div>

        {healthError && (
          <div className="mb-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to fetch health data
            {healthErr instanceof Error && `: ${healthErr.message}`}
          </div>
        )}

        {/* Health metric cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {healthLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : health ? (
            <>
              <MetricCard
                label="Service Status"
                value={health.status}
                icon={health.status === 'healthy' ? Wifi : WifiOff}
                trend={health.status === 'healthy' ? 'up' : 'down'}
                trendLabel={health.status === 'healthy' ? 'All systems operational' : 'Degraded'}
              />
              <MetricCard
                label="Models Loaded"
                value={Object.values(health.models_loaded).filter(Boolean).length}
                icon={Cpu}
                subtitle={`of ${Object.keys(health.models_loaded).length} available`}
              />
              <MetricCard
                label="Cache"
                value={health.cache_available ? 'Connected' : 'Unavailable'}
                icon={HardDrive}
                trend={health.cache_available ? 'up' : 'down'}
              />
              <MetricCard
                label="GPU"
                value={health.gpu_available ? 'Available' : 'CPU Only'}
                icon={Cpu}
                trend={health.gpu_available ? 'up' : 'neutral'}
              />
            </>
          ) : null}
        </div>

        {/* Data Overview + Queue Depths */}
        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Data Overview */}
          <div className="rounded-lg border border-border bg-card p-6">
            <h3 className="text-sm font-medium text-muted-foreground">Data Overview</h3>
            {statsLoading ? (
              <div className="mt-4 grid grid-cols-2 gap-3">
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
              </div>
            ) : stats ? (
              <div className="mt-4 grid grid-cols-2 gap-3">
                <MetricCard
                  label="Total Documents"
                  value={stats.total_count.toLocaleString()}
                  icon={Database}
                />
                <MetricCard
                  label="Top Platform"
                  value={topPlatform?.platform ?? '—'}
                  subtitle={topPlatform ? `${topPlatform.count.toLocaleString()} docs` : undefined}
                  icon={BarChart3}
                />
                <MetricCard
                  label="Embedding Coverage"
                  value={pct(stats.embedding_coverage.finbert_pct)}
                  subtitle={`MiniLM: ${pct(stats.embedding_coverage.minilm_pct)}`}
                  icon={Cpu}
                />
                <MetricCard
                  label="Latest Ingestion"
                  value={timeAgo(stats.latest_fetched_at ?? stats.latest_document)}
                  icon={Clock}
                />
              </div>
            ) : (
              <p className="mt-2 text-xs text-muted-foreground">
                No data available — check that the API is running.
              </p>
            )}
          </div>

          {/* Queue Depths */}
          <div className="rounded-lg border border-border bg-card p-6">
            <h3 className="text-sm font-medium text-muted-foreground">Queue Status</h3>
            {healthLoading ? (
              <div className="mt-4 space-y-3">
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
              </div>
            ) : health?.queue_depths ? (
              <div className="mt-4 space-y-3">
                {Object.entries(QUEUE_LABELS).map(([key, label]) => {
                  const metrics = health.queue_depths[key];
                  const pending = metrics?.pending ?? 0;
                  const processed = metrics?.processed ?? 0;
                  return (
                    <div
                      key={key}
                      className="flex items-center justify-between rounded-md border border-border bg-secondary/30 px-4 py-3"
                    >
                      <div className="flex items-center gap-2">
                        <Layers className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-foreground">{label}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span
                          className={
                            pending > 100
                              ? 'text-sm font-semibold text-amber-400'
                              : 'text-sm font-semibold text-foreground'
                          }
                        >
                          {pending.toLocaleString()} pending
                        </span>
                        <span className="text-muted-foreground">|</span>
                        <span className="flex items-center gap-1 text-sm text-muted-foreground">
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          {processed.toLocaleString()} done
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="mt-2 text-xs text-muted-foreground">
                No queue data available — check that Redis is running.
              </p>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
