import { Activity, ArrowDownRight, ArrowUpRight, Target } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { CatalystCard, CatalystCardSkeleton } from '@/components/domain/CatalystCard';
import { useThemeCatalysts } from '@/api/hooks/useThemes';
import { latency } from '@/lib/formatters';

export default function Catalysts() {
  const limit = 12;
  const days = 7;
  const { data, isLoading, isError, error } = useThemeCatalysts(limit, days);
  const showMetricSkeletons = isLoading || !data || isError;

  const bullishCount = data?.catalysts.filter((item) => item.bias === 'bullish').length ?? 0;
  const bearishCount = data?.catalysts.filter((item) => item.bias === 'bearish').length ?? 0;
  const avgImpact = data?.catalysts.length
    ? Math.round(
        data.catalysts.reduce((sum, item) => sum + item.market_impact_score, 0) / data.catalysts.length,
      )
    : 0;

  return (
    <>
      <Header title="Catalyst Radar" />
      <div className="mx-auto max-w-6xl p-6">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Stock-Market Situation Room</h2>
            <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
              Live narratives translated into tradable setups. Rankings favor corroborated situations,
              not just raw headline volume.
            </p>
          </div>
          <div className="hidden rounded-full border border-primary/20 bg-primary/5 px-3 py-1.5 text-xs text-primary lg:block">
            {days}-day event corroboration window
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
          {showMetricSkeletons ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            <>
              <MetricCard label="Live Catalysts" value={data?.total ?? 0} icon={Target} />
              <MetricCard label="Bullish" value={bullishCount} icon={ArrowUpRight} trend={bullishCount > 0 ? 'up' : 'neutral'} />
              <MetricCard label="Bearish" value={bearishCount} icon={ArrowDownRight} trend={bearishCount > 0 ? 'down' : 'neutral'} />
              <MetricCard label="Avg Impact" value={avgImpact} icon={Activity} />
            </>
          )}
        </div>

        {data && (
          <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
            <span>
              {data.total} catalyst{data.total !== 1 && 's'}
            </span>
            <span>{latency(data.latency_ms)}</span>
          </div>
        )}

        {isError && (
          <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to load catalysts{error instanceof Error && `: ${error.message}`}
          </div>
        )}

        <div className="mt-4 space-y-4">
          {isLoading && (
            <>
              <CatalystCardSkeleton />
              <CatalystCardSkeleton />
              <CatalystCardSkeleton />
            </>
          )}

          {data && data.catalysts.length === 0 && (
            <div className="flex flex-col items-center justify-center rounded-lg border border-border bg-card py-20 text-muted-foreground">
              <Target className="h-12 w-12" />
              <p className="mt-3 text-sm">No live catalysts found</p>
              <p className="mt-1 text-xs">
                Catalysts appear once narratives have enough corroboration to be tradeable.
              </p>
            </div>
          )}

          {data && data.catalysts.map((catalyst) => (
            <CatalystCard key={catalyst.run_id} catalyst={catalyst} />
          ))}
        </div>
      </div>
    </>
  );
}
