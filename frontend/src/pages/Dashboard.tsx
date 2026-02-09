import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { useHealth } from '@/api/hooks/useHealth';
import { Wifi, WifiOff, Cpu, HardDrive } from 'lucide-react';

export default function Dashboard() {
  const { data, isLoading, isError, error } = useHealth();

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

        {isError && (
          <div className="mb-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Failed to fetch health data
            {error instanceof Error && `: ${error.message}`}
          </div>
        )}

        {/* Health metric cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {isLoading ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : data ? (
            <>
              <MetricCard
                label="Service Status"
                value={data.status}
                icon={data.status === 'healthy' ? Wifi : WifiOff}
                trend={data.status === 'healthy' ? 'up' : 'down'}
                trendLabel={data.status === 'healthy' ? 'All systems operational' : 'Degraded'}
              />
              <MetricCard
                label="Models Loaded"
                value={Object.values(data.models_loaded).filter(Boolean).length}
                icon={Cpu}
                subtitle={`of ${Object.keys(data.models_loaded).length} available`}
              />
              <MetricCard
                label="Cache"
                value={data.cache_available ? 'Connected' : 'Unavailable'}
                icon={HardDrive}
                trend={data.cache_available ? 'up' : 'down'}
              />
              <MetricCard
                label="GPU"
                value={data.gpu_available ? 'Available' : 'CPU Only'}
                icon={Cpu}
                trend={data.gpu_available ? 'up' : 'neutral'}
              />
            </>
          ) : null}
        </div>

        {/* Placeholder sections for future content */}
        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border border-border bg-card p-6">
            <h3 className="text-sm font-medium text-muted-foreground">Data Overview</h3>
            <p className="mt-2 text-xs text-muted-foreground">
              Document counts and processing coverage will appear here once the /stats/overview
              endpoint is available.
            </p>
          </div>
          <div className="rounded-lg border border-border bg-card p-6">
            <h3 className="text-sm font-medium text-muted-foreground">Queue Depths</h3>
            <p className="mt-2 text-xs text-muted-foreground">
              Redis stream queue depths will appear here once the /ops/queues endpoint is available.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
