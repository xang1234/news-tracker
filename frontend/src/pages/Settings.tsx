import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Radio, Plus, Upload, ChevronLeft, ChevronRight, RefreshCw, Shield } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MetricCard, MetricCardSkeleton } from '@/components/domain/MetricCard';
import { SourcesTable, SourcesTableSkeleton } from '@/components/domain/SourcesTable';
import { SecuritiesTable, SecuritiesTableSkeleton } from '@/components/domain/SecuritiesTable';
import {
  SourcesFilters,
  DEFAULT_SOURCES_FILTERS,
  type SourcesFilterValues,
} from '@/components/domain/SourcesFilters';
import {
  SecuritiesFilters,
  DEFAULT_SECURITIES_FILTERS,
  type SecuritiesFilterValues,
} from '@/components/domain/SecuritiesFilters';
import { SourceFormModal, type SourceFormData } from '@/components/domain/SourceFormModal';
import { SecurityFormModal } from '@/components/domain/SecurityFormModal';
import { BulkAddSourcesModal } from '@/components/domain/BulkAddSourcesModal';
import {
  useSources,
  useCreateSource,
  useUpdateSource,
  useDeactivateSource,
  useTriggerIngestion,
  type SourceFilters as ApiSourceFilters,
  type SourceItem,
} from '@/api/hooks/useSources';
import {
  useSecurities,
  useCreateSecurity,
  useUpdateSecurity,
  useDeactivateSecurity,
  type SecurityFilters as ApiSecurityFilters,
  type SecurityItem,
} from '@/api/hooks/useSecurities';
import { latency } from '@/lib/formatters';
import { cn } from '@/lib/utils';
import axios from 'axios';

type SettingsTab = 'sources' | 'securities';

function buildSourceApiFilters(f: SourcesFilterValues, offset: number): ApiSourceFilters {
  return {
    platform: f.platform || undefined,
    search: f.search || undefined,
    active_only: f.activeOnly || undefined,
    limit: f.limit,
    offset,
  };
}

function buildSecurityApiFilters(f: SecuritiesFilterValues, offset: number): ApiSecurityFilters {
  return {
    search: f.search || undefined,
    active_only: f.activeOnly || undefined,
    exchange: f.exchange || undefined,
    limit: f.limit,
    offset,
  };
}

export default function Settings() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedTab = searchParams.get('tab');
  const initialTab: SettingsTab = requestedTab === 'securities' ? 'securities' : 'sources';
  const [activeTab, setActiveTab] = useState<SettingsTab>(initialTab);

  // --- Sources state ---
  const [srcFiltersOpen, setSrcFiltersOpen] = useState(false);
  const [srcFilters, setSrcFilters] = useState<SourcesFilterValues>(DEFAULT_SOURCES_FILTERS);
  const [srcOffset, setSrcOffset] = useState(0);
  const [addOpen, setAddOpen] = useState(false);
  const [bulkOpen, setBulkOpen] = useState(false);
  const [editItem, setEditItem] = useState<SourceItem | null>(null);

  const srcApiFilters = buildSourceApiFilters(srcFilters, srcOffset);
  const sources = useSources(srcApiFilters);
  const createMutation = useCreateSource();
  const updateMutation = useUpdateSource();
  const deactivateMutation = useDeactivateSource();
  const triggerMutation = useTriggerIngestion();

  useEffect(() => {
    if (triggerMutation.isSuccess || triggerMutation.isError) {
      const timer = setTimeout(() => triggerMutation.reset(), 5000);
      return () => clearTimeout(timer);
    }
  }, [triggerMutation.isSuccess, triggerMutation.isError]);

  // --- Securities state ---
  const [secFiltersOpen, setSecFiltersOpen] = useState(false);
  const [secFilters, setSecFilters] = useState<SecuritiesFilterValues>(DEFAULT_SECURITIES_FILTERS);
  const [secOffset, setSecOffset] = useState(0);
  const [secAddOpen, setSecAddOpen] = useState(false);
  const [secEditItem, setSecEditItem] = useState<SecurityItem | null>(null);

  const secApiFilters = buildSecurityApiFilters(secFilters, secOffset);
  const securities = useSecurities(secApiFilters);
  const secCreateMutation = useCreateSecurity();
  const secUpdateMutation = useUpdateSecurity();
  const secDeactivateMutation = useDeactivateSecurity();

  // Sources pagination
  const srcTotal = sources.data?.total ?? 0;
  const srcShowing = sources.data ? sources.data.sources.length : 0;
  const srcHasNext = sources.data?.has_more ?? false;
  const srcHasPrev = srcOffset > 0;

  // Securities pagination
  const secTotal = securities.data?.total ?? 0;
  const secShowing = securities.data ? securities.data.securities.length : 0;
  const secHasNext = securities.data?.has_more ?? false;
  const secHasPrev = secOffset > 0;

  const twitterCount = sources.data?.sources.filter((s) => s.platform === 'twitter').length ?? 0;
  const redditCount = sources.data?.sources.filter((s) => s.platform === 'reddit').length ?? 0;
  const substackCount = sources.data?.sources.filter((s) => s.platform === 'substack').length ?? 0;

  useEffect(() => {
    const nextTab: SettingsTab = requestedTab === 'securities' ? 'securities' : 'sources';
    setActiveTab((current) => (current === nextTab ? current : nextTab));
  }, [requestedTab]);

  const selectTab = (tab: SettingsTab) => {
    setActiveTab(tab);
    setSearchParams(tab === 'sources' ? {} : { tab }, { replace: true });
  };

  return (
    <>
      <Header title="Settings" />
      <div className="mx-auto max-w-5xl p-6">
        {/* Tab bar */}
        <div className="border-b border-border">
          <div className="flex gap-0" role="tablist" aria-label="Settings tabs">
            {(['sources', 'securities'] as const).map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={activeTab === t}
                onClick={() => selectTab(t)}
                className={cn(
                  'border-b-2 px-4 py-2.5 text-sm font-medium transition-colors',
                  activeTab === t
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground',
                )}
              >
                {t === 'sources' ? 'Sources' : 'Securities'}
              </button>
            ))}
          </div>
        </div>

        {/* Sources tab */}
        {activeTab === 'sources' && (
          <div className="mt-6">
            <p className="text-sm text-muted-foreground">
              Manage ingestion sources for Twitter, Reddit, and Substack.
            </p>

            <div className="mt-4 flex items-center justify-end gap-2">
              <button
                onClick={() => triggerMutation.mutate()}
                disabled={triggerMutation.isPending}
                className="flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-sm font-medium text-muted-foreground hover:bg-secondary hover:text-foreground disabled:opacity-50"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${triggerMutation.isPending ? 'animate-spin' : ''}`} />
                {triggerMutation.isPending ? 'Polling...' : 'Poll Now'}
              </button>
              <button
                onClick={() => setBulkOpen(true)}
                className="flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-sm font-medium text-muted-foreground hover:bg-secondary hover:text-foreground"
              >
                <Upload className="h-3.5 w-3.5" />
                Bulk Add
              </button>
              <button
                onClick={() => setAddOpen(true)}
                className="flex items-center gap-1.5 rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              >
                <Plus className="h-3.5 w-3.5" />
                Add Source
              </button>
            </div>

            {triggerMutation.isSuccess && (
              <div className="mt-3 rounded border border-green-500/30 bg-green-500/10 px-4 py-2 text-sm text-green-400">
                {triggerMutation.data.message}
              </div>
            )}
            {triggerMutation.isError && (
              <div className="mt-3 rounded border border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">
                {axios.isAxiosError(triggerMutation.error)
                  ? triggerMutation.error.response?.data?.detail ?? triggerMutation.error.message
                  : 'Failed to trigger ingestion'}
              </div>
            )}

            <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
              {sources.isLoading ? (
                <>
                  <MetricCardSkeleton />
                  <MetricCardSkeleton />
                  <MetricCardSkeleton />
                  <MetricCardSkeleton />
                </>
              ) : (
                <>
                  <MetricCard label="Total Sources" value={srcTotal} icon={Radio} />
                  <MetricCard label="Twitter" value={twitterCount} />
                  <MetricCard label="Reddit" value={redditCount} />
                  <MetricCard label="Substack" value={substackCount} />
                </>
              )}
            </div>

            <div className="mt-4">
              <SourcesFilters
                isOpen={srcFiltersOpen}
                onToggle={() => setSrcFiltersOpen(!srcFiltersOpen)}
                filters={srcFilters}
                onChange={(next) => { setSrcFilters(next); setSrcOffset(0); }}
              />
            </div>

            {sources.data && (
              <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
                <span>{srcTotal} source{srcTotal !== 1 ? 's' : ''}</span>
                <span>{latency(sources.data.latency_ms)}</span>
              </div>
            )}

            {sources.isError && (
              <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                Failed to load sources
              </div>
            )}

            <div className="mt-4">
              {sources.isLoading && <SourcesTableSkeleton />}
              {sources.data && sources.data.sources.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                  <Radio className="h-12 w-12" />
                  <p className="mt-3 text-sm">No sources found</p>
                </div>
              )}
              {sources.data && sources.data.sources.length > 0 && (
                <SourcesTable
                  sources={sources.data.sources}
                  onEdit={(p, i) => {
                    const item = sources.data?.sources.find((s) => s.platform === p && s.identifier === i);
                    if (item) setEditItem(item);
                  }}
                  onDeactivate={(p, i) => {
                    if (confirm(`Deactivate ${p}/${i}?`)) deactivateMutation.mutate({ platform: p, identifier: i });
                  }}
                />
              )}
            </div>

            {sources.data && srcTotal > 0 && (
              <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground">
                <span>Showing {srcOffset + 1}–{srcOffset + srcShowing} of {srcTotal}</span>
                <div className="flex items-center gap-2">
                  <button type="button" disabled={!srcHasPrev} onClick={() => setSrcOffset(Math.max(0, srcOffset - srcFilters.limit))} className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40">
                    <ChevronLeft className="h-3.5 w-3.5" /> Previous
                  </button>
                  <button type="button" disabled={!srcHasNext} onClick={() => setSrcOffset(srcOffset + srcFilters.limit)} className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40">
                    Next <ChevronRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            )}

            <SourceFormModal
              isOpen={addOpen}
              onClose={() => setAddOpen(false)}
              mode="create"
              isPending={createMutation.isPending}
              onSubmit={(vals: SourceFormData) => { createMutation.mutate(vals, { onSuccess: () => setAddOpen(false) }); }}
            />
            <BulkAddSourcesModal isOpen={bulkOpen} onClose={() => setBulkOpen(false)} />
            {editItem && (
              <SourceFormModal
                isOpen={!!editItem}
                onClose={() => setEditItem(null)}
                mode="edit"
                initialValues={{ platform: editItem.platform, identifier: editItem.identifier, display_name: editItem.display_name, description: editItem.description }}
                isPending={updateMutation.isPending}
                onSubmit={(vals: SourceFormData) => {
                  updateMutation.mutate({ platform: editItem.platform, identifier: editItem.identifier, display_name: vals.display_name, description: vals.description }, { onSuccess: () => setEditItem(null) });
                }}
              />
            )}
          </div>
        )}

        {/* Securities tab */}
        {activeTab === 'securities' && (
          <div className="mt-6">
            <div className="flex items-center justify-end">
              <button
                onClick={() => setSecAddOpen(true)}
                className="flex items-center gap-1.5 rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              >
                <Plus className="h-3.5 w-3.5" />
                Add Security
              </button>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3">
              {securities.isLoading ? (
                <>
                  <MetricCardSkeleton />
                  <MetricCardSkeleton />
                </>
              ) : (
                <>
                  <MetricCard label="Total Securities" value={secTotal} icon={Shield} />
                  <MetricCard label="Active" value={securities.data?.securities.filter((s) => s.is_active).length ?? 0} />
                </>
              )}
            </div>

            <div className="mt-4">
              <SecuritiesFilters
                isOpen={secFiltersOpen}
                onToggle={() => setSecFiltersOpen(!secFiltersOpen)}
                filters={secFilters}
                onChange={(next) => { setSecFilters(next); setSecOffset(0); }}
              />
            </div>

            {securities.data && (
              <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
                <span>{secTotal} securit{secTotal !== 1 ? 'ies' : 'y'}</span>
                <span>{latency(securities.data.latency_ms)}</span>
              </div>
            )}

            {securities.isError && (
              <div className="mt-4 rounded border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                Failed to load securities
              </div>
            )}

            <div className="mt-4">
              {securities.isLoading && <SecuritiesTableSkeleton />}
              {securities.data && securities.data.securities.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                  <Shield className="h-12 w-12" />
                  <p className="mt-3 text-sm">No securities found</p>
                </div>
              )}
              {securities.data && securities.data.securities.length > 0 && (
                <SecuritiesTable
                  securities={securities.data.securities}
                  onEdit={(t, e) => {
                    const item = securities.data?.securities.find((s) => s.ticker === t && s.exchange === e);
                    if (item) setSecEditItem(item);
                  }}
                  onDeactivate={(t, e) => {
                    if (confirm(`Deactivate ${t} (${e})?`)) secDeactivateMutation.mutate({ ticker: t, exchange: e });
                  }}
                />
              )}
            </div>

            {securities.data && secTotal > 0 && (
              <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground">
                <span>Showing {secOffset + 1}–{secOffset + secShowing} of {secTotal}</span>
                <div className="flex items-center gap-2">
                  <button type="button" disabled={!secHasPrev} onClick={() => setSecOffset(Math.max(0, secOffset - secFilters.limit))} className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40">
                    <ChevronLeft className="h-3.5 w-3.5" /> Previous
                  </button>
                  <button type="button" disabled={!secHasNext} onClick={() => setSecOffset(secOffset + secFilters.limit)} className="flex items-center gap-1 rounded border border-border px-3 py-1.5 text-sm hover:bg-secondary/50 disabled:opacity-40">
                    Next <ChevronRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            )}

            <SecurityFormModal
              isOpen={secAddOpen}
              onClose={() => setSecAddOpen(false)}
              mode="create"
              isPending={secCreateMutation.isPending}
              onSubmit={(vals) => { secCreateMutation.mutate(vals, { onSuccess: () => setSecAddOpen(false) }); }}
            />
            {secEditItem && (
              <SecurityFormModal
                isOpen={!!secEditItem}
                onClose={() => setSecEditItem(null)}
                mode="edit"
                initialValues={secEditItem}
                isPending={secUpdateMutation.isPending}
                onSubmit={(vals) => {
                  secUpdateMutation.mutate({ ticker: secEditItem.ticker, exchange: secEditItem.exchange, ...vals }, { onSuccess: () => setSecEditItem(null) });
                }}
              />
            )}
          </div>
        )}
      </div>
    </>
  );
}
