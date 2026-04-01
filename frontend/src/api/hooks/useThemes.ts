import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Filter params sent to GET /themes ──

export interface ThemeFilters {
  lifecycle_stage?: string;
  limit?: number;
  offset?: number;
}

// ── Theme item (shared between list and detail) ──

export interface ThemeItem {
  theme_id: string;
  name: string;
  top_keywords: string[];
  top_tickers: string[];
  top_entities: { name: string; score: number }[];
  lifecycle_stage: string;
  document_count: number;
  description: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
  centroid: number[] | null;
}

export interface ThemeListResponse {
  themes: ThemeItem[];
  total: number;
  latency_ms: number;
}

export interface ThemeDetailResponse {
  theme: ThemeItem;
  latency_ms: number;
}

export interface NarrativeAlertSummary {
  alert_id: string;
  trigger_type: string;
  severity: string;
  conviction_score: number | null;
  title: string;
  created_at: string;
}

export interface NarrativeBucketPoint {
  bucket_start: string;
  doc_count: number;
}

export interface NarrativeTickerCount {
  ticker: string;
  count: number;
}

export interface NarrativeDocumentItem {
  document_id: string;
  platform: string | null;
  title: string | null;
  content_preview: string | null;
  url: string | null;
  author_name: string | null;
  tickers: string[];
  authority_score: number | null;
  sentiment_label: string | null;
  sentiment_confidence: number | null;
  similarity: number;
  timestamp: string | null;
}

export interface NarrativeRunItem {
  run_id: string;
  theme_id: string;
  theme_name: string | null;
  status: string;
  label: string;
  conviction_score: number;
  current_rate_per_hour: number;
  current_acceleration: number;
  platform_count: number;
  top_tickers: NarrativeTickerCount[];
  last_document_at: string;
  started_at: string;
  recent_alerts: NarrativeAlertSummary[];
  sparkline: NarrativeBucketPoint[];
}

export interface NarrativeMomentumResponse {
  runs: NarrativeRunItem[];
  total: number;
  latency_ms: number;
}

export interface ThemeNarrativesResponse {
  theme_id: string;
  runs: NarrativeRunItem[];
  total: number;
  latency_ms: number;
}

export interface NarrativeRunDetailResponse {
  run: NarrativeRunItem;
  platform_timeline: Record<string, string>;
  ticker_counts: Record<string, number>;
  documents: NarrativeDocumentItem[];
  latency_ms: number;
}

export interface NarrativeRunDocumentsResponse {
  theme_id: string;
  run_id: string;
  documents: NarrativeDocumentItem[];
  total: number;
  latency_ms: number;
}

// ── Ranked themes ──

export interface RankedThemeItem {
  theme: ThemeItem;
  score: number;
  tier: number;
  components: Record<string, number>;
}

export interface RankedThemesResponse {
  themes: RankedThemeItem[];
  total: number;
  strategy: string;
  latency_ms: number;
}

// ── Theme metrics ──

export interface ThemeMetricsItem {
  date: string;
  document_count: number;
  sentiment_score: number | null;
  volume_zscore: number | null;
  velocity: number | null;
  acceleration: number | null;
  avg_authority: number | null;
  bullish_ratio: number | null;
}

export interface ThemeMetricsResponse {
  metrics: ThemeMetricsItem[];
  total: number;
  theme_id: string;
  latency_ms: number;
}

// ── Theme sentiment ──

export interface ThemeSentimentResponse {
  theme_id: string;
  bullish_ratio: number;
  bearish_ratio: number;
  neutral_ratio: number;
  avg_confidence: number;
  avg_authority: number | null;
  sentiment_velocity: number | null;
  extreme_sentiment: string | null;
  document_count: number;
  window_start: string;
  window_end: string;
  latency_ms: number;
}

// ── Theme documents ──

export interface ThemeDocumentItem {
  document_id: string;
  platform: string | null;
  title: string | null;
  content_preview: string | null;
  url: string | null;
  author_name: string | null;
  tickers: string[];
  authority_score: number | null;
  sentiment_label: string | null;
  sentiment_confidence: number | null;
  timestamp: string | null;
}

export interface ThemeDocumentsResponse {
  documents: ThemeDocumentItem[];
  total: number;
  theme_id: string;
  latency_ms: number;
}

// ── Theme events ──

export interface ThemeEventItem {
  event_id: string;
  doc_id: string;
  event_type: string;
  actor: string | null;
  action: string;
  object: string | null;
  time_ref: string | null;
  quantity: string | null;
  tickers: string[];
  confidence: number;
  source_doc_ids: string[];
  created_at: string | null;
}

export interface ThemeEventsResponse {
  events: ThemeEventItem[];
  total: number;
  theme_id: string;
  event_counts: Record<string, number>;
  investment_signal: string | null;
  latency_ms: number;
}

// ── Hooks ──

export function useThemes(filters?: ThemeFilters) {
  return useQuery({
    queryKey: queryKeys.themes(filters as Record<string, unknown>),
    queryFn: async (): Promise<ThemeListResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.lifecycle_stage) params.lifecycle_stage = filters.lifecycle_stage;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<ThemeListResponse>('/themes', { params });
      return data;
    },
    staleTime: 30_000,
    retry: 1,
  });
}

export function useRankedThemes(strategy?: string, maxTier?: number, limit?: number) {
  return useQuery({
    queryKey: queryKeys.rankedThemes(strategy, maxTier),
    queryFn: async (): Promise<RankedThemesResponse> => {
      const params: Record<string, unknown> = {};
      if (strategy) params.strategy = strategy;
      if (maxTier != null) params.max_tier = maxTier;
      if (limit != null) params.limit = limit;
      const { data } = await api.get<RankedThemesResponse>('/themes/ranked', { params });
      return data;
    },
    staleTime: 30_000,
    retry: 1,
  });
}

export function useThemeDetail(id: string | undefined) {
  return useQuery({
    queryKey: queryKeys.themeDetail(id ?? ''),
    queryFn: async (): Promise<ThemeDetailResponse> => {
      const { data } = await api.get<ThemeDetailResponse>(`/themes/${id}`);
      return data;
    },
    enabled: !!id,
    staleTime: 60_000,
  });
}

export function useThemeMomentum() {
  return useQuery({
    queryKey: queryKeys.themeMomentum(),
    queryFn: async (): Promise<NarrativeMomentumResponse> => {
      const { data } = await api.get<NarrativeMomentumResponse>('/themes/momentum');
      return data;
    },
    staleTime: 15_000,
    retry: 1,
  });
}

export function useThemeNarratives(id: string | undefined, runStatus?: string) {
  return useQuery({
    queryKey: queryKeys.themeNarratives(id ?? ''),
    queryFn: async (): Promise<ThemeNarrativesResponse> => {
      const params: Record<string, unknown> = {};
      if (runStatus) params.run_status = runStatus;
      const { data } = await api.get<ThemeNarrativesResponse>(`/themes/${id}/narratives`, { params });
      return data;
    },
    enabled: !!id,
    staleTime: 20_000,
    retry: 1,
  });
}

export function useThemeNarrativeDetail(themeId: string | undefined, runId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.themeNarrative(themeId ?? '', runId ?? ''),
    queryFn: async (): Promise<NarrativeRunDetailResponse> => {
      const { data } = await api.get<NarrativeRunDetailResponse>(`/themes/${themeId}/narratives/${runId}`);
      return data;
    },
    enabled: !!themeId && !!runId,
    staleTime: 20_000,
    retry: 1,
  });
}

export function useThemeNarrativeDocuments(themeId: string | undefined, runId: string | undefined, limit = 20) {
  return useQuery({
    queryKey: queryKeys.themeNarrativeDocuments(themeId ?? '', runId ?? ''),
    queryFn: async (): Promise<NarrativeRunDocumentsResponse> => {
      const { data } = await api.get<NarrativeRunDocumentsResponse>(
        `/themes/${themeId}/narratives/${runId}/documents`,
        { params: { limit } },
      );
      return data;
    },
    enabled: !!themeId && !!runId,
    staleTime: 20_000,
    retry: 1,
  });
}

export function useThemeMetrics(id: string | undefined) {
  return useQuery({
    queryKey: queryKeys.themeMetrics(id ?? ''),
    queryFn: async (): Promise<ThemeMetricsResponse> => {
      const { data } = await api.get<ThemeMetricsResponse>(`/themes/${id}/metrics`);
      return data;
    },
    enabled: !!id,
    staleTime: 60_000,
  });
}

export function useThemeSentiment(id: string | undefined) {
  return useQuery({
    queryKey: queryKeys.themeSentiment(id ?? ''),
    queryFn: async (): Promise<ThemeSentimentResponse> => {
      const { data } = await api.get<ThemeSentimentResponse>(`/themes/${id}/sentiment`);
      return data;
    },
    enabled: !!id,
    staleTime: 60_000,
  });
}

export function useThemeDocuments(id: string | undefined, filters?: { platform?: string; limit?: number; offset?: number }) {
  return useQuery({
    queryKey: queryKeys.themeDocuments(id ?? '', filters as Record<string, unknown>),
    queryFn: async (): Promise<ThemeDocumentsResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.platform) params.platform = filters.platform;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<ThemeDocumentsResponse>(`/themes/${id}/documents`, { params });
      return data;
    },
    enabled: !!id,
    staleTime: 30_000,
    retry: 1,
  });
}

export function useThemeEvents(id: string | undefined) {
  return useQuery({
    queryKey: queryKeys.themeEvents(id ?? ''),
    queryFn: async (): Promise<ThemeEventsResponse> => {
      const { data } = await api.get<ThemeEventsResponse>(`/themes/${id}/events`);
      return data;
    },
    enabled: !!id,
    staleTime: 60_000,
  });
}
