import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types ──

export interface EntityFilters {
  entity_type?: string;
  search?: string;
  sort?: string;
  limit?: number;
  offset?: number;
}

export interface EntitySummaryItem {
  type: string;
  normalized: string;
  mention_count: number;
  first_seen: string | null;
  last_seen: string | null;
}

export interface EntityListResponse {
  entities: EntitySummaryItem[];
  total: number;
  has_more: boolean;
  latency_ms: number;
}

export interface TrendingEntityItem {
  type: string;
  normalized: string;
  recent_count: number;
  baseline_count: number;
  spike_ratio: number;
}

export interface TrendingEntitiesResponse {
  trending: TrendingEntityItem[];
  latency_ms: number;
}

export interface EntityStatsResponse {
  total_entities: number;
  documents_with_entities: number;
  by_type: Record<string, number>;
  latency_ms: number;
}

export interface EntityDetailResponse {
  type: string;
  normalized: string;
  mention_count: number;
  first_seen: string | null;
  last_seen: string | null;
  platforms: Record<string, number>;
  graph_node_id: string | null;
  latency_ms: number;
}

export interface EntitySentimentResponse {
  avg_score: number | null;
  pos_count: number;
  neg_count: number;
  neu_count: number;
  trend: string;
  latency_ms: number;
}

export interface CooccurringEntityItem {
  type: string;
  normalized: string;
  cooccurrence_count: number;
  jaccard: number;
}

export interface CooccurrenceResponse {
  entities: CooccurringEntityItem[];
  latency_ms: number;
}

export interface EntityDocumentsResponse {
  documents: {
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
  }[];
  total: number;
  theme_id: string;
  latency_ms: number;
}

export interface EntityMergeResponse {
  affected_documents: number;
  merged_from: string;
  merged_to: string;
  latency_ms: number;
}

// ── Hooks ──

export function useEntities(filters?: EntityFilters) {
  return useQuery({
    queryKey: queryKeys.entities(filters as Record<string, unknown>),
    queryFn: async (): Promise<EntityListResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.entity_type) params.entity_type = filters.entity_type;
        if (filters.search) params.search = filters.search;
        if (filters.sort) params.sort = filters.sort;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<EntityListResponse>('/entities', { params });
      return data;
    },
    staleTime: 15_000,
    retry: 1,
  });
}

export function useEntityStats() {
  return useQuery({
    queryKey: queryKeys.entityStats(),
    queryFn: async (): Promise<EntityStatsResponse> => {
      const { data } = await api.get<EntityStatsResponse>('/entities/stats');
      return data;
    },
    staleTime: 30_000,
    retry: 1,
  });
}

export function useTrendingEntities() {
  return useQuery({
    queryKey: queryKeys.trendingEntities(),
    queryFn: async (): Promise<TrendingEntitiesResponse> => {
      const { data } = await api.get<TrendingEntitiesResponse>('/entities/trending');
      return data;
    },
    staleTime: 30_000,
    retry: 1,
  });
}

export function useEntityDetail(type: string, normalized: string) {
  return useQuery({
    queryKey: queryKeys.entityDetail(type, normalized),
    queryFn: async (): Promise<EntityDetailResponse> => {
      const { data } = await api.get<EntityDetailResponse>(
        `/entities/${encodeURIComponent(type)}/${encodeURIComponent(normalized)}`,
      );
      return data;
    },
    enabled: !!type && !!normalized,
    staleTime: 15_000,
    retry: 1,
  });
}

export function useEntityDocuments(
  type: string,
  normalized: string,
  filters?: { platform?: string; limit?: number; offset?: number },
) {
  return useQuery({
    queryKey: queryKeys.entityDocuments(type, normalized, filters as Record<string, unknown>),
    queryFn: async (): Promise<EntityDocumentsResponse> => {
      const params: Record<string, unknown> = {};
      if (filters?.platform) params.platform = filters.platform;
      if (filters?.limit != null) params.limit = filters.limit;
      if (filters?.offset != null) params.offset = filters.offset;
      const { data } = await api.get<EntityDocumentsResponse>(
        `/entities/${encodeURIComponent(type)}/${encodeURIComponent(normalized)}/documents`,
        { params },
      );
      return data;
    },
    enabled: !!type && !!normalized,
    staleTime: 15_000,
    retry: 1,
  });
}

export function useEntityCooccurrence(type: string, normalized: string) {
  return useQuery({
    queryKey: queryKeys.entityCooccurrence(type, normalized),
    queryFn: async (): Promise<CooccurrenceResponse> => {
      const { data } = await api.get<CooccurrenceResponse>(
        `/entities/${encodeURIComponent(type)}/${encodeURIComponent(normalized)}/cooccurrence`,
      );
      return data;
    },
    enabled: !!type && !!normalized,
    staleTime: 30_000,
    retry: 1,
  });
}

export function useEntitySentiment(type: string, normalized: string) {
  return useQuery({
    queryKey: queryKeys.entitySentiment(type, normalized),
    queryFn: async (): Promise<EntitySentimentResponse> => {
      const { data } = await api.get<EntitySentimentResponse>(
        `/entities/${encodeURIComponent(type)}/${encodeURIComponent(normalized)}/sentiment`,
      );
      return data;
    },
    enabled: !!type && !!normalized,
    staleTime: 30_000,
    retry: 1,
  });
}

export function useMergeEntity() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      fromType,
      fromNormalized,
      toType,
      toNormalized,
    }: {
      fromType: string;
      fromNormalized: string;
      toType: string;
      toNormalized: string;
    }): Promise<EntityMergeResponse> => {
      const { data } = await api.post<EntityMergeResponse>(
        `/entities/${encodeURIComponent(fromType)}/${encodeURIComponent(fromNormalized)}/merge`,
        { to_type: toType, to_normalized: toNormalized },
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['entities'] });
    },
  });
}
