import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Filter params sent to GET /documents ──

export interface DocumentFilters {
  platform?: string;
  content_type?: string;
  ticker?: string;
  q?: string;
  since?: string;
  until?: string;
  max_spam?: number;
  min_authority?: number;
  sort?: string;
  order?: string;
  limit?: number;
  offset?: number;
}

// ── List view (lightweight, no embeddings) ──

export interface DocumentListItem {
  document_id: string;
  platform: string | null;
  content_type: string | null;
  title: string | null;
  content_preview: string | null;
  url: string | null;
  author_name: string | null;
  author_verified: boolean;
  author_followers: number | null;
  tickers: string[];
  spam_score: number | null;
  authority_score: number | null;
  sentiment_label: string | null;
  sentiment_confidence: number | null;
  engagement: Record<string, number>;
  theme_ids: string[];
  timestamp: string | null;
  fetched_at: string | null;
}

export interface DocumentListResponse {
  documents: DocumentListItem[];
  total: number;
  page_size: number;
  offset: number;
  latency_ms: number;
}

// ── Full document detail ──

export interface DocumentDetail {
  document_id: string;
  platform: string | null;
  content_type: string | null;
  title: string | null;
  content: string | null;
  content_preview: string | null;
  url: string | null;
  author_id: string | null;
  author_name: string | null;
  author_verified: boolean;
  author_followers: number | null;
  tickers: string[];
  spam_score: number | null;
  bot_probability: number | null;
  authority_score: number | null;
  sentiment: Record<string, number | string> | null;
  sentiment_label: string | null;
  sentiment_confidence: number | null;
  entities: { type: string; name: string }[];
  keywords: { word: string; score: number }[];
  events: { type: string; actor: string; action: string; object: string; time_ref: string }[];
  urls_mentioned: string[];
  engagement: Record<string, number>;
  theme_ids: string[];
  has_embedding: boolean;
  has_embedding_minilm: boolean;
  timestamp: string | null;
  fetched_at: string | null;
}

// ── Stats response ──

export interface DocumentStats {
  total_count: number;
  platform_counts: { platform: string; count: number }[];
  embedding_coverage: { finbert_pct: number; minilm_pct: number };
  sentiment_coverage: number;
  earliest_document: string | null;
  latest_document: string | null;
  latest_fetched_at: string | null;
}

// ── Hooks ──

export function useDocuments(filters?: DocumentFilters) {
  return useQuery({
    queryKey: queryKeys.documents(filters as Record<string, unknown>),
    queryFn: async (): Promise<DocumentListResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.platform) params.platform = filters.platform;
        if (filters.content_type) params.content_type = filters.content_type;
        if (filters.ticker) params.ticker = filters.ticker;
        if (filters.q) params.q = filters.q;
        if (filters.since) params.since = filters.since;
        if (filters.until) params.until = filters.until;
        if (filters.max_spam != null) params.max_spam = filters.max_spam;
        if (filters.min_authority != null) params.min_authority = filters.min_authority;
        if (filters.sort) params.sort = filters.sort;
        if (filters.order) params.order = filters.order;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<DocumentListResponse>('/documents', { params });
      return data;
    },
    staleTime: 30_000,
    retry: 1,
  });
}

export function useDocument(id: string | undefined) {
  return useQuery({
    queryKey: queryKeys.document(id ?? ''),
    queryFn: async (): Promise<DocumentDetail> => {
      const { data } = await api.get<DocumentDetail>(`/documents/${id}`);
      return data;
    },
    enabled: !!id,
    staleTime: 60_000,
  });
}

export function useDocumentStats(opts?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: queryKeys.stats(),
    queryFn: async (): Promise<DocumentStats> => {
      const { data } = await api.get<DocumentStats>('/documents/stats');
      return data;
    },
    staleTime: 120_000,
    refetchInterval: opts?.refetchInterval,
  });
}
