import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

export interface SearchFilters {
  platforms?: string[];
  tickers?: string[];
  threshold?: number;
  min_authority_score?: number;
  limit?: number;
}

export interface SearchResultItem {
  document_id: string;
  score: number;
  platform: string | null;
  title: string | null;
  content_preview: string | null;
  url: string | null;
  author_name: string | null;
  author_verified: boolean;
  tickers: string[];
  authority_score: number | null;
  timestamp: string | null;
}

export interface SearchResponse {
  results: SearchResultItem[];
  total: number;
  latency_ms: number;
}

export function useSearch(query: string, filters?: SearchFilters) {
  return useQuery({
    queryKey: queryKeys.search(query, filters as Record<string, unknown>),
    queryFn: async (): Promise<SearchResponse> => {
      const body: Record<string, unknown> = { query };

      if (filters) {
        if (filters.platforms?.length) body.platforms = filters.platforms;
        if (filters.tickers?.length) body.tickers = filters.tickers;
        if (filters.threshold != null) body.threshold = filters.threshold;
        if (filters.min_authority_score != null)
          body.min_authority_score = filters.min_authority_score;
        if (filters.limit != null) body.limit = filters.limit;
      }

      const { data } = await api.post<SearchResponse>('/search/similar', body);
      return data;
    },
    enabled: query.trim().length > 0,
    staleTime: 60_000,
    retry: 1,
  });
}
