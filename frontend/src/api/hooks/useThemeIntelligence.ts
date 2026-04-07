import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types ──

export interface BasketMember {
  concept_id: string;
  concept_name: string;
  role: string;
  best_score: number;
  best_sign: number;
  min_hops: number;
  path_count: number;
  has_mixed_signals: boolean;
}

export interface BasketResponse {
  theme_id: string;
  members: BasketMember[];
  latency_ms: number;
}

export interface PathHop {
  from_concept: string;
  to_concept: string;
  predicate: string;
  sign: number;
  confidence: number;
  freshness: string;
}

export interface PathExplanationResponse {
  theme_id: string;
  concept_id: string;
  paths: { hops: PathHop[]; total_score: number }[];
  latency_ms: number;
}

// ── Hooks ──

export function useThemeBasket(themeId?: string) {
  return useQuery({
    queryKey: queryKeys.baskets(themeId ?? ''),
    queryFn: async (): Promise<BasketResponse> => {
      const { data } = await api.get<BasketResponse>(`/intel/baskets/${themeId}`);
      return data;
    },
    enabled: !!themeId,
    staleTime: 30_000,
    retry: 1,
  });
}

export function useBasketPaths(themeId?: string, conceptId?: string) {
  return useQuery({
    queryKey: queryKeys.basketPaths(themeId ?? '', conceptId ?? ''),
    queryFn: async (): Promise<PathExplanationResponse> => {
      const { data } = await api.get<PathExplanationResponse>(
        `/intel/baskets/${themeId}/paths/${conceptId}`,
      );
      return data;
    },
    enabled: !!themeId && !!conceptId,
    staleTime: 30_000,
    retry: 1,
  });
}
