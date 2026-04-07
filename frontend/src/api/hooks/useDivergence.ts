import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types ──

export interface DivergenceItem {
  id: string;
  issuer_concept_id: string;
  issuer_name: string;
  theme_concept_id: string;
  theme_name: string;
  reason: string;
  severity: string;
  title: string;
  summary: string;
  narrative_score: number | null;
  filing_adoption_score: number | null;
  created_at: string;
}

export interface DivergenceFilters {
  severity?: string;
  reason_code?: string;
  issuer?: string;
  theme?: string;
  limit?: number;
  offset?: number;
}

export interface DivergenceListResponse {
  divergences: DivergenceItem[];
  total: number;
  severity_counts: { critical: number; warning: number; info: number };
  latency_ms: number;
}

export interface AdoptionBreakdown {
  section_coverage: number;
  section_depth: number;
  fact_alignment: number;
  temporal_consistency: number;
}

export interface DriftDimension {
  dimension: string;
  z_score: number;
  magnitude: number;
  is_unusual: boolean;
}

export interface DivergenceDetailResponse {
  divergences: DivergenceItem[];
  adoption: AdoptionBreakdown | null;
  drift_dimensions: DriftDimension[];
  latency_ms: number;
}

// ── Hooks ──

export function useDivergences(filters?: DivergenceFilters) {
  return useQuery({
    queryKey: queryKeys.divergences(filters as Record<string, unknown>),
    enabled: !!filters,
    queryFn: async (): Promise<DivergenceListResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.severity) params.severity = filters.severity;
        if (filters.reason_code) params.reason_code = filters.reason_code;
        if (filters.issuer) params.issuer = filters.issuer;
        if (filters.theme) params.theme = filters.theme;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<DivergenceListResponse>('/intel/divergence', { params });
      return data;
    },
    staleTime: 30_000,
    retry: 1,
  });
}

export function useDivergenceDetail(issuerId?: string) {
  return useQuery({
    queryKey: queryKeys.divergenceDetail(issuerId ?? ''),
    queryFn: async (): Promise<DivergenceDetailResponse> => {
      const { data } = await api.get<DivergenceDetailResponse>(`/intel/divergence/${issuerId}`);
      return data;
    },
    enabled: !!issuerId,
    staleTime: 30_000,
    retry: 1,
  });
}
