import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types ──

export interface AssertionItem {
  assertion_id: string;
  subject_concept_id: string;
  predicate: string;
  object_concept_id: string | null;
  confidence: number;
  status: string;
  support_count: number;
  contradiction_count: number;
  source_diversity: number;
  first_seen_at: string;
  last_evidence_at: string;
  valid_from: string | null;
  valid_to: string | null;
}

export interface AssertionFilters {
  concept_id?: string;
  predicate?: string;
  status?: string;
  min_confidence?: number;
  limit?: number;
  offset?: number;
}

export interface AssertionListResponse {
  assertions: AssertionItem[];
  total: number;
  latency_ms: number;
}

export interface ClaimSummary {
  claim_id: string;
  lane: string;
  source_id: string;
  source_type: string;
  subject_text: string;
  predicate: string;
  object_text: string;
  confidence: number;
  extraction_method: string;
  created_at: string;
}

export interface ClaimLinkItem {
  assertion_id: string;
  claim_id: string;
  link_type: string;
  contribution_weight: number;
  claim: ClaimSummary | null;
}

export interface AssertionDetailResponse {
  assertion: AssertionItem;
  claim_links: ClaimLinkItem[];
  latency_ms: number;
}

export interface ClaimItem {
  claim_id: string;
  claim_key: string;
  lane: string;
  source_id: string;
  source_type: string;
  subject_text: string;
  subject_concept_id: string | null;
  predicate: string;
  object_text: string;
  object_concept_id: string | null;
  confidence: number;
  extraction_method: string;
  status: string;
  created_at: string;
}

export interface ClaimFilters {
  assertion_id?: string;
  lane?: string;
  source_id?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

export interface ClaimListResponse {
  claims: ClaimItem[];
  total: number;
  latency_ms: number;
}

// ── Hooks ──

export function useAssertions(filters?: AssertionFilters) {
  return useQuery({
    queryKey: queryKeys.assertions(filters as Record<string, unknown>),
    enabled: !!filters,
    queryFn: async (): Promise<AssertionListResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.concept_id) params.concept_id = filters.concept_id;
        if (filters.predicate) params.predicate = filters.predicate;
        if (filters.status) params.status = filters.status;
        if (filters.min_confidence != null) params.min_confidence = filters.min_confidence;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<AssertionListResponse>('/intel/assertions', { params });
      return data;
    },
    staleTime: 30_000,
    retry: 1,
  });
}

export function useAssertionDetail(id?: string) {
  return useQuery({
    queryKey: queryKeys.assertionDetail(id ?? ''),
    queryFn: async (): Promise<AssertionDetailResponse> => {
      const { data } = await api.get<AssertionDetailResponse>(`/intel/assertions/${id}`);
      return data;
    },
    enabled: !!id,
    staleTime: 30_000,
    retry: 1,
  });
}

export function useClaims(filters?: ClaimFilters) {
  return useQuery({
    queryKey: queryKeys.claims(filters as Record<string, unknown>),
    enabled: !!filters,
    queryFn: async (): Promise<ClaimListResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.assertion_id) params.assertion_id = filters.assertion_id;
        if (filters.lane) params.lane = filters.lane;
        if (filters.source_id) params.source_id = filters.source_id;
        if (filters.status) params.status = filters.status;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<ClaimListResponse>('/intel/claims', { params });
      return data;
    },
    staleTime: 30_000,
    retry: 1,
  });
}
