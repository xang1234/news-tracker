import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types ──

export interface SecurityFilters {
  search?: string;
  active_only?: boolean;
  exchange?: string;
  limit?: number;
  offset?: number;
}

export interface SecurityItem {
  ticker: string;
  exchange: string;
  name: string;
  aliases: string[];
  sector: string;
  country: string;
  currency: string;
  is_active: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface SecuritiesListResponse {
  securities: SecurityItem[];
  total: number;
  has_more: boolean;
  latency_ms: number;
}

export interface CreateSecurityPayload {
  ticker: string;
  exchange?: string;
  name: string;
  aliases?: string[];
  sector?: string;
  country?: string;
  currency?: string;
}

export interface UpdateSecurityPayload {
  ticker: string;
  exchange: string;
  name?: string;
  aliases?: string[];
  sector?: string;
  country?: string;
  currency?: string;
}

// ── Hooks ──

export function useSecurities(filters?: SecurityFilters) {
  return useQuery({
    queryKey: queryKeys.securities(filters as Record<string, unknown>),
    queryFn: async (): Promise<SecuritiesListResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.search) params.search = filters.search;
        if (filters.active_only) params.active_only = filters.active_only;
        if (filters.exchange) params.exchange = filters.exchange;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<SecuritiesListResponse>('/securities', { params });
      return data;
    },
    staleTime: 15_000,
    retry: 1,
  });
}

export function useCreateSecurity() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (payload: CreateSecurityPayload): Promise<SecurityItem> => {
      const { data } = await api.post<SecurityItem>('/securities', payload);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['securities'] });
    },
  });
}

export function useUpdateSecurity() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (payload: UpdateSecurityPayload): Promise<SecurityItem> => {
      const { ticker, exchange, ...body } = payload;
      const { data } = await api.put<SecurityItem>(
        `/securities/${encodeURIComponent(ticker)}/${encodeURIComponent(exchange)}`,
        body,
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['securities'] });
    },
  });
}

export function useDeactivateSecurity() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ ticker, exchange }: { ticker: string; exchange: string }) => {
      await api.delete(
        `/securities/${encodeURIComponent(ticker)}/${encodeURIComponent(exchange)}`,
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['securities'] });
    },
  });
}
