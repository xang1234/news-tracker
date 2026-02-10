import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types ──

export interface SourceFilters {
  platform?: string;
  search?: string;
  active_only?: boolean;
  limit?: number;
  offset?: number;
}

export interface SourceItem {
  platform: string;
  identifier: string;
  display_name: string;
  description: string;
  is_active: boolean;
  metadata: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
}

export interface SourcesListResponse {
  sources: SourceItem[];
  total: number;
  has_more: boolean;
  latency_ms: number;
}

export interface CreateSourcePayload {
  platform: string;
  identifier: string;
  display_name?: string;
  description?: string;
  metadata?: Record<string, unknown>;
}

export interface UpdateSourcePayload {
  platform: string;
  identifier: string;
  display_name?: string;
  description?: string;
  is_active?: boolean;
  metadata?: Record<string, unknown>;
}

// ── Hooks ──

export function useSources(filters?: SourceFilters) {
  return useQuery({
    queryKey: queryKeys.sources(filters as Record<string, unknown>),
    queryFn: async (): Promise<SourcesListResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.platform) params.platform = filters.platform;
        if (filters.search) params.search = filters.search;
        if (filters.active_only) params.active_only = filters.active_only;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<SourcesListResponse>('/sources', { params });
      return data;
    },
    staleTime: 15_000,
    retry: 1,
  });
}

export function useCreateSource() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (payload: CreateSourcePayload): Promise<SourceItem> => {
      const { data } = await api.post<SourceItem>('/sources', payload);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] });
    },
  });
}

export function useUpdateSource() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (payload: UpdateSourcePayload): Promise<SourceItem> => {
      const { platform, identifier, ...body } = payload;
      const { data } = await api.put<SourceItem>(
        `/sources/${encodeURIComponent(platform)}/${encodeURIComponent(identifier)}`,
        body,
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] });
    },
  });
}

export function useDeactivateSource() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ platform, identifier }: { platform: string; identifier: string }) => {
      await api.delete(
        `/sources/${encodeURIComponent(platform)}/${encodeURIComponent(identifier)}`,
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] });
    },
  });
}
