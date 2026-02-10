import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Filter params sent to GET /alerts ──

export interface AlertFilters {
  severity?: string;
  trigger_type?: string;
  theme_id?: string;
  acknowledged?: boolean;
  limit?: number;
  offset?: number;
}

// ── Alert item ──

export interface AlertItem {
  alert_id: string;
  theme_id: string;
  trigger_type: string;
  severity: string;
  title: string;
  message: string;
  trigger_data: Record<string, unknown>;
  acknowledged: boolean;
  created_at: string;
}

export interface AlertsResponse {
  alerts: AlertItem[];
  total: number;
  latency_ms: number;
}

export interface AlertAcknowledgeResponse {
  alert_id: string;
  acknowledged: boolean;
  latency_ms: number;
}

// ── Hooks ──

export function useAlerts(filters?: AlertFilters) {
  return useQuery({
    queryKey: queryKeys.alerts(filters as Record<string, unknown>),
    queryFn: async (): Promise<AlertsResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.severity) params.severity = filters.severity;
        if (filters.trigger_type) params.trigger_type = filters.trigger_type;
        if (filters.theme_id) params.theme_id = filters.theme_id;
        if (filters.acknowledged != null) params.acknowledged = filters.acknowledged;
        if (filters.limit != null) params.limit = filters.limit;
        if (filters.offset != null) params.offset = filters.offset;
      }
      const { data } = await api.get<AlertsResponse>('/alerts', { params });
      return data;
    },
    staleTime: 15_000,
    retry: 1,
  });
}

export function useAcknowledgeAlert() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (alertId: string): Promise<AlertAcknowledgeResponse> => {
      const { data } = await api.patch<AlertAcknowledgeResponse>(
        `/alerts/${alertId}/acknowledge`,
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    },
  });
}
