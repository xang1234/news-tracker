import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

export interface HealthData {
  status: string;
  models_loaded: Record<string, boolean>;
  cache_available: boolean;
  gpu_available: boolean;
  service_stats: Record<string, unknown>;
  queue_depths: Record<string, number>;
}

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health(),
    queryFn: async (): Promise<HealthData> => {
      const { data } = await api.get<HealthData>('/health');
      return data;
    },
    staleTime: 30_000,
    retry: 2,
    refetchInterval: 60_000,
  });
}
