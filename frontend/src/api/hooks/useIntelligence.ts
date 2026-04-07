import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types ──

export interface LaneHealthItem {
  lane: string;
  freshness: string;
  quality: string;
  quarantine: string;
  readiness: string;
  last_completed_at: string | null;
}

export interface QualityMetricItem {
  metric_type: string;
  value: number;
  severity: string;
  message: string;
}

export interface IntelHealthResponse {
  lanes: LaneHealthItem[];
  quality_metrics: QualityMetricItem[];
  overall_severity: string;
}

// ── Hooks ──

export function useIntelHealth() {
  return useQuery({
    queryKey: queryKeys.intelHealth(),
    queryFn: async (): Promise<IntelHealthResponse> => {
      const { data } = await api.get<IntelHealthResponse>('/intel/health');
      return data;
    },
    staleTime: 60_000,
    retry: 1,
  });
}

// Re-export useDivergences for dashboard use with a limit param
export { useDivergences } from '@/api/hooks/useDivergence';
