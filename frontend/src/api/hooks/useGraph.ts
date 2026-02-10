import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types ──

export interface GraphNodeItem {
  node_id: string;
  node_type: string;
  name: string;
  metadata: Record<string, unknown>;
}

export interface GraphEdgeItem {
  source: string;
  target: string;
  relation: string;
  confidence: number;
}

export interface GraphNodesResponse {
  nodes: GraphNodeItem[];
  total: number;
  latency_ms: number;
}

export interface SubgraphResponse {
  nodes: GraphNodeItem[];
  edges: GraphEdgeItem[];
  center_node: string;
  latency_ms: number;
}

export interface GraphNodeFilters {
  node_type?: string;
  limit?: number;
}

// ── Propagation types ──

export interface PropagationImpactItem {
  node_id: string;
  impact: number;
  depth: number;
  relation: string;
  edge_confidence: number;
}

export interface PropagateRequest {
  source_node: string;
  sentiment_delta: number;
}

export interface PropagateResponse {
  source_node: string;
  sentiment_delta: number;
  impacts: PropagationImpactItem[];
  total_affected: number;
  latency_ms: number;
}

// ── Hooks ──

export function useGraphNodes(filters?: GraphNodeFilters) {
  return useQuery({
    queryKey: queryKeys.graphNodes(filters as Record<string, unknown>),
    queryFn: async (): Promise<GraphNodesResponse> => {
      const params: Record<string, unknown> = {};
      if (filters) {
        if (filters.node_type) params.node_type = filters.node_type;
        if (filters.limit != null) params.limit = filters.limit;
      }
      const { data } = await api.get<GraphNodesResponse>('/graph/nodes', { params });
      return data;
    },
    staleTime: 60_000,
    retry: 1,
  });
}

export function useGraphSubgraph(nodeId: string | undefined, depth?: number) {
  return useQuery({
    queryKey: queryKeys.graphSubgraph(nodeId ?? '', depth),
    queryFn: async (): Promise<SubgraphResponse> => {
      const params: Record<string, unknown> = {};
      if (depth != null) params.depth = depth;
      const { data } = await api.get<SubgraphResponse>(
        `/graph/nodes/${nodeId}/subgraph`,
        { params },
      );
      return data;
    },
    enabled: !!nodeId,
    staleTime: 60_000,
  });
}

export function useGraphPropagate() {
  return useMutation({
    mutationFn: async (request: PropagateRequest): Promise<PropagateResponse> => {
      const { data } = await api.post<PropagateResponse>('/graph/propagate', request);
      return data;
    },
  });
}
