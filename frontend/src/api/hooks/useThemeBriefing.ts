import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/http';
import { queryKeys } from '@/api/queryKeys';

// ── Types (mirror src/api/models.py: ThemeBriefingResponse) ──

export interface BriefingClauseItem {
  text: string;
  claim_ids: string[];
}

export interface ClaimCitationItem {
  claim_id: string;
  subject_text: string;
  predicate: string;
  object_text: string | null;
  source_type: string;
  source_id: string;
  source_span_start: number | null;
  source_span_end: number | null;
  snippet: string | null;
}

export interface ThemeBriefingResponse {
  theme_id: string;
  clauses: BriefingClauseItem[];
  citations: ClaimCitationItem[];
  generated_by: 'llm' | 'template';
  claim_count: number;
  model: string | null;
  generated_at: string | null;
  latency_ms: number;
}

/**
 * Fetch a grounded, claim-cited briefing for a theme.
 *
 * Longer staleTime than list queries — the brief is LLM-generated, so we
 * avoid regenerating on every focus.
 */
export function useThemeBriefing(themeId?: string) {
  return useQuery({
    queryKey: queryKeys.themeBriefing(themeId ?? ''),
    enabled: !!themeId,
    queryFn: async (): Promise<ThemeBriefingResponse> => {
      const { data } = await api.get<ThemeBriefingResponse>(`/themes/${themeId}/briefing`);
      return data;
    },
    staleTime: 60_000,
    retry: 1,
  });
}
