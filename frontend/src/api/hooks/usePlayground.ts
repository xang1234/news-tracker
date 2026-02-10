import { useMutation } from '@tanstack/react-query';
import { api } from '@/api/http';

// ── Embed ─────────────────────────────────────

export interface EmbedRequest {
  texts: string[];
  model?: 'auto' | 'finbert' | 'minilm';
}

export interface EmbedResultItem {
  embedding: number[];
  model: string;
  dimensions: number;
  cached: boolean;
}

export interface EmbedResponse {
  results: EmbedResultItem[];
  total: number;
  model: string;
  dimensions: number;
  latency_ms: number;
}

export function useEmbedMutation() {
  return useMutation({
    mutationFn: async (req: EmbedRequest): Promise<EmbedResponse> => {
      const { data } = await api.post<EmbedResponse>('/embed', req);
      return data;
    },
  });
}

// ── Sentiment ─────────────────────────────────

export interface SentimentScores {
  positive: number;
  neutral: number;
  negative: number;
}

export interface SentimentResultItem {
  label: string;
  confidence: number;
  scores: SentimentScores;
}

export interface SentimentResponse {
  results: SentimentResultItem[];
  total: number;
  model: string;
  latency_ms: number;
}

export function useSentimentMutation() {
  return useMutation({
    mutationFn: async (texts: string[]): Promise<SentimentResponse> => {
      const { data } = await api.post<SentimentResponse>('/sentiment', { texts });
      return data;
    },
  });
}

// ── NER ───────────────────────────────────────

export interface NEREntityItem {
  text: string;
  type: string;
  normalized: string;
  start: number;
  end: number;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface NERResultItem {
  entities: NEREntityItem[];
  text_length: number;
}

export interface NERResponse {
  results: NERResultItem[];
  total: number;
  latency_ms: number;
}

export function useNERMutation() {
  return useMutation({
    mutationFn: async (texts: string[]): Promise<NERResponse> => {
      const { data } = await api.post<NERResponse>('/ner', { texts });
      return data;
    },
  });
}

// ── Keywords ──────────────────────────────────

export interface KeywordItem {
  text: string;
  score: number;
  rank: number;
  lemma: string;
  count: number;
}

export interface KeywordsResultItem {
  keywords: KeywordItem[];
  text_length: number;
}

export interface KeywordsResponse {
  results: KeywordsResultItem[];
  total: number;
  latency_ms: number;
}

export function useKeywordsMutation() {
  return useMutation({
    mutationFn: async (req: {
      texts: string[];
      top_n?: number;
    }): Promise<KeywordsResponse> => {
      const { data } = await api.post<KeywordsResponse>('/keywords', req);
      return data;
    },
  });
}

// ── Events ────────────────────────────────────

export interface ExtractedEventItem {
  event_type: string;
  actor: string | null;
  action: string;
  object: string | null;
  time_ref: string | null;
  quantity: string | null;
  tickers: string[];
  confidence: number;
  span_start: number;
  span_end: number;
}

export interface EventsExtractResponse {
  events: ExtractedEventItem[];
  total: number;
  latency_ms: number;
}

export function useEventsExtractMutation() {
  return useMutation({
    mutationFn: async (req: {
      text: string;
      tickers?: string[];
    }): Promise<EventsExtractResponse> => {
      const { data } = await api.post<EventsExtractResponse>(
        '/events/extract',
        req,
      );
      return data;
    },
  });
}
