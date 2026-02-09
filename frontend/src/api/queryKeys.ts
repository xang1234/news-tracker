/** Centralized query key factory for React Query — ensures cache key consistency */
export const queryKeys = {
  health: () => ['health'] as const,

  search: (query: string, filters?: Record<string, unknown>) =>
    ['search', query, filters] as const,

  documents: (filters?: Record<string, unknown>) => ['documents', filters] as const,
  document: (id: string) => ['documents', id] as const,

  stats: () => ['stats'] as const,

  themes: (filters?: Record<string, unknown>) => ['themes', filters] as const,
  rankedThemes: (strategy?: string, maxTier?: number) =>
    ['themes', 'ranked', strategy, maxTier] as const,
  themeDetail: (id: string) => ['themes', id] as const,
  themeMetrics: (id: string) => ['themes', id, 'metrics'] as const,
  themeSentiment: (id: string) => ['themes', id, 'sentiment'] as const,
  themeDocuments: (id: string, filters?: Record<string, unknown>) =>
    ['themes', id, 'documents', filters] as const,
  themeEvents: (id: string) => ['themes', id, 'events'] as const,

  alerts: (filters?: Record<string, unknown>) => ['alerts', filters] as const,

  feedback: () => ['feedback'] as const,
  feedbackStats: () => ['feedback', 'stats'] as const,

  queues: () => ['ops', 'queues'] as const,
  workers: () => ['ops', 'workers'] as const,
  version: () => ['ops', 'version'] as const,

  entities: (filters?: Record<string, unknown>) => ['entities', filters] as const,
  entity: (name: string) => ['entities', name] as const,

  settings: () => ['settings'] as const,
} as const;
