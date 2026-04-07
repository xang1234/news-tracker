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
  themeMomentum: () => ['themes', 'momentum'] as const,
  themeCatalysts: (limit?: number, days?: number) =>
    ['themes', 'catalysts', limit, days] as const,
  themeNarratives: (id: string) => ['themes', id, 'narratives'] as const,
  themeNarrative: (themeId: string, runId: string) =>
    ['themes', themeId, 'narratives', runId] as const,
  themeNarrativeDocuments: (themeId: string, runId: string) =>
    ['themes', themeId, 'narratives', runId, 'documents'] as const,
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
  entityDetail: (type: string, normalized: string) =>
    ['entities', type, normalized] as const,
  entityDocuments: (type: string, normalized: string, filters?: Record<string, unknown>) =>
    ['entities', type, normalized, 'documents', filters] as const,
  entityCooccurrence: (type: string, normalized: string) =>
    ['entities', type, normalized, 'cooccurrence'] as const,
  entitySentiment: (type: string, normalized: string) =>
    ['entities', type, normalized, 'sentiment'] as const,
  entityStats: () => ['entities', 'stats'] as const,
  trendingEntities: () => ['entities', 'trending'] as const,

  securities: (filters?: Record<string, unknown>) => ['securities', filters] as const,

  sources: (filters?: Record<string, unknown>) => ['sources', filters] as const,

  graphNodes: (filters?: Record<string, unknown>) => ['graph', 'nodes', filters] as const,
  graphSubgraph: (nodeId: string, depth?: number) => ['graph', 'subgraph', nodeId, depth] as const,

  settings: () => ['settings'] as const,

  // Intelligence layer
  intelHealth: () => ['intel', 'health'] as const,
  assertions: (filters?: Record<string, unknown>) => ['intel', 'assertions', filters] as const,
  assertionDetail: (id: string) => ['intel', 'assertions', id] as const,
  claims: (filters?: Record<string, unknown>) => ['intel', 'claims', filters] as const,
  divergences: (filters?: Record<string, unknown>) => ['intel', 'divergence', filters] as const,
  divergenceDetail: (issuerId: string) => ['intel', 'divergence', issuerId] as const,
  baskets: (themeId: string) => ['intel', 'baskets', themeId] as const,
  basketPaths: (themeId: string, conceptId: string) =>
    ['intel', 'baskets', themeId, 'paths', conceptId] as const,
} as const;
