/** Platform display names and colors */
export const PLATFORMS: Record<string, { label: string; color: string }> = {
  twitter: { label: 'Twitter', color: 'bg-sky-500/20 text-sky-400' },
  reddit: { label: 'Reddit', color: 'bg-orange-500/20 text-orange-400' },
  substack: { label: 'Substack', color: 'bg-violet-500/20 text-violet-400' },
  news_api: { label: 'News', color: 'bg-emerald-500/20 text-emerald-400' },
  newsfilter: { label: 'NewsFilter', color: 'bg-emerald-500/20 text-emerald-400' },
  marketaux: { label: 'MarketAux', color: 'bg-teal-500/20 text-teal-400' },
  finlight: { label: 'FinLight', color: 'bg-amber-500/20 text-amber-400' },
};

/** Sentiment label colors */
export const SENTIMENT_COLORS: Record<string, string> = {
  positive: 'bg-emerald-500/20 text-emerald-400',
  neutral: 'bg-slate-500/20 text-slate-400',
  negative: 'bg-red-500/20 text-red-400',
};

/** Lifecycle stage colors */
export const LIFECYCLE_COLORS: Record<string, string> = {
  emerging: 'bg-cyan-500/20 text-cyan-400',
  accelerating: 'bg-emerald-500/20 text-emerald-400',
  mature: 'bg-blue-500/20 text-blue-400',
  fading: 'bg-amber-500/20 text-amber-400',
};

/** Severity colors */
export const SEVERITY_COLORS: Record<string, string> = {
  critical: 'bg-red-500/20 text-red-400',
  warning: 'bg-amber-500/20 text-amber-400',
  info: 'bg-cyan-500/20 text-cyan-400',
};

/** Graph node type colors */
export const NODE_TYPE_COLORS: Record<string, string> = {
  ticker: '#3b82f6',
  theme: '#22c55e',
  technology: '#a855f7',
};

/** Graph relation colors */
export const RELATION_COLORS: Record<string, string> = {
  supplies_to: '#60a5fa',
  depends_on: '#34d399',
  competes_with: '#f87171',
  enables: '#c084fc',
  derived_from: '#fbbf24',
};

/** Alert trigger type display labels */
export const TRIGGER_TYPE_LABELS: Record<string, string> = {
  volume_spike: 'Volume Spike',
  sentiment_shift: 'Sentiment Shift',
  new_theme: 'New Theme',
  theme_acceleration: 'Theme Acceleration',
  extreme_sentiment: 'Extreme Sentiment',
  cross_theme: 'Cross-Theme',
};

/** Similarity score color thresholds */
export function similarityColor(score: number): string {
  if (score >= 0.9) return 'bg-emerald-500/20 text-emerald-400';
  if (score >= 0.8) return 'bg-cyan-500/20 text-cyan-400';
  if (score >= 0.7) return 'bg-amber-500/20 text-amber-400';
  return 'bg-slate-500/20 text-slate-400';
}

/** Data freshness color based on age in hours */
export function freshnessColor(hoursAgo: number): string {
  if (hoursAgo < 1) return 'text-emerald-400';
  if (hoursAgo < 24) return 'text-amber-400';
  return 'text-red-400';
}
