import { formatDistanceToNow, format, parseISO } from 'date-fns';

/** Relative time: "3 minutes ago" */
export function timeAgo(isoDate: string | null | undefined): string {
  if (!isoDate) return '—';
  try {
    return formatDistanceToNow(parseISO(isoDate), { addSuffix: true });
  } catch {
    return isoDate;
  }
}

/** Absolute date: "2026-02-10 14:30" */
export function formatDate(isoDate: string | null | undefined): string {
  if (!isoDate) return '—';
  try {
    return format(parseISO(isoDate), 'yyyy-MM-dd HH:mm');
  } catch {
    return isoDate;
  }
}

/** Format a float as percentage: 0.873 → "87.3%" */
export function pct(value: number | null | undefined, decimals = 1): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format milliseconds as latency: 123.4 → "123ms" */
export function latency(ms: number | null | undefined): string {
  if (ms == null) return '—';
  if (ms < 1) return '<1ms';
  return `${Math.round(ms)}ms`;
}

/** Truncate text with ellipsis */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
}

/** Humanize a snake_case string: "adverse_drift" → "Adverse Drift" */
export function humanize(snakeCase: string): string {
  return snakeCase
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/**
 * Derive a human-readable trust verdict + badge styling from an assertion's
 * reconciliation state. Contradiction dominates; otherwise corroboration is
 * graded by support count and source diversity.
 */
export interface AssertionVerdict {
  label: string;
  className: string;
}

export function assertionVerdict(a: {
  status: string;
  support_count: number;
  contradiction_count: number;
  source_diversity: number;
}): AssertionVerdict {
  if (a.status === 'disputed') {
    return { label: 'Disputed', className: 'bg-red-500/20 text-red-400' };
  }
  if (a.contradiction_count > 0) {
    return { label: 'Contested', className: 'bg-amber-500/20 text-amber-400' };
  }
  if (a.support_count >= 2 && a.source_diversity >= 2) {
    return {
      label: `Corroborated · ${a.source_diversity} sources`,
      className: 'bg-emerald-500/20 text-emerald-400',
    };
  }
  if (a.support_count >= 2) {
    return { label: 'Supported', className: 'bg-emerald-500/20 text-emerald-300' };
  }
  return { label: 'Unverified', className: 'bg-slate-500/20 text-slate-400' };
}

/** Badge for how a briefing/answer was produced: LLM-synthesized vs templated fallback. */
export function generatedByBadge(method: 'llm' | 'template'): AssertionVerdict {
  return method === 'llm'
    ? { label: 'AI-generated', className: 'bg-violet-500/20 text-violet-300' }
    : { label: 'Templated', className: 'bg-cyan-500/20 text-cyan-300' };
}
