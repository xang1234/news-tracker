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
