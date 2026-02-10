import { cn } from '@/lib/utils';
import type { NEREntityItem } from '@/api/hooks/usePlayground';

const ENTITY_COLORS: Record<string, string> = {
  TICKER: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
  COMPANY: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  PRODUCT: 'bg-violet-500/20 text-violet-300 border-violet-500/30',
  TECHNOLOGY: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  METRIC: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
};

function entityColor(type: string): string {
  return ENTITY_COLORS[type] ?? 'bg-slate-500/20 text-slate-300 border-slate-500/30';
}

export const ENTITY_LEGEND = [
  { type: 'TICKER', label: 'Ticker', color: ENTITY_COLORS.TICKER },
  { type: 'COMPANY', label: 'Company', color: ENTITY_COLORS.COMPANY },
  { type: 'PRODUCT', label: 'Product', color: ENTITY_COLORS.PRODUCT },
  { type: 'TECHNOLOGY', label: 'Technology', color: ENTITY_COLORS.TECHNOLOGY },
  { type: 'METRIC', label: 'Metric', color: ENTITY_COLORS.METRIC },
];

export function EntityHighlighter({
  text,
  entities,
}: {
  text: string;
  entities: NEREntityItem[];
}) {
  if (!entities.length) {
    return <p className="text-sm text-muted-foreground">{text}</p>;
  }

  // Sort entities by start offset and remove overlaps (keep first)
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  const deduped: NEREntityItem[] = [];
  let lastEnd = -1;
  for (const e of sorted) {
    if (e.start >= lastEnd) {
      deduped.push(e);
      lastEnd = e.end;
    }
  }

  const parts: React.ReactNode[] = [];
  let cursor = 0;

  for (const entity of deduped) {
    // Text before entity
    if (entity.start > cursor) {
      parts.push(
        <span key={`t-${cursor}`}>{text.slice(cursor, entity.start)}</span>,
      );
    }

    // Entity span
    parts.push(
      <span
        key={`e-${entity.start}`}
        className={cn(
          'inline rounded border px-0.5 font-medium',
          entityColor(entity.type),
        )}
        title={`${entity.type}: ${entity.normalized} (${(entity.confidence * 100).toFixed(0)}%)`}
      >
        {text.slice(entity.start, entity.end)}
      </span>,
    );

    cursor = entity.end;
  }

  // Remaining text
  if (cursor < text.length) {
    parts.push(<span key={`t-${cursor}`}>{text.slice(cursor)}</span>);
  }

  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
      {parts}
    </p>
  );
}
