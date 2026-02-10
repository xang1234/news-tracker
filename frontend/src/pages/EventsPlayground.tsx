import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import {
  PlaygroundTextarea,
  PlaygroundActions,
  PlaygroundLatency,
  PlaygroundError,
} from '@/components/domain/PlaygroundShell';
import {
  useEventsExtractMutation,
  type EventsExtractResponse,
  type ExtractedEventItem,
} from '@/api/hooks/usePlayground';
import { SAMPLE_EVENTS } from '@/lib/sampleTexts';
import { cn } from '@/lib/utils';

const EVENT_TYPE_COLORS: Record<string, string> = {
  capacity_expansion: 'bg-emerald-500/20 text-emerald-400',
  capacity_constraint: 'bg-red-500/20 text-red-400',
  product_launch: 'bg-sky-500/20 text-sky-400',
  product_delay: 'bg-amber-500/20 text-amber-400',
  price_change: 'bg-violet-500/20 text-violet-400',
  guidance_change: 'bg-cyan-500/20 text-cyan-400',
};

function EventCard({ event }: { event: ExtractedEventItem }) {
  return (
    <div className="rounded-md border border-border bg-card/50 p-4 space-y-3">
      {/* Header: type badge + confidence */}
      <div className="flex items-center justify-between">
        <span
          className={cn(
            'rounded-full px-2.5 py-0.5 text-xs font-medium',
            EVENT_TYPE_COLORS[event.event_type] ?? 'bg-slate-500/20 text-slate-400',
          )}
        >
          {event.event_type.replace(/_/g, ' ')}
        </span>
        <span className="font-mono text-xs text-muted-foreground">
          {(event.confidence * 100).toFixed(0)}%
        </span>
      </div>

      {/* SVO structure */}
      <div className="flex flex-wrap items-baseline gap-1 text-sm">
        {event.actor && (
          <span className="font-semibold text-foreground">{event.actor}</span>
        )}
        <span className="text-primary">{event.action}</span>
        {event.object && (
          <span className="text-foreground">{event.object}</span>
        )}
      </div>

      {/* Metadata badges */}
      <div className="flex flex-wrap gap-2">
        {event.time_ref && (
          <span className="rounded bg-slate-500/20 px-2 py-0.5 text-xs text-slate-300">
            {event.time_ref}
          </span>
        )}
        {event.quantity && (
          <span className="rounded bg-amber-500/20 px-2 py-0.5 text-xs text-amber-300">
            {event.quantity}
          </span>
        )}
        {event.tickers.map((t) => (
          <span
            key={t}
            className="rounded bg-sky-500/20 px-2 py-0.5 text-xs text-sky-300"
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function EventsPlayground() {
  const [text, setText] = useState('');
  const [tickersInput, setTickersInput] = useState('');
  const mutation = useEventsExtractMutation();

  const loadSample = () => {
    setText(SAMPLE_EVENTS);
    setTickersInput('TSM, INTC, ASML');
  };

  const analyze = () => {
    if (!text.trim()) return;
    const tickers = tickersInput
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);
    mutation.mutate({ text, tickers: tickers.length > 0 ? tickers : undefined });
  };

  const data: EventsExtractResponse | undefined = mutation.data;

  return (
    <>
      <Header title="Events Playground" />
      <div className="space-y-6 p-6">
        <PlaygroundTextarea value={text} onChange={setText} rows={5} />

        {/* Tickers input */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">
            Tickers (optional, comma-separated)
          </label>
          <input
            value={tickersInput}
            onChange={(e) => setTickersInput(e.target.value)}
            placeholder="TSM, INTC, NVDA"
            className="w-full max-w-md rounded-md border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        <div className="flex items-center gap-4">
          <PlaygroundActions
            onSample={loadSample}
            onAnalyze={analyze}
            isPending={mutation.isPending}
            analyzeDisabled={!text.trim()}
            analyzeLabel="Extract Events"
          />
          {data && <PlaygroundLatency ms={data.latency_ms} />}
        </div>

        <PlaygroundError error={mutation.error} />

        {data && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Found <span className="text-foreground">{data.total}</span> event
              {data.total !== 1 && 's'}
            </p>

            {data.events.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No structured events found in this text. Try financial news about
                capacity, products, pricing, or guidance changes.
              </p>
            )}

            <div className="grid gap-3 sm:grid-cols-2">
              {data.events.map((ev, i) => (
                <EventCard key={i} event={ev} />
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
