import { useState } from 'react';
import { Plus, X } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import {
  PlaygroundTextarea,
  PlaygroundActions,
  PlaygroundLatency,
  PlaygroundError,
} from '@/components/domain/PlaygroundShell';
import {
  useSentimentMutation,
  type SentimentResponse,
  type SentimentResultItem,
} from '@/api/hooks/usePlayground';
import { SAMPLE_SENTIMENT } from '@/lib/sampleTexts';
import { SENTIMENT_COLORS } from '@/lib/constants';
import { cn } from '@/lib/utils';

function SentimentBar({ result }: { result: SentimentResultItem }) {
  const { positive, neutral, negative } = result.scores;
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'rounded-full px-2 py-0.5 text-xs font-medium',
            SENTIMENT_COLORS[result.label] ?? 'bg-slate-500/20 text-slate-400',
          )}
        >
          {result.label}
        </span>
        <span className="text-xs text-muted-foreground">
          {(result.confidence * 100).toFixed(1)}% confidence
        </span>
      </div>

      {/* Stacked horizontal bar */}
      <div className="flex h-5 w-full overflow-hidden rounded-md">
        {positive > 0 && (
          <div
            className="flex items-center justify-center bg-emerald-500/60 text-[10px] text-emerald-100"
            style={{ width: `${positive * 100}%` }}
          >
            {positive > 0.1 && `${(positive * 100).toFixed(0)}%`}
          </div>
        )}
        {neutral > 0 && (
          <div
            className="flex items-center justify-center bg-slate-500/40 text-[10px] text-slate-200"
            style={{ width: `${neutral * 100}%` }}
          >
            {neutral > 0.1 && `${(neutral * 100).toFixed(0)}%`}
          </div>
        )}
        {negative > 0 && (
          <div
            className="flex items-center justify-center bg-red-500/60 text-[10px] text-red-100"
            style={{ width: `${negative * 100}%` }}
          >
            {negative > 0.1 && `${(negative * 100).toFixed(0)}%`}
          </div>
        )}
      </div>

      <div className="flex gap-4 text-xs text-muted-foreground">
        <span>
          Positive:{' '}
          <span className="text-emerald-400">{(positive * 100).toFixed(1)}%</span>
        </span>
        <span>
          Neutral:{' '}
          <span className="text-slate-400">{(neutral * 100).toFixed(1)}%</span>
        </span>
        <span>
          Negative:{' '}
          <span className="text-red-400">{(negative * 100).toFixed(1)}%</span>
        </span>
      </div>
    </div>
  );
}

export default function SentimentPlayground() {
  const [texts, setTexts] = useState<string[]>(['']);
  const mutation = useSentimentMutation();

  const updateText = (idx: number, value: string) => {
    setTexts((prev) => prev.map((t, i) => (i === idx ? value : t)));
  };

  const addText = () => {
    if (texts.length < 8) setTexts((prev) => [...prev, '']);
  };

  const removeText = (idx: number) => {
    if (texts.length > 1) setTexts((prev) => prev.filter((_, i) => i !== idx));
  };

  const loadSample = () => {
    setTexts([SAMPLE_SENTIMENT]);
  };

  const analyze = () => {
    const nonEmpty = texts.filter((t) => t.trim().length > 0);
    if (nonEmpty.length === 0) return;
    mutation.mutate(nonEmpty);
  };

  const data: SentimentResponse | undefined = mutation.data;

  return (
    <>
      <Header title="Sentiment Playground" />
      <div className="space-y-6 p-6">
        {/* Text inputs */}
        <div className="space-y-3">
          {texts.map((text, idx) => (
            <div key={idx}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">
                  Text {idx + 1}
                </span>
                {texts.length > 1 && (
                  <button
                    onClick={() => removeText(idx)}
                    className="text-muted-foreground hover:text-foreground"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
              <PlaygroundTextarea value={text} onChange={(v) => updateText(idx, v)} rows={3} />
            </div>
          ))}

          {texts.length < 8 && (
            <button
              onClick={addText}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
            >
              <Plus className="h-3.5 w-3.5" /> Add text
            </button>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-4">
          <PlaygroundActions
            onSample={loadSample}
            onAnalyze={analyze}
            isPending={mutation.isPending}
            analyzeDisabled={texts.every((t) => !t.trim())}
          />
          {data && <PlaygroundLatency ms={data.latency_ms} />}
        </div>

        <PlaygroundError error={mutation.error} />

        {/* Results */}
        {data && (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Model: <span className="text-foreground">{data.model}</span>
            </p>

            {data.results.map((r, i) => {
              const inputText = texts.filter((t) => t.trim().length > 0)[i] ?? '';
              return (
                <div
                  key={i}
                  className="space-y-3 rounded-md border border-border bg-card/50 p-4"
                >
                  <p className="text-xs text-muted-foreground line-clamp-2">
                    {inputText}
                  </p>
                  <SentimentBar result={r} />
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
