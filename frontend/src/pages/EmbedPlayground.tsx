import { useState } from 'react';
import { Plus, X } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import {
  PlaygroundTextarea,
  PlaygroundActions,
  PlaygroundLatency,
  PlaygroundError,
} from '@/components/domain/PlaygroundShell';
import { SimilarityMatrix } from '@/components/domain/SimilarityMatrix';
import { useEmbedMutation, type EmbedResponse } from '@/api/hooks/usePlayground';
import { similarityMatrix } from '@/lib/cosineSimilarity';
import { SAMPLE_EMBED_1, SAMPLE_EMBED_2 } from '@/lib/sampleTexts';
import { truncate } from '@/lib/formatters';
import { cn } from '@/lib/utils';

type ModelChoice = 'auto' | 'finbert' | 'minilm';

export default function EmbedPlayground() {
  const [texts, setTexts] = useState<string[]>(['', '']);
  const [model, setModel] = useState<ModelChoice>('auto');
  const mutation = useEmbedMutation();

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
    setTexts([SAMPLE_EMBED_1, SAMPLE_EMBED_2]);
  };

  const analyze = () => {
    const nonEmpty = texts.filter((t) => t.trim().length > 0);
    if (nonEmpty.length === 0) return;
    mutation.mutate({ texts: nonEmpty, model });
  };

  const data: EmbedResponse | undefined = mutation.data;
  const matrix =
    data && data.results.length > 1
      ? similarityMatrix(data.results.map((r) => r.embedding))
      : null;

  return (
    <>
      <Header title="Embed Playground" />
      <div className="space-y-6 p-6">
        {/* Model selector */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">Model:</span>
          {(['auto', 'finbert', 'minilm'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setModel(m)}
              className={cn(
                'rounded-md px-3 py-1 text-sm',
                model === m
                  ? 'bg-primary text-primary-foreground'
                  : 'border border-border text-muted-foreground hover:bg-secondary',
              )}
            >
              {m === 'auto' ? 'Auto' : m === 'finbert' ? 'FinBERT (768d)' : 'MiniLM (384d)'}
            </button>
          ))}
        </div>

        {/* Text inputs */}
        <div className="space-y-3">
          {texts.map((text, idx) => (
            <div key={idx} className="relative">
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
            analyzeLabel="Embed"
          />
          {data && <PlaygroundLatency ms={data.latency_ms} />}
        </div>

        <PlaygroundError error={mutation.error} />

        {/* Results */}
        {data && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
              <span>
                Model: <span className="text-foreground">{data.model}</span>
              </span>
              <span>
                Dimensions: <span className="text-foreground">{data.dimensions}</span>
              </span>
              <span>
                Vectors: <span className="text-foreground">{data.total}</span>
              </span>
            </div>

            {/* Per-text result cards */}
            <div className="space-y-2">
              {data.results.map((r, i) => (
                <div
                  key={i}
                  className="rounded-md border border-border bg-card/50 px-4 py-2 text-xs"
                >
                  <span className="text-muted-foreground">
                    [{i + 1}] {r.dimensions}d
                    {r.cached && (
                      <span className="ml-2 rounded bg-slate-500/20 px-1.5 py-0.5 text-slate-400">
                        cached
                      </span>
                    )}
                    {' — '}
                    {truncate(
                      texts.filter((t) => t.trim().length > 0)[i] ?? '',
                      60,
                    )}
                  </span>
                </div>
              ))}
            </div>

            {/* Similarity matrix */}
            {matrix && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-foreground">
                  Pairwise Cosine Similarity
                </h3>
                <SimilarityMatrix
                  labels={texts
                    .filter((t) => t.trim().length > 0)
                    .map((t, i) => `Text ${i + 1}`)}
                  matrix={matrix}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}
