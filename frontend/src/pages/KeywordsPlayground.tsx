import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import {
  PlaygroundTextarea,
  PlaygroundActions,
  PlaygroundLatency,
  PlaygroundError,
} from '@/components/domain/PlaygroundShell';
import {
  useKeywordsMutation,
  type KeywordsResponse,
} from '@/api/hooks/usePlayground';
import { SAMPLE_KEYWORDS } from '@/lib/sampleTexts';

export default function KeywordsPlayground() {
  const [text, setText] = useState('');
  const [topN, setTopN] = useState(10);
  const mutation = useKeywordsMutation();

  const loadSample = () => setText(SAMPLE_KEYWORDS);

  const analyze = () => {
    if (!text.trim()) return;
    mutation.mutate({ texts: [text], top_n: topN });
  };

  const data: KeywordsResponse | undefined = mutation.data;
  const result = data?.results[0];
  const maxScore = result?.keywords[0]?.score ?? 1;

  return (
    <>
      <Header title="Keywords Playground" />
      <div className="space-y-6 p-6">
        <PlaygroundTextarea value={text} onChange={setText} rows={5} />

        {/* top_n selector */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">Top N:</span>
          <input
            type="number"
            min={1}
            max={50}
            value={topN}
            onChange={(e) => setTopN(Math.max(1, Math.min(50, Number(e.target.value))))}
            className="w-16 rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground"
          />
        </div>

        <div className="flex items-center gap-4">
          <PlaygroundActions
            onSample={loadSample}
            onAnalyze={analyze}
            isPending={mutation.isPending}
            analyzeDisabled={!text.trim()}
            analyzeLabel="Extract Keywords"
          />
          {data && <PlaygroundLatency ms={data.latency_ms} />}
        </div>

        <PlaygroundError error={mutation.error} />

        {result && (
          <div className="space-y-1">
            {result.keywords.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No keywords extracted from this text.
              </p>
            )}

            {result.keywords.map((kw, i) => {
              const pct = maxScore > 0 ? (kw.score / maxScore) * 100 : 0;
              return (
                <div key={i} className="flex items-center gap-3 py-1">
                  <span className="w-6 text-right text-xs font-medium text-muted-foreground">
                    {kw.rank}
                  </span>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-foreground">
                        {kw.text}
                      </span>
                      <span className="font-mono text-xs text-muted-foreground">
                        {kw.score.toFixed(4)}
                      </span>
                    </div>
                    <div className="mt-0.5 h-1.5 w-full rounded-full bg-slate-700/50">
                      <div
                        className="h-full rounded-full bg-primary/70"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                  {kw.count > 1 && (
                    <span className="text-xs text-muted-foreground">
                      x{kw.count}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
