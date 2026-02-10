import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import {
  PlaygroundTextarea,
  PlaygroundActions,
  PlaygroundLatency,
  PlaygroundError,
} from '@/components/domain/PlaygroundShell';
import {
  EntityHighlighter,
  ENTITY_LEGEND,
} from '@/components/domain/EntityHighlighter';
import { useNERMutation, type NERResponse } from '@/api/hooks/usePlayground';
import { SAMPLE_NER } from '@/lib/sampleTexts';
import { cn } from '@/lib/utils';

export default function NERPlayground() {
  const [text, setText] = useState('');
  const mutation = useNERMutation();

  const loadSample = () => setText(SAMPLE_NER);

  const analyze = () => {
    if (!text.trim()) return;
    mutation.mutate([text]);
  };

  const data: NERResponse | undefined = mutation.data;
  const result = data?.results[0];
  // Use the text that was actually analyzed (in case user edits after analyzing)
  const analyzedText = text;

  return (
    <>
      <Header title="NER Playground" />
      <div className="space-y-6 p-6">
        <PlaygroundTextarea value={text} onChange={setText} rows={5} />

        <div className="flex items-center gap-4">
          <PlaygroundActions
            onSample={loadSample}
            onAnalyze={analyze}
            isPending={mutation.isPending}
            analyzeDisabled={!text.trim()}
            analyzeLabel="Extract Entities"
          />
          {data && <PlaygroundLatency ms={data.latency_ms} />}
        </div>

        <PlaygroundError error={mutation.error} />

        {result && (
          <div className="space-y-6">
            {/* Entity legend */}
            <div className="flex flex-wrap gap-2">
              {ENTITY_LEGEND.map(({ type, label, color }) => (
                <span
                  key={type}
                  className={cn('rounded border px-2 py-0.5 text-xs', color)}
                >
                  {label}
                </span>
              ))}
            </div>

            {/* Highlighted text */}
            <div className="rounded-md border border-border bg-card/50 p-4">
              <EntityHighlighter text={analyzedText} entities={result.entities} />
            </div>

            {/* Entity table */}
            {result.entities.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left text-xs text-muted-foreground">
                      <th className="px-3 py-2">Text</th>
                      <th className="px-3 py-2">Type</th>
                      <th className="px-3 py-2">Normalized</th>
                      <th className="px-3 py-2">Confidence</th>
                      <th className="px-3 py-2">Offset</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.entities.map((e, i) => (
                      <tr key={i} className="border-b border-border/50">
                        <td className="px-3 py-2 font-medium text-foreground">
                          {e.text}
                        </td>
                        <td className="px-3 py-2">
                          <span className="rounded bg-slate-500/20 px-1.5 py-0.5 text-xs text-slate-300">
                            {e.type}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-muted-foreground">
                          {e.normalized}
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                          {(e.confidence * 100).toFixed(0)}%
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                          {e.start}:{e.end}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {result.entities.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No entities found in this text.
              </p>
            )}
          </div>
        )}
      </div>
    </>
  );
}
