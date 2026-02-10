import { cn } from '@/lib/utils';
import { truncate } from '@/lib/formatters';

function cellColor(score: number): string {
  if (score >= 0.95) return 'bg-emerald-500/30 text-emerald-300';
  if (score >= 0.85) return 'bg-emerald-500/20 text-emerald-400';
  if (score >= 0.7) return 'bg-cyan-500/20 text-cyan-400';
  if (score >= 0.5) return 'bg-amber-500/20 text-amber-400';
  return 'bg-slate-500/10 text-slate-400';
}

export function SimilarityMatrix({
  labels,
  matrix,
}: {
  labels: string[];
  matrix: number[][];
}) {
  if (labels.length === 0) return null;

  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="px-2 py-1" />
            {labels.map((label, i) => (
              <th
                key={i}
                className="max-w-[120px] truncate px-2 py-1 text-left font-medium text-muted-foreground"
                title={label}
              >
                {truncate(label, 20)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td
                className="max-w-[120px] truncate px-2 py-1 font-medium text-muted-foreground"
                title={labels[i]}
              >
                {truncate(labels[i], 20)}
              </td>
              {row.map((score, j) => (
                <td
                  key={j}
                  className={cn(
                    'px-3 py-1.5 text-center font-mono tabular-nums rounded',
                    i === j ? 'bg-slate-500/5 text-slate-500' : cellColor(score),
                  )}
                >
                  {score.toFixed(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
