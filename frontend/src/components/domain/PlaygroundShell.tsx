import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { latency as fmtLatency } from '@/lib/formatters';

/** Styled textarea with character count */
export function PlaygroundTextarea({
  value,
  onChange,
  placeholder = 'Enter text to analyze...',
  maxLength = 5000,
  rows = 4,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  maxLength?: number;
  rows?: number;
}) {
  return (
    <div className="relative">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        maxLength={maxLength}
        rows={rows}
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
      />
      <span className="absolute bottom-2 right-2 text-xs text-muted-foreground">
        {value.length}/{maxLength}
      </span>
    </div>
  );
}

/** Action buttons: Load Sample + Analyze */
export function PlaygroundActions({
  onSample,
  onAnalyze,
  isPending,
  analyzeDisabled,
  analyzeLabel = 'Analyze',
}: {
  onSample: () => void;
  onAnalyze: () => void;
  isPending: boolean;
  analyzeDisabled?: boolean;
  analyzeLabel?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={onSample}
        className="rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-secondary hover:text-foreground"
      >
        Load Sample
      </button>
      <button
        onClick={onAnalyze}
        disabled={isPending || analyzeDisabled}
        className={cn(
          'flex items-center gap-2 rounded-md px-4 py-1.5 text-sm font-medium',
          isPending || analyzeDisabled
            ? 'cursor-not-allowed bg-primary/50 text-primary-foreground/50'
            : 'bg-primary text-primary-foreground hover:bg-primary/90',
        )}
      >
        {isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
        {isPending ? 'Processing...' : analyzeLabel}
      </button>
    </div>
  );
}

/** Latency badge */
export function PlaygroundLatency({ ms }: { ms: number | undefined }) {
  if (ms == null) return null;
  return (
    <span className="inline-flex items-center rounded-full bg-cyan-500/10 px-2.5 py-0.5 text-xs font-medium text-cyan-400">
      {fmtLatency(ms)}
    </span>
  );
}

/** Error banner */
export function PlaygroundError({ error }: { error: Error | null }) {
  if (!error) return null;
  return (
    <div className="rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
      {error.message}
    </div>
  );
}
