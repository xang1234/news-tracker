import { useState, useMemo } from 'react';
import { X, Upload, CheckCircle } from 'lucide-react';
import { SOURCE_PLATFORMS } from '@/lib/constants';
import {
  useBulkCreateSources,
  type BulkCreateSourcesResponse,
} from '@/api/hooks/useSources';

interface BulkAddSourcesModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const PLATFORM_OPTIONS = Object.entries(SOURCE_PLATFORMS).map(([value, info]) => ({
  value,
  label: info.label,
  placeholder: info.placeholder,
}));

function parseIdentifiers(raw: string): string[] {
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

export function BulkAddSourcesModal({ isOpen, onClose }: BulkAddSourcesModalProps) {
  const [platform, setPlatform] = useState('twitter');
  const [text, setText] = useState('');
  const [lastResult, setLastResult] = useState<BulkCreateSourcesResponse | null>(null);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const bulkMutation = useBulkCreateSources();

  const parsed = useMemo(() => parseIdentifiers(text), [text]);
  const unique = useMemo(() => [...new Set(parsed)], [parsed]);
  const duplicateCount = parsed.length - unique.length;

  if (!isOpen) return null;

  const platformInfo = PLATFORM_OPTIONS.find((p) => p.value === platform);

  function handleSubmit() {
    if (unique.length === 0) return;

    bulkMutation.mutate(
      { platform, identifiers: unique },
      {
        onSuccess: (data) => {
          setLastResult(data);
          setHasSubmitted(true);
          setText('');
        },
      },
    );
  }

  function handleClose() {
    setPlatform('twitter');
    setText('');
    setLastResult(null);
    setHasSubmitted(false);
    bulkMutation.reset();
    onClose();
  }

  function handleOverlayClick(e: React.MouseEvent) {
    if (e.target === e.currentTarget) handleClose();
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      role="dialog"
      aria-modal="true"
      aria-labelledby="bulk-add-title"
      onClick={handleOverlayClick}
    >
      <div className="w-full max-w-lg rounded-lg border border-border bg-card shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <h2 id="bulk-add-title" className="text-lg font-medium text-foreground">
            Bulk Add Sources
          </h2>
          <button
            type="button"
            onClick={handleClose}
            className="rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-4">
          {/* Success banner */}
          {lastResult && (
            <div className="mb-4 flex items-start gap-2 rounded border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-400">
              <CheckCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>
                Created {lastResult.created} source{lastResult.created !== 1 ? 's' : ''}
                {lastResult.skipped > 0 && `, ${lastResult.skipped} already existed`}
              </span>
            </div>
          )}

          {/* Error banner */}
          {bulkMutation.isError && (
            <div className="mb-4 rounded border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {bulkMutation.error instanceof Error
                ? bulkMutation.error.message
                : 'Failed to create sources'}
            </div>
          )}

          {/* Platform selector */}
          <div className="mb-4">
            <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
              Platform
            </label>
            <select
              value={platform}
              onChange={(e) => setPlatform(e.target.value)}
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              {PLATFORM_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Textarea */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
              Identifiers (one per line)
            </label>
            <textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setLastResult(null);
              }}
              placeholder={`Paste ${platformInfo?.placeholder ?? 'identifiers'} here, one per line...\n\nExample:\nSemiAnalysis\nAsianometry\nTechInsights`}
              rows={8}
              className="w-full resize-y rounded border border-border bg-background px-3 py-2 font-mono text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>

          {/* Preview line */}
          <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground">
            {unique.length > 0 && (
              <span>
                {unique.length} unique identifier{unique.length !== 1 ? 's' : ''}
              </span>
            )}
            {duplicateCount > 0 && (
              <span className="text-amber-400">
                {duplicateCount} duplicate{duplicateCount !== 1 ? 's' : ''} removed
              </span>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 border-t border-border px-6 py-4">
          <button
            type="button"
            onClick={handleClose}
            className="rounded border border-border px-4 py-1.5 text-sm text-muted-foreground hover:bg-secondary hover:text-foreground"
          >
            {hasSubmitted ? 'Done' : 'Cancel'}
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={unique.length === 0 || bulkMutation.isPending}
            className="flex items-center gap-1.5 rounded bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            <Upload className="h-3.5 w-3.5" />
            {bulkMutation.isPending
              ? 'Adding...'
              : unique.length > 0
                ? `Add ${unique.length} Source${unique.length !== 1 ? 's' : ''}`
                : 'Add Sources'}
          </button>
        </div>
      </div>
    </div>
  );
}
