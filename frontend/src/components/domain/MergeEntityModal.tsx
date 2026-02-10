import { useState } from 'react';
import { X, GitMerge } from 'lucide-react';
import { cn } from '@/lib/utils';

const ENTITY_TYPES = ['COMPANY', 'PRODUCT', 'TECHNOLOGY', 'TICKER', 'METRIC'];

interface MergeEntityModalProps {
  isOpen: boolean;
  onClose: () => void;
  fromType: string;
  fromNormalized: string;
  onMerge: (toType: string, toNormalized: string) => void;
}

export function MergeEntityModal({
  isOpen,
  onClose,
  fromType,
  fromNormalized,
  onMerge,
}: MergeEntityModalProps) {
  const [toType, setToType] = useState(fromType);
  const [toNormalized, setToNormalized] = useState('');

  if (!isOpen) return null;

  const canSubmit = toNormalized.trim().length > 0;

  function handleSubmit() {
    if (!canSubmit) return;
    onMerge(toType, toNormalized.trim());
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Escape') {
      onClose();
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
      onKeyDown={handleKeyDown}
      role="dialog"
      aria-modal="true"
      aria-label="Merge entity"
    >
      <div
        className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-foreground">
            <GitMerge className="h-5 w-5 text-muted-foreground" />
            <h2 className="text-lg font-semibold">Merge Entity</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* From entity display */}
        <div className="mt-4 rounded-lg border border-border bg-background p-3">
          <div className="text-xs text-muted-foreground">Merging from</div>
          <div className="mt-1 flex items-center gap-2">
            <span className="rounded bg-secondary px-1.5 py-0.5 text-xs font-medium text-muted-foreground">
              {fromType}
            </span>
            <span className="font-medium text-foreground">{fromNormalized}</span>
          </div>
        </div>

        {/* Target entity type */}
        <div className="mt-4">
          <label className="mb-2 block text-xs font-medium text-muted-foreground">
            Target Type
          </label>
          <select
            value={toType}
            onChange={(e) => setToType(e.target.value)}
            className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          >
            {ENTITY_TYPES.map((t) => (
              <option key={t} value={t}>
                {t.charAt(0) + t.slice(1).toLowerCase()}
              </option>
            ))}
          </select>
        </div>

        {/* Target normalized name */}
        <div className="mt-4">
          <label className="mb-2 block text-xs font-medium text-muted-foreground">
            Target Name
          </label>
          <input
            type="text"
            value={toNormalized}
            onChange={(e) => setToNormalized(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSubmit(); }}
            placeholder="Canonical entity name..."
            className={cn(
              'w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground',
              'placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring',
            )}
            autoFocus
          />
        </div>

        {/* Actions */}
        <div className="mt-6 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded border border-border px-4 py-1.5 text-sm text-muted-foreground hover:bg-secondary hover:text-foreground"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={!canSubmit}
            onClick={handleSubmit}
            className={cn(
              'flex items-center gap-1.5 rounded px-4 py-1.5 text-sm font-medium',
              'bg-primary text-primary-foreground hover:bg-primary/90',
              'disabled:cursor-not-allowed disabled:opacity-40',
            )}
          >
            <GitMerge className="h-3.5 w-3.5" />
            Merge
          </button>
        </div>
      </div>
    </div>
  );
}
