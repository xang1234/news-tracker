import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { SOURCE_PLATFORMS } from '@/lib/constants';

export interface SourceFormData {
  platform: string;
  identifier: string;
  display_name: string;
  description: string;
}

interface SourceFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: SourceFormData) => void;
  initialValues?: SourceFormData;
  mode: 'create' | 'edit';
  isPending?: boolean;
}

const EMPTY_FORM: SourceFormData = {
  platform: 'twitter',
  identifier: '',
  display_name: '',
  description: '',
};

const PLATFORM_OPTIONS = Object.entries(SOURCE_PLATFORMS).map(([value, info]) => ({
  value,
  label: info.label,
}));

export function SourceFormModal({
  isOpen,
  onClose,
  onSubmit,
  initialValues,
  mode,
  isPending = false,
}: SourceFormModalProps) {
  const [form, setForm] = useState<SourceFormData>(EMPTY_FORM);

  useEffect(() => {
    if (isOpen) {
      setForm(initialValues ?? EMPTY_FORM);
    }
  }, [isOpen, initialValues]);

  if (!isOpen) {
    return null;
  }

  const isEdit = mode === 'edit';
  const title = isEdit ? 'Edit Source' : 'Add Source';
  const platformInfo = SOURCE_PLATFORMS[form.platform];

  function handleFieldChange(field: keyof SourceFormData, value: string): void {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function handleSubmit(e: React.FormEvent): void {
    e.preventDefault();
    onSubmit(form);
  }

  function handleOverlayClick(e: React.MouseEvent): void {
    if (e.target === e.currentTarget) {
      onClose();
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      role="dialog"
      aria-modal="true"
      aria-labelledby="source-form-title"
      onClick={handleOverlayClick}
    >
      <div className="w-full max-w-lg rounded-lg border border-border bg-card shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <h2 id="source-form-title" className="text-lg font-medium text-foreground">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="px-6 py-4">
          <div className="grid grid-cols-2 gap-4">
            {/* Platform */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Platform
              </label>
              <select
                value={form.platform}
                onChange={(e) => handleFieldChange('platform', e.target.value)}
                disabled={isEdit}
                className={cn(
                  'w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring',
                  isEdit && 'cursor-not-allowed opacity-60',
                )}
              >
                {PLATFORM_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Identifier */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Identifier
              </label>
              <input
                type="text"
                value={form.identifier}
                onChange={(e) => handleFieldChange('identifier', e.target.value)}
                readOnly={isEdit}
                placeholder={platformInfo?.placeholder ?? 'Identifier'}
                className={cn(
                  'w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring',
                  isEdit && 'cursor-not-allowed opacity-60',
                )}
              />
            </div>

            {/* Display Name (full width) */}
            <div className="col-span-2">
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Display Name
              </label>
              <input
                type="text"
                value={form.display_name}
                onChange={(e) => handleFieldChange('display_name', e.target.value)}
                placeholder="e.g. SemiAnalysis"
                className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>

            {/* Description (full width) */}
            <div className="col-span-2">
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Description
              </label>
              <input
                type="text"
                value={form.description}
                onChange={(e) => handleFieldChange('description', e.target.value)}
                placeholder="Short description of this source"
                className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="mt-6 flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded border border-border px-4 py-1.5 text-sm text-muted-foreground hover:bg-secondary hover:text-foreground"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isPending}
              className="rounded bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              {isPending ? 'Saving...' : isEdit ? 'Save Changes' : 'Add Source'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
