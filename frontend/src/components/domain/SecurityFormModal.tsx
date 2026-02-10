import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface SecurityFormData {
  ticker: string;
  exchange: string;
  name: string;
  aliases: string[];
  sector: string;
  country: string;
  currency: string;
}

interface SecurityFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: SecurityFormData) => void;
  initialValues?: SecurityFormData;
  mode: 'create' | 'edit';
  isPending?: boolean;
}

const EMPTY_FORM: SecurityFormData = {
  ticker: '',
  exchange: '',
  name: '',
  aliases: [],
  sector: '',
  country: '',
  currency: '',
};

export function SecurityFormModal({
  isOpen,
  onClose,
  onSubmit,
  initialValues,
  mode,
  isPending = false,
}: SecurityFormModalProps) {
  const [form, setForm] = useState<SecurityFormData>(EMPTY_FORM);
  const [aliasText, setAliasText] = useState('');

  useEffect(() => {
    if (isOpen) {
      const values = initialValues ?? EMPTY_FORM;
      setForm(values);
      setAliasText(values.aliases.join(', '));
    }
  }, [isOpen, initialValues]);

  if (!isOpen) {
    return null;
  }

  const isEdit = mode === 'edit';
  const title = isEdit ? 'Edit Security' : 'Create Security';

  function handleFieldChange(field: keyof SecurityFormData, value: string): void {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function handleAliasChange(value: string): void {
    setAliasText(value);
    const parsed = value
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    setForm((prev) => ({ ...prev, aliases: parsed }));
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
      aria-labelledby="security-form-title"
      onClick={handleOverlayClick}
    >
      <div className="w-full max-w-lg rounded-lg border border-border bg-card shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <h2 id="security-form-title" className="text-lg font-medium text-foreground">{title}</h2>
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
            {/* Ticker */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Ticker
              </label>
              <input
                type="text"
                value={form.ticker}
                onChange={(e) => handleFieldChange('ticker', e.target.value.toUpperCase())}
                readOnly={isEdit}
                placeholder="e.g. NVDA"
                className={cn(
                  'w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring',
                  isEdit && 'cursor-not-allowed opacity-60',
                )}
              />
            </div>

            {/* Exchange */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Exchange
              </label>
              <input
                type="text"
                value={form.exchange}
                onChange={(e) => handleFieldChange('exchange', e.target.value.toUpperCase())}
                readOnly={isEdit}
                placeholder="e.g. US"
                className={cn(
                  'w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring',
                  isEdit && 'cursor-not-allowed opacity-60',
                )}
              />
            </div>

            {/* Name (full width) */}
            <div className="col-span-2">
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Name
              </label>
              <input
                type="text"
                value={form.name}
                onChange={(e) => handleFieldChange('name', e.target.value)}
                placeholder="e.g. NVIDIA Corporation"
                className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>

            {/* Aliases (full width) */}
            <div className="col-span-2">
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Aliases (comma-separated)
              </label>
              <input
                type="text"
                value={aliasText}
                onChange={(e) => handleAliasChange(e.target.value)}
                placeholder="e.g. Nvidia, NVDA Corp"
                className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>

            {/* Sector */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Sector
              </label>
              <input
                type="text"
                value={form.sector}
                onChange={(e) => handleFieldChange('sector', e.target.value)}
                placeholder="e.g. Semiconductors"
                className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>

            {/* Country */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Country
              </label>
              <input
                type="text"
                value={form.country}
                onChange={(e) => handleFieldChange('country', e.target.value)}
                placeholder="e.g. US"
                className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>

            {/* Currency */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                Currency
              </label>
              <input
                type="text"
                value={form.currency}
                onChange={(e) => handleFieldChange('currency', e.target.value.toUpperCase())}
                placeholder="e.g. USD"
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
              {isPending ? 'Saving...' : isEdit ? 'Save Changes' : 'Create Security'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
