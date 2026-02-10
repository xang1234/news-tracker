import { Pencil, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface Security {
  ticker: string;
  exchange: string;
  name: string;
  aliases: string[];
  sector: string;
  country: string;
  currency: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface SecuritiesTableProps {
  securities: Security[];
  onEdit: (ticker: string, exchange: string) => void;
  onDeactivate: (ticker: string, exchange: string) => void;
}

const COLUMNS = ['Ticker', 'Exchange', 'Name', 'Sector', 'Active', 'Actions'] as const;

export function SecuritiesTable({ securities, onEdit, onDeactivate }: SecuritiesTableProps) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-secondary/50">
            {COLUMNS.map((col) => (
              <th
                key={col}
                className="px-4 py-3 text-left text-xs font-medium text-muted-foreground"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {securities.length === 0 && (
            <tr>
              <td
                colSpan={COLUMNS.length}
                className="px-4 py-8 text-center text-sm text-muted-foreground"
              >
                No securities found.
              </td>
            </tr>
          )}
          {securities.map((sec) => (
            <tr
              key={`${sec.ticker}-${sec.exchange}`}
              className="border-b border-border bg-card transition-colors hover:bg-secondary/30"
            >
              <td className="px-4 py-3 font-mono text-foreground">{sec.ticker}</td>
              <td className="px-4 py-3 text-muted-foreground">{sec.exchange}</td>
              <td className="px-4 py-3 text-foreground">
                <div>{sec.name}</div>
                {sec.aliases.length > 0 && (
                  <div className="mt-0.5 text-xs text-muted-foreground">
                    {sec.aliases.join(', ')}
                  </div>
                )}
              </td>
              <td className="px-4 py-3 text-muted-foreground">{sec.sector || '---'}</td>
              <td className="px-4 py-3">
                <span
                  className={cn(
                    'inline-block rounded-full px-2 py-0.5 text-xs font-medium',
                    sec.is_active
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : 'bg-red-500/20 text-red-400',
                  )}
                >
                  {sec.is_active ? 'Active' : 'Inactive'}
                </span>
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onEdit(sec.ticker, sec.exchange)}
                    className="rounded p-1.5 text-muted-foreground hover:bg-secondary hover:text-foreground"
                    title="Edit security"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </button>
                  {sec.is_active && (
                    <button
                      type="button"
                      onClick={() => onDeactivate(sec.ticker, sec.exchange)}
                      className="rounded p-1.5 text-muted-foreground hover:bg-red-500/20 hover:text-red-400"
                      title="Deactivate security"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const SKELETON_ROWS = 5;

export function SecuritiesTableSkeleton() {
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-secondary/50">
            {COLUMNS.map((col) => (
              <th
                key={col}
                className="px-4 py-3 text-left text-xs font-medium text-muted-foreground"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: SKELETON_ROWS }, (_, i) => (
            <tr key={i} className="border-b border-border bg-card">
              <td className="px-4 py-3">
                <div className="h-4 w-14 animate-pulse rounded bg-secondary" />
              </td>
              <td className="px-4 py-3">
                <div className="h-4 w-10 animate-pulse rounded bg-secondary" />
              </td>
              <td className="px-4 py-3">
                <div className="h-4 w-40 animate-pulse rounded bg-secondary" />
              </td>
              <td className="px-4 py-3">
                <div className="h-4 w-24 animate-pulse rounded bg-secondary" />
              </td>
              <td className="px-4 py-3">
                <div className="h-5 w-14 animate-pulse rounded-full bg-secondary" />
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="h-6 w-6 animate-pulse rounded bg-secondary" />
                  <div className="h-6 w-6 animate-pulse rounded bg-secondary" />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
