import { Pencil, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { SOURCE_PLATFORMS } from '@/lib/constants';

export interface SourceRow {
  platform: string;
  identifier: string;
  display_name: string;
  description: string;
  is_active: boolean;
  created_at: string | null;
  updated_at: string | null;
}

interface SourcesTableProps {
  sources: SourceRow[];
  onEdit: (platform: string, identifier: string) => void;
  onDeactivate: (platform: string, identifier: string) => void;
}

const COLUMNS = ['Platform', 'Identifier', 'Display Name', 'Active', 'Actions'] as const;

function formatIdentifier(platform: string, identifier: string): string {
  if (platform === 'twitter') return `@${identifier}`;
  if (platform === 'reddit') return `r/${identifier}`;
  return identifier;
}

export function SourcesTable({ sources, onEdit, onDeactivate }: SourcesTableProps) {
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
          {sources.length === 0 && (
            <tr>
              <td
                colSpan={COLUMNS.length}
                className="px-4 py-8 text-center text-sm text-muted-foreground"
              >
                No sources found.
              </td>
            </tr>
          )}
          {sources.map((src) => {
            const platformInfo = SOURCE_PLATFORMS[src.platform];
            return (
              <tr
                key={`${src.platform}-${src.identifier}`}
                className="border-b border-border bg-card transition-colors hover:bg-secondary/30"
              >
                <td className="px-4 py-3">
                  <span
                    className={cn(
                      'inline-block rounded-full px-2 py-0.5 text-xs font-medium',
                      platformInfo?.color ?? 'bg-slate-500/20 text-slate-400',
                    )}
                  >
                    {platformInfo?.label ?? src.platform}
                  </span>
                </td>
                <td className="px-4 py-3 font-mono text-foreground">
                  {formatIdentifier(src.platform, src.identifier)}
                </td>
                <td className="px-4 py-3 text-foreground">
                  <div>{src.display_name || '---'}</div>
                  {src.description && (
                    <div className="mt-0.5 text-xs text-muted-foreground">
                      {src.description}
                    </div>
                  )}
                </td>
                <td className="px-4 py-3">
                  <span
                    className={cn(
                      'inline-block rounded-full px-2 py-0.5 text-xs font-medium',
                      src.is_active
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : 'bg-red-500/20 text-red-400',
                    )}
                  >
                    {src.is_active ? 'Active' : 'Inactive'}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => onEdit(src.platform, src.identifier)}
                      className="rounded p-1.5 text-muted-foreground hover:bg-secondary hover:text-foreground"
                      title="Edit source"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    {src.is_active && (
                      <button
                        type="button"
                        onClick={() => onDeactivate(src.platform, src.identifier)}
                        className="rounded p-1.5 text-muted-foreground hover:bg-red-500/20 hover:text-red-400"
                        title="Deactivate source"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const SKELETON_ROWS = 5;

export function SourcesTableSkeleton() {
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
                <div className="h-5 w-16 animate-pulse rounded-full bg-secondary" />
              </td>
              <td className="px-4 py-3">
                <div className="h-4 w-28 animate-pulse rounded bg-secondary" />
              </td>
              <td className="px-4 py-3">
                <div className="h-4 w-36 animate-pulse rounded bg-secondary" />
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
