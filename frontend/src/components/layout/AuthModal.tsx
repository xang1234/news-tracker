import { useState } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { Key, X } from 'lucide-react';

export function AuthModal() {
  const showAuthModal = useAuthStore((s) => s.showAuthModal);
  const setShowAuthModal = useAuthStore((s) => s.setShowAuthModal);
  const setApiKey = useAuthStore((s) => s.setApiKey);
  const clearApiKey = useAuthStore((s) => s.clearApiKey);
  const currentKey = useAuthStore((s) => s.apiKey);
  const [input, setInput] = useState('');

  if (!showAuthModal) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      setApiKey(input.trim());
      setInput('');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-6">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Key className="h-5 w-5 text-primary" />
            <h2 className="text-sm font-semibold text-foreground">API Key</h2>
          </div>
          <button
            onClick={() => setShowAuthModal(false)}
            className="text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {currentKey && (
          <div className="mb-3 flex items-center justify-between rounded bg-secondary px-3 py-2 text-xs">
            <span className="text-muted-foreground">
              Current: {currentKey.slice(0, 8)}...{currentKey.slice(-4)}
            </span>
            <button onClick={clearApiKey} className="text-destructive hover:text-destructive/80">
              Remove
            </button>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <input
            type="password"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Paste your API key..."
            className="w-full rounded border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"
            autoFocus
          />
          <div className="mt-3 flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowAuthModal(false)}
              className="rounded px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!input.trim()}
              className="rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
            >
              Save
            </button>
          </div>
        </form>

        <p className="mt-3 text-xs text-muted-foreground">
          Stored in session only — clears when you close this tab.
        </p>
      </div>
    </div>
  );
}
