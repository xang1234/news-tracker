import { Moon, Sun, Key } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';
import { useUIStore } from '@/stores/uiStore';
import { cn } from '@/lib/utils';

export function Header({ title }: { title?: string }) {
  const darkMode = useUIStore((s) => s.darkMode);
  const toggleDarkMode = useUIStore((s) => s.toggleDarkMode);
  const apiKey = useAuthStore((s) => s.apiKey);
  const setShowAuthModal = useAuthStore((s) => s.setShowAuthModal);

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card px-4">
      <h1 className="text-sm font-medium text-foreground">{title ?? 'Dashboard'}</h1>

      <div className="flex items-center gap-3">
        {/* Auth indicator */}
        <button
          onClick={() => setShowAuthModal(true)}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
        >
          <span
            className={cn('h-2 w-2 rounded-full', apiKey ? 'bg-emerald-400' : 'bg-slate-600')}
          />
          <Key className="h-3.5 w-3.5" />
        </button>

        {/* Dark mode toggle */}
        <button
          onClick={toggleDarkMode}
          className="text-muted-foreground hover:text-foreground"
          aria-label="Toggle dark mode"
        >
          {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </div>
    </header>
  );
}
