import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Search,
  FileText,
  FlaskConical,
  Activity,
  Tag,
  Hash,
  Zap,
  Layers,
  Bell,
  GitBranch,
  Settings,
  ChevronLeft,
  ChevronRight,
  Cpu,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useUIStore } from '@/stores/uiStore';

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/search', icon: Search, label: 'Search' },
  { to: '/documents', icon: FileText, label: 'Documents' },
  { to: '/playground/embed', icon: FlaskConical, label: 'Embed' },
  { to: '/playground/sentiment', icon: Activity, label: 'Sentiment' },
  { to: '/playground/ner', icon: Tag, label: 'NER' },
  { to: '/playground/keywords', icon: Hash, label: 'Keywords' },
  { to: '/playground/events', icon: Zap, label: 'Events' },
  { to: '/monitoring', icon: Activity, label: 'Monitoring' },
  { to: '/themes', icon: Layers, label: 'Themes' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { to: '/graph', icon: GitBranch, label: 'Graph' },
  { to: '/settings', icon: Settings, label: 'Settings' },
] as const;

export function Sidebar() {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);

  return (
    <aside
      className={cn(
        'flex h-screen flex-col border-r border-border bg-card transition-all duration-200',
        sidebarOpen ? 'w-56' : 'w-14',
      )}
    >
      {/* Logo */}
      <div className="flex h-14 items-center gap-2 border-b border-border px-3">
        <Cpu className="h-6 w-6 shrink-0 text-primary" />
        {sidebarOpen && (
          <span className="text-sm font-semibold tracking-tight text-foreground">
            News Tracker
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'mx-2 my-0.5 flex items-center gap-3 rounded-md px-2 py-1.5 text-sm transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground',
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" />
            {sidebarOpen && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={toggleSidebar}
        className="flex h-10 items-center justify-center border-t border-border text-muted-foreground hover:text-foreground"
      >
        {sidebarOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </button>
    </aside>
  );
}
