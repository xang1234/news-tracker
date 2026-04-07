import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Search,
  FileText,
  FlaskConical,
  Tag,
  Layers,
  Bell,
  GitBranch,
  Fingerprint,
  Settings,
  Target,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Eye,
  AlertTriangle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useUIStore } from '@/stores/uiStore';

interface NavGroup {
  label: string;
  items: readonly { to: string; icon: typeof LayoutDashboard; label: string }[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    label: 'INTELLIGENCE',
    items: [
      { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
      { to: '/catalysts', icon: Target, label: 'Catalysts' },
      { to: '/evidence', icon: Eye, label: 'Evidence' },
      { to: '/divergence', icon: AlertTriangle, label: 'Divergence' },
    ],
  },
  {
    label: 'EXPLORE',
    items: [
      { to: '/search', icon: Search, label: 'Search' },
      { to: '/documents', icon: FileText, label: 'Documents' },
      { to: '/themes', icon: Layers, label: 'Themes' },
      { to: '/entities', icon: Fingerprint, label: 'Entities' },
      { to: '/graph', icon: GitBranch, label: 'Graph' },
    ],
  },
  {
    label: 'OPERATIONS',
    items: [
      { to: '/alerts', icon: Bell, label: 'Alerts' },
      { to: '/settings', icon: Settings, label: 'Settings' },
    ],
  },
  {
    label: 'DEV TOOLS',
    items: [
      { to: '/playground', icon: FlaskConical, label: 'Playground' },
    ],
  },
];

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
        {NAV_GROUPS.map((group) => (
          <div key={group.label} className="mt-2 first:mt-0">
            {sidebarOpen && (
              <div className="mx-3 mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60">
                {group.label}
              </div>
            )}
            {group.items.map(({ to, icon: Icon, label }) => (
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
          </div>
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
