import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { AuthModal } from '@/components/layout/AuthModal';

export function DashboardShell() {
  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-y-auto">
        <Outlet />
      </main>
      <AuthModal />
    </div>
  );
}
