import { lazy, Suspense } from 'react';
import { createBrowserRouter, Navigate, RouterProvider } from 'react-router-dom';
import { DashboardShell } from '@/components/layout/DashboardShell';

// Lazy-loaded pages — each becomes a separate chunk
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const SearchPage = lazy(() => import('@/pages/Search'));
const CatalystsPage = lazy(() => import('@/pages/Catalysts'));
const DocumentExplorer = lazy(() => import('@/pages/Documents'));
const DocumentDetail = lazy(() => import('@/pages/DocumentDetail'));
const PlaygroundPage = lazy(() => import('@/pages/PlaygroundPage'));
const ThemeExplorer = lazy(() => import('@/pages/Themes'));
const ThemeDetail = lazy(() => import('@/pages/ThemeDetail'));
const AlertCenter = lazy(() => import('@/pages/Alerts'));
const GraphViewer = lazy(() => import('@/pages/Graph'));
const EntityExplorer = lazy(() => import('@/pages/Entities'));
const EntityDetailPage = lazy(() => import('@/pages/EntityDetail'));
const SettingsPage = lazy(() => import('@/pages/Settings'));
const EvidencePage = lazy(() => import('@/pages/EvidencePage'));
const DivergencePage = lazy(() => import('@/pages/DivergencePage'));

function PageLoader() {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
    </div>
  );
}

const router = createBrowserRouter([
  {
    element: <DashboardShell />,
    children: [
      {
        index: true,
        element: (
          <Suspense fallback={<PageLoader />}>
            <Dashboard />
          </Suspense>
        ),
      },
      {
        path: 'catalysts',
        element: (
          <Suspense fallback={<PageLoader />}>
            <CatalystsPage />
          </Suspense>
        ),
      },
      {
        path: 'evidence',
        element: (
          <Suspense fallback={<PageLoader />}>
            <EvidencePage />
          </Suspense>
        ),
      },
      {
        path: 'divergence',
        element: (
          <Suspense fallback={<PageLoader />}>
            <DivergencePage />
          </Suspense>
        ),
      },
      {
        path: 'search',
        element: (
          <Suspense fallback={<PageLoader />}>
            <SearchPage />
          </Suspense>
        ),
      },
      {
        path: 'documents',
        element: (
          <Suspense fallback={<PageLoader />}>
            <DocumentExplorer />
          </Suspense>
        ),
      },
      {
        path: 'documents/:documentId',
        element: (
          <Suspense fallback={<PageLoader />}>
            <DocumentDetail />
          </Suspense>
        ),
      },
      {
        path: 'playground',
        element: (
          <Suspense fallback={<PageLoader />}>
            <PlaygroundPage />
          </Suspense>
        ),
      },
      {
        path: 'playground/:tab',
        element: (
          <Suspense fallback={<PageLoader />}>
            <PlaygroundPage />
          </Suspense>
        ),
      },
      {
        path: 'themes',
        element: (
          <Suspense fallback={<PageLoader />}>
            <ThemeExplorer />
          </Suspense>
        ),
      },
      {
        path: 'themes/:themeId',
        element: (
          <Suspense fallback={<PageLoader />}>
            <ThemeDetail />
          </Suspense>
        ),
      },
      {
        path: 'alerts',
        element: (
          <Suspense fallback={<PageLoader />}>
            <AlertCenter />
          </Suspense>
        ),
      },
      {
        path: 'graph',
        element: (
          <Suspense fallback={<PageLoader />}>
            <GraphViewer />
          </Suspense>
        ),
      },
      {
        path: 'entities',
        element: (
          <Suspense fallback={<PageLoader />}>
            <EntityExplorer />
          </Suspense>
        ),
      },
      {
        path: 'entities/:type/:normalized',
        element: (
          <Suspense fallback={<PageLoader />}>
            <EntityDetailPage />
          </Suspense>
        ),
      },
      {
        path: 'securities',
        element: <Navigate to="/settings" replace />,
      },
      {
        path: 'monitoring',
        element: <Navigate to="/" replace />,
      },
      {
        path: 'settings',
        element: (
          <Suspense fallback={<PageLoader />}>
            <SettingsPage />
          </Suspense>
        ),
      },
      {
        path: '*',
        element: (
          <div className="flex h-full flex-col items-center justify-center gap-2">
            <span className="text-4xl font-bold text-muted-foreground">404</span>
            <span className="text-sm text-muted-foreground">Page not found</span>
          </div>
        ),
      },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
