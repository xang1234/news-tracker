import { lazy, Suspense } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { DashboardShell } from '@/components/layout/DashboardShell';

// Lazy-loaded pages — each becomes a separate chunk
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const SearchPage = lazy(() => import('@/pages/Search'));
const DocumentExplorer = lazy(() => import('@/pages/Documents'));
const DocumentDetail = lazy(() => import('@/pages/DocumentDetail'));
const EmbedPlayground = lazy(() => import('@/pages/EmbedPlayground'));
const SentimentPlayground = lazy(() => import('@/pages/SentimentPlayground'));
const MonitoringPage = lazy(() => import('@/pages/Monitoring'));
const ThemeExplorer = lazy(() => import('@/pages/Themes'));
const ThemeDetail = lazy(() => import('@/pages/ThemeDetail'));
const AlertCenter = lazy(() => import('@/pages/Alerts'));
const GraphViewer = lazy(() => import('@/pages/Graph'));
const SettingsPage = lazy(() => import('@/pages/Settings'));

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
        path: 'playground/embed',
        element: (
          <Suspense fallback={<PageLoader />}>
            <EmbedPlayground />
          </Suspense>
        ),
      },
      {
        path: 'playground/sentiment',
        element: (
          <Suspense fallback={<PageLoader />}>
            <SentimentPlayground />
          </Suspense>
        ),
      },
      {
        path: 'monitoring',
        element: (
          <Suspense fallback={<PageLoader />}>
            <MonitoringPage />
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
