import { lazy, Suspense } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Header } from '@/components/layout/Header';
import { cn } from '@/lib/utils';

const EmbedPlayground = lazy(() => import('@/pages/EmbedPlayground'));
const SentimentPlayground = lazy(() => import('@/pages/SentimentPlayground'));
const NERPlayground = lazy(() => import('@/pages/NERPlayground'));
const KeywordsPlayground = lazy(() => import('@/pages/KeywordsPlayground'));
const EventsPlayground = lazy(() => import('@/pages/EventsPlayground'));

const TABS = [
  { key: 'embed', label: 'Embed' },
  { key: 'sentiment', label: 'Sentiment' },
  { key: 'ner', label: 'NER' },
  { key: 'keywords', label: 'Keywords' },
  { key: 'events', label: 'Events' },
] as const;

type TabKey = (typeof TABS)[number]['key'];

export default function PlaygroundPage() {
  const { tab: urlTab } = useParams<{ tab?: string }>();
  const navigate = useNavigate();
  const activeTab: TabKey = TABS.some((t) => t.key === urlTab) ? (urlTab as TabKey) : 'embed';

  function handleTabChange(key: TabKey) {
    navigate(`/playground/${key}`, { replace: true });
  }

  return (
    <>
      <Header title="Playground" />
      <div className="border-b border-border px-6">
        <div className="flex gap-0" role="tablist" aria-label="Playground tabs">
          {TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              role="tab"
              aria-selected={activeTab === t.key}
              onClick={() => handleTabChange(t.key)}
              className={cn(
                'border-b-2 px-4 py-2.5 text-sm font-medium transition-colors',
                activeTab === t.key
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground',
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
      <Suspense
        fallback={
          <div className="flex items-center justify-center p-12">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        }
      >
        <div className="[&>header]:hidden">
          {activeTab === 'embed' && <EmbedPlayground />}
          {activeTab === 'sentiment' && <SentimentPlayground />}
          {activeTab === 'ner' && <NERPlayground />}
          {activeTab === 'keywords' && <KeywordsPlayground />}
          {activeTab === 'events' && <EventsPlayground />}
        </div>
      </Suspense>
    </>
  );
}
