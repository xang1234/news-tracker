# News Tracker — Web UI

React single-page application for the News Tracker analyst workflow. It centers source operations, document review, theme exploration, alert triage, and a small set of playground tools.

## Stack

- **React 19** with TypeScript
- **Vite** for builds and HMR
- **Tailwind CSS** for styling (persistent dark theme)
- **React Query** for server state management
- **Zustand** for client state
- **React Router** for client-side routing

## Getting Started

Requires Node.js 18+ (22 recommended).

```bash
npm install                      # Install dependencies
npm run dev                      # Dev server at http://localhost:5151
npx tsc --noEmit                 # Type check (zero output = success)
npx eslint .                     # Lint
npx vite build                   # Production build
```

The dev server proxies API requests to the backend at `http://localhost:8001`.

## Project Structure

```
src/
├── api/
│   ├── http.ts              # Axios instance with /api baseURL, auth, correlation IDs
│   ├── queryKeys.ts         # React Query key factories (use these, never hand-craft keys)
│   └── hooks/use*.ts        # One hook per API endpoint, typed request/response
├── components/
│   ├── layout/              # DashboardShell, navigation, sidebar
│   ├── domain/              # Reusable cards/panels (each exports a Skeleton variant)
│   └── ui/                  # Base UI primitives
├── lib/
│   ├── constants.ts         # PLATFORMS, color maps
│   ├── formatters.ts        # timeAgo, pct, truncate, latency
│   └── utils.ts             # cn() for Tailwind class merging
├── pages/                   # One default export per route, lazy-loaded in App.tsx
├── stores/                  # Zustand stores
└── App.tsx                  # Router definition
```

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | System overview and key metrics |
| Search | `/search` | Semantic search with filters |
| Documents | `/documents` | Document browser with filters and sorting |
| Document Detail | `/documents/:documentId` | Full document view (entities, keywords, events) |
| Theme Explorer | `/themes` | Themes by lifecycle stage |
| Theme Detail | `/themes/:themeId` | Theme documents, sentiment, metrics, events |
| Alert Center | `/alerts` | Alert list with severity/trigger filters |
| Causal Graph | `/graph` | Interactive graph visualization and propagation |
| Entity Explorer | `/entities` | Entity list with trending and search |
| Entity Detail | `/entities/:type/:normalized` | Entity stats, co-occurrence, sentiment, merge |
| Settings | `/settings` | Source operations and security master administration |
| Securities | `/securities` | Alias to the securities tab under settings |
| Embed Playground | `/playground/embed` | Test embedding endpoint |
| Sentiment Playground | `/playground/sentiment` | Test sentiment analysis |
| NER Playground | `/playground/ner` | Test entity extraction |
| Keywords Playground | `/playground/keywords` | Test keyword extraction |
| Events Playground | `/playground/events` | Test event extraction |

## Conventions

- **Path alias:** `@/` maps to `src/` (configured in `tsconfig.json`)
- **Dark theme:** Always on (`<html class="dark">`). Use semantic tokens: `text-foreground`, `bg-card`, `border-border`
- **Loading states:** Domain components export a `Skeleton` variant for suspense/loading
- **Query keys:** Always use factory functions from `src/api/queryKeys.ts`
- **Hooks:** One file per API endpoint in `src/api/hooks/`, with typed interfaces co-located
