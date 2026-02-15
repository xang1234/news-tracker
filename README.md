# News Tracker

A full-stack financial news intelligence platform focused on the semiconductor industry. It ingests content from social media, newsletters, and news APIs, enriches it with NLP (entity recognition, sentiment, keywords, event extraction), clusters documents into tradeable themes, models causal supply chain relationships, and surfaces actionable alerts — all backed by a REST/WebSocket API and React dashboard.

## Design Intent

The semiconductor supply chain is dense with signal: a single TSMC capacity announcement ripples through fabless designers, equipment suppliers, and end-market companies. News Tracker is built to capture that signal across platforms, enrich it into structured intelligence, and surface the themes that matter for investment decisions.

Key design goals:

- **Multi-source triangulation.** No single platform tells the full story. Combining Twitter, Reddit, Substack, and financial news APIs provides breadth and corroboration.
- **Opt-in complexity.** Every NLP and analytics feature is behind a feature flag, disabled by default. A deployment can start as a simple ingestion pipeline and graduate to full ML enrichment as needs grow.
- **Streaming architecture.** Redis Streams decouple ingestion from processing from analysis. Workers scale independently and propagate trace context across boundaries.
- **Domain-tuned models.** FinBERT (not general-purpose BERT) for embeddings and sentiment. spaCy EntityRuler for semiconductor-specific entities before statistical NER runs.

## Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │                     Data Sources                       │
                    │  Twitter  ·  Reddit  ·  Substack  ·  Finnhub          │
                    │  NewsAPI  ·  Newsfilter  ·  Marketaux  ·  Finlight    │
                    └──────────────────────┬─────────────────────────────────┘
                                           ▼
                                ┌─────────────────────┐
                                │  Platform Adapters   │
                                │  (rate-limited I/O)  │
                                └──────────┬──────────┘
                                           ▼
                                ┌─────────────────────┐
                                │    Redis Streams     │
                                │   (document_queue)   │
                                └──────────┬──────────┘
                                           ▼
                                ┌─────────────────────┐
                                │  Processing Pipeline │
                                │  Spam · Dedup · NER  │
                                │  Keywords · Events   │
                                │  Tickers · Authority │
                                └──────────┬──────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
          ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
          │  embedding_queue │  │ sentiment_queue   │  │ clustering_queue  │
          └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                   ▼                     ▼                      ▼
          ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
          │ EmbeddingWorker  │  │ SentimentWorker   │  │ ClusteringWorker │
          │ FinBERT · MiniLM │  │ FinBERT sentiment │  │ pgvector HNSW    │
          └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                   └──────────────────────┼──────────────────────┘
                                          ▼
                        ┌─────────────────────────────────┐
                        │     PostgreSQL  +  pgvector       │
                        │  Documents · Themes · Graph       │
                        │  Alerts · Backtest · Securities   │
                        └────────────────┬────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
          ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
          │   FastAPI REST   │ │  WebSocket /ws/   │ │   React Web UI   │
          │   API + CORS     │ │  alerts (pub/sub) │ │   (18 pages)     │
          └──────────────────┘ └──────────────────┘ └──────────────────┘
```

### How data flows

1. **Adapters** fetch raw content from each platform, respecting per-source rate limits. Each adapter implements `_fetch_raw()` (I/O with rate limiting) and `_transform()` (normalization into `NormalizedDocument`). News APIs support multiple API keys with round-robin rotation.

2. **Redis Streams** buffer documents between stages. Workers use consumer groups (`XREADGROUP`) with automatic claim (`XAUTOCLAIM`) for at-least-once delivery. Failed messages get exponential backoff. W3C `traceparent` fields propagate distributed traces across worker boundaries.

3. **Processing** runs spam detection, MinHash deduplication, ticker extraction, and optional NER/keywords/events. Each enrichment stage is independently gated by a feature flag.

4. **Workers** consume from dedicated queues. The embedding worker selects FinBERT (768-dim) for long-form financial content and MiniLM (384-dim) for short social posts (<300 chars). The sentiment worker runs ProsusAI/finbert with optional entity-level context windows. The clustering worker assigns documents to themes in real-time via pgvector HNSW cosine similarity.

5. **PostgreSQL + pgvector** stores everything: documents with dual embedding columns, themes with EMA-updated centroids, a causal supply chain graph, alerts, backtest results, and a security master. HNSW indexes enable sub-millisecond semantic search.

## Key Subsystems

### Theme Clustering

Dual-path clustering combines offline discovery with real-time assignment:

- **Batch** (daily cron): BERTopic (UMAP + HDBSCAN + c-TF-IDF) discovers themes from recent documents. Produces deterministic theme IDs via `theme_{sha256(sorted_topic_words)[:12]}`.
- **Real-time**: ClusteringWorker assigns each incoming document to existing themes via cosine similarity against centroids. Three tiers — strong match (updates centroid via EMA), weak match (assigns without update), or new candidate.

Themes have lifecycle stages (`emerging → active → mature → declining → dead`) and are ranked by actionability for swing or position trading, producing tiered output (Tier 1/2/3).

### Causal Graph

A directed graph models the semiconductor supply chain (~50 nodes, 100+ edges). Node types: company, technology, product, market_segment. Edge types: `supplies_to`, `competes_with`, `depends_on`, `manufactures`, `develops`.

Sentiment propagation runs BFS through causal edges. Edge weights carry sign — `competes_with` flips impact direction (Samsung bad news → TSMC slight positive). The first-arrival-wins rule prevents double-counting at deeper hops. Traversal uses recursive CTEs with cycle detection (`NOT node_id = ANY(path)`).

### Alerts & Notifications

Stateless trigger functions evaluate conditions (volume spike, sentiment shift, new theme, authority change) and persist alerts with deduplication via Redis `SET NX`. Notifications dispatch to Webhook and Slack channels, each wrapped in a circuit breaker. Failures never block alert persistence. Real-time alerts stream to WebSocket clients via Redis pub/sub fan-out.

### Authority Scoring

Bayesian model with Beta prior smoothing: `(correct + alpha) / (total + alpha + beta)`. New sources ramp linearly over a 30-day probation period. Tier-based base weights (anonymous 1.0 / verified 5.0 / research 10.0) and exponential time decay ensure established, accurate sources outrank noisy ones.

### Backtesting

Evaluates theme ranking strategies against historical price data. Point-in-time queries filter on `fetched_at` (ingestion time), not `timestamp` (publication time), to prevent look-ahead bias. Produces Sharpe ratio, max drawdown, hit rate, and cumulative returns. Matplotlib visualization generates comparison charts.

### NLP Pipeline

| Stage | Model / Method | Output |
|-------|---------------|--------|
| **Embeddings** | FinBERT (768-dim) / MiniLM (384-dim) | Vector representations, cached by content hash |
| **Sentiment** | ProsusAI/finbert | Document + entity-level pos/neg/neutral with confidence |
| **NER** | spaCy transformer + EntityRuler + fastcoref | TICKER, COMPANY, PRODUCT, TECHNOLOGY, METRIC entities |
| **Keywords** | TextRank (rapid-textrank) | Top-N keywords with relevance scores |
| **Events** | Regex SVO patterns + time normalization | Structured events with investment signal classification |

All ML models use lazy loading (deferred until first use) and content-hash caching in Redis.

### Drift Detection

Four monitoring checks detect model and data drift:

| Check | Metric | Frequency |
|-------|--------|-----------|
| Embedding | KL divergence on L2 norm distributions | Hourly |
| Fragmentation | Cluster count and singleton ratio | Daily |
| Sentiment | Z-score deviation from baseline | Daily |
| Centroid Stability | Cosine distance of theme centroids over time | Daily |

## Web UI

A React single-page application (React 18 + TypeScript + Vite + Tailwind CSS + React Query + Zustand) with 18 lazy-loaded pages:

| Page | Route | Purpose |
|------|-------|---------|
| Dashboard | `/` | System overview, key metrics, recent activity |
| Search | `/search` | Semantic similarity search with filters |
| Documents | `/documents` | Browse and filter ingested documents |
| Document Detail | `/documents/:id` | Full content with entities, keywords, events, sentiment |
| Themes | `/themes` | Theme explorer by lifecycle stage |
| Theme Detail | `/themes/:id` | Documents, sentiment, metrics, events, graph links |
| Alerts | `/alerts` | Alert center with severity/trigger filters |
| Graph | `/graph` | Interactive causal graph visualization |
| Entities | `/entities` | Entity explorer with trending view |
| Entity Detail | `/entities/:type/:name` | Stats, documents, co-occurrence, sentiment |
| Securities | `/securities` | Security master CRUD |
| Monitoring | `/monitoring` | Drift detection and system health |
| Playgrounds | `/playground/*` | Interactive testing for embed, sentiment, NER, keywords, events |
| Settings | `/settings` | Configuration and preferences |

The UI uses a persistent dark theme and renders all server state through React Query hooks with typed request/response interfaces.

## Quick Start

```bash
# Install Python dependencies
uv sync --extra dev

# Start infrastructure
docker compose up -d

# Initialize schema and seed data
uv run news-tracker init-db
uv run news-tracker graph seed

# Run a mock ingestion cycle
uv run news-tracker run-once --mock

# Start the API server
uv run news-tracker serve

# Start the frontend (requires Node.js 18+)
cd frontend && npm install && npx vite
```

## Docker Compose

`docker compose up -d` launches the full stack:

| Service | Port | Role |
|---------|------|------|
| PostgreSQL + pgvector | 5432 | Document storage, vector search (HNSW indexes) |
| Redis | 6379 | Streams (queues), pub/sub (alerts), caching |
| Prometheus | 9090 | Metrics scraping + alert rules |
| AlertManager | 9093 | Alert routing |
| Grafana | 3000 | Auto-provisioned dashboards (admin/admin) |
| Jaeger | 16686 | Distributed tracing (OTLP gRPC on 4317) |
| API server | 8001 | FastAPI application |
| Workers | — | Ingestion, embedding, sentiment, clustering |
| Frontend | 5151 | React dev server (Vite) |

## CLI

All commands are available via the `news-tracker` entry point.

```bash
# Core
news-tracker serve                        # FastAPI server
news-tracker worker                       # Ingestion + processing loop
news-tracker embedding-worker             # FinBERT/MiniLM embedding generation
news-tracker sentiment-worker             # Sentiment analysis
news-tracker clustering-worker            # Real-time theme assignment
news-tracker init-db                      # Initialize schema
news-tracker health                       # Check dependencies

# Ingestion
news-tracker run-once [--mock] [--with-embeddings] [--with-sentiment] [--verify]
news-tracker cleanup [--days 90] [--dry-run]

# Clustering
news-tracker daily-clustering [--date YYYY-MM-DD]
news-tracker cluster fit|run|backfill|merge|status|recompute-centroids

# Search
news-tracker vector-search "query" [--limit N] [--platform X] [--ticker Y]

# Backtesting
news-tracker backtest run --start YYYY-MM-DD --end YYYY-MM-DD [--strategy swing|position]
news-tracker backtest plot --run-id <id>

# Graph & Monitoring
news-tracker graph seed
news-tracker drift check-quick|check-daily|report
```

## Feature Flags

Every enrichment and analytics feature is opt-in via environment variable:

| Feature | Flag | What it enables |
|---------|------|-----------------|
| NER | `NER_ENABLED` | spaCy entity recognition + coreference resolution |
| Keywords | `KEYWORDS_ENABLED` | TextRank keyword extraction |
| Events | `EVENTS_ENABLED` | SVO event extraction + time normalization |
| Clustering | `CLUSTERING_ENABLED` | BERTopic discovery + real-time assignment |
| Volume Metrics | `VOLUME_METRICS_ENABLED` | EMA-based volume and velocity tracking |
| Ranking | `RANKING_ENABLED` | Theme actionability scoring (swing/position) |
| Graph | `GRAPH_ENABLED` | Causal supply chain graph traversal |
| Propagation | `PROPAGATION_ENABLED` | Sentiment propagation through graph edges |
| Alerts | `ALERTS_ENABLED` | Volume spike, sentiment shift, new theme triggers |
| Notifications | `NOTIFICATIONS_ENABLED` | Webhook + Slack channels with circuit breakers |
| Backtest | `BACKTEST_ENABLED` | Historical strategy backtesting |
| Scoring | `SCORING_ENABLED` | LLM compellingness scoring (rule → GPT → Claude) |
| Security Master | `SECURITY_MASTER_ENABLED` | DB-backed ticker lookup with fuzzy search |
| Authority | `AUTHORITY_ENABLED` | Bayesian source authority scoring |
| Drift Detection | `DRIFT_ENABLED` | Embedding, fragmentation, sentiment, stability checks |
| Tracing | `TRACING_ENABLED` | OpenTelemetry distributed tracing (OTLP → Jaeger) |
| WebSocket Alerts | `WS_ALERTS_ENABLED` | Real-time alert streaming via WebSocket |
| Rate Limiting | `RATE_LIMIT_ENABLED` | Per-endpoint rate limiting (slowapi) |

Each feature has its own `ENV_PREFIX_*` namespace for fine-grained configuration. See `src/config/settings.py` for the full reference.

## Configuration

Settings are managed via environment variables (Pydantic BaseSettings). Key groups:

```bash
# Infrastructure
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/news_tracker
REDIS_URL=redis://localhost:6379/0
API_KEYS=key1,key2               # Comma-separated; empty = dev mode (no auth)

# Data sources (each adapter has its own credentials)
TWITTER_BEARER_TOKEN=...
REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=...
NEWSFILTER_API_KEYS=k1,k2,k3    # Comma-separated for key rotation
MARKETAUX_API_KEYS=...
FINLIGHT_API_KEYS=...

# ML models
EMBEDDING_MODEL_NAME=ProsusAI/finbert
SENTIMENT_MODEL_NAME=ProsusAI/finbert
NER_SPACY_MODEL=en_core_web_trf

# Processing
SPAM_THRESHOLD=0.7
DUPLICATE_THRESHOLD=0.85
```

## Observability

- **Tracing:** OpenTelemetry with OTLP gRPC exporter to Jaeger. W3C `traceparent` propagated through Redis Streams so traces span worker boundaries.
- **Metrics:** Prometheus scraping with auto-provisioned Grafana dashboards. Workers expose `/metrics` on configurable ports.
- **Logging:** structlog with correlation ID contextvars. `X-Request-ID` / `X-Correlation-ID` headers propagate through the request lifecycle.
- **Health:** Three-tier status — `unhealthy` (DB down) → `degraded` (Redis down) → `healthy` — with per-subsystem detail.

## Development

```bash
uv sync --extra dev                              # Install with test dependencies
uv run pytest tests/ -v                          # Run all tests
uv run pytest tests/ -v -m "not integration"     # Skip integration tests
uv run pytest tests/test_embedding/test_service.py -v  # Single file
docker build -t news-tracker .                   # Build Docker image
```

Tests mirror the source tree (`tests/test_<module>/test_<file>.py`). Every service module follows the convention of `config.py`, `service.py`, and `__init__.py` with `__all__` exports.

## Project Structure

```
src/
├── alerts/          # Triggers, notifications (webhook + Slack), WebSocket broadcast
├── api/             # FastAPI routes, middleware, rate limiting
├── authority/       # Bayesian source authority scoring
├── backtest/        # Historical simulation engine + visualization
├── clustering/      # BERTopic (batch) + ClusteringWorker (real-time)
├── config/          # Pydantic settings, ticker lists, source config
├── embedding/       # FinBERT + MiniLM dual-model service + worker
├── event_extraction/# SVO pattern extraction + time normalization
├── feedback/        # User quality ratings
├── graph/           # Causal graph (recursive CTE) + sentiment propagation
├── ingestion/       # Platform adapters, Redis queue, preprocessing
├── keywords/        # TextRank keyword extraction
├── monitoring/      # 4-check drift detection
├── ner/             # spaCy NER + EntityRuler + fastcoref
├── observability/   # Logging, Prometheus metrics, OTLP tracing
├── queues/          # Redis Streams base queue with backoff
├── scoring/         # 3-tier LLM compellingness scoring
├── security_master/ # Ticker database with pg_trgm fuzzy search
├── sentiment/       # FinBERT sentiment service + worker
├── sources/         # Platform source management
├── storage/         # asyncpg database + document repository
├── themes/          # Lifecycle, volume metrics, ranking
└── vectorstore/     # pgvector HNSW semantic search

frontend/src/
├── api/             # Axios client, query key factory, typed hooks
├── components/      # Layout shell + domain components (with Skeleton variants)
├── lib/             # Constants, formatters, Tailwind utilities
├── pages/           # 18 lazy-loaded route components
└── stores/          # Zustand stores (auth, UI state)
```

## License

MIT
