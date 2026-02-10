# News Tracker

Full-stack financial news intelligence platform for semiconductor and tech markets. Ingests multi-platform data, applies NLP enrichment (NER, sentiment, keywords, events), clusters documents into tradeable themes, ranks them by actionability, models causal supply chain relationships, and surfaces alerts — with a React web UI, REST API, backtesting engine, and comprehensive observability.

## Architecture

```
                        ┌────────────────────────────────────────────────────────┐
                        │                     Data Sources                       │
                        │  Twitter · Reddit · Substack · Finnhub · NewsAPI       │
                        │  Alpha Vantage · Newsfilter · Marketaux · Finlight     │
                        └──────────────────────┬─────────────────────────────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │  Adapters (fetch +   │
                                    │  rate-limited I/O)   │
                                    └──────────┬──────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │    Redis Streams     │
                                    │  (document_queue)    │
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
              │  (Redis Stream)  │  │ (Redis Stream)    │  │ (Redis Stream)    │
              └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                       ▼                     ▼                      ▼
              ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
              │ EmbeddingWorker  │  │ SentimentWorker   │  │ ClusteringWorker │
              │ FinBERT (768)    │  │ FinBERT sentiment │  │ pgvector HNSW    │
              │ MiniLM  (384)    │  │ + entity-level    │  │ real-time assign │
              └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                       └──────────────────────┼──────────────────────┘
                                              ▼
                            ┌─────────────────────────────────┐
                            │     PostgreSQL + pgvector        │
                            │  Documents · Themes · Graph      │
                            │  Alerts · Backtest · Securities  │
                            └────────────────┬────────────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        ▼                    ▼                    ▼
              ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
              │   FastAPI REST   │ │  WebSocket /ws/   │ │   React Web UI   │
              │   39 endpoints   │ │  alerts (pub/sub) │ │  18 pages        │
              └──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Data Sources

| Platform | API/Method | Rate Limit | Content Type |
|----------|------------|------------|--------------|
| **Twitter** | Twitter API v2 (Bearer Token) | 30 req/min | Posts with cashtags, financial influencers |
| **Reddit** | Reddit OAuth API | 60 req/min | Posts from r/wallstreetbets, r/stocks, r/semiconductors, etc. |
| **Substack** | RSS feeds (public) | 10 req/min | Newsletter articles from SemiAnalysis, Stratechery, Asianometry |
| **News APIs** | Finnhub, NewsAPI, Alpha Vantage, Newsfilter, Marketaux, Finlight | 60 req/min | Financial news with multi-source fallback |

- Deduplicates across sources, applies source authority weighting (WSJ/Bloomberg/Reuters ranked higher)
- Newsfilter, Marketaux, and Finlight support multiple API keys with round-robin rotation and exponential backoff on rate limits

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Start infrastructure (PostgreSQL, Redis, Prometheus, Grafana, Jaeger)
docker compose up -d

# Initialize database schema
uv run news-tracker init-db

# Seed causal graph with semiconductor supply chain data
uv run news-tracker graph seed

# Run with mock data (for testing)
uv run news-tracker run-once --mock

# Run full pipeline (embeddings + sentiment + verification)
uv run news-tracker run-once --mock --with-embeddings --with-sentiment --verify

# Start the API server
uv run news-tracker serve

# Start the frontend dev server (requires Node.js 18+)
cd frontend && npm install && npx vite
```

## CLI Reference

All commands are available via the `news-tracker` entry point.

### Core Services

```bash
news-tracker ingest [--mock] [--metrics/--no-metrics]       # Run ingestion service
news-tracker process [--batch-size 32] [--metrics/--no-metrics]  # Run processing service
news-tracker worker [--mock] [--metrics-port 8000]          # Run ingestion + processing together
news-tracker serve [--host 0.0.0.0] [--port 8001] [--reload] [--metrics-port 8000]  # FastAPI server
news-tracker init-db                                         # Initialize database schema
news-tracker health                                          # Check all dependencies
```

### Ingestion & Processing

```bash
# Single ingestion cycle with optional enrichment stages
news-tracker run-once [--mock] [--with-embeddings] [--with-sentiment] [--verify]

# Storage management
news-tracker cleanup [--days 90] [--dry-run]
```

### Workers

```bash
news-tracker sentiment-worker [--batch-size 16] [--metrics/--no-metrics] [--metrics-port 8001]
news-tracker clustering-worker [--batch-size 32] [--metrics/--no-metrics] [--metrics-port 8002]
```

### Theme Clustering

```bash
news-tracker cluster fit [--days 30]                         # Discover themes via BERTopic
news-tracker cluster run [--date YYYY-MM-DD] [--dry-run]     # Daily batch clustering
news-tracker cluster backfill --start YYYY-MM-DD --end YYYY-MM-DD  # Backfill date range
news-tracker cluster merge [--dry-run] [--threshold 0.85]    # Merge similar themes
news-tracker cluster status                                   # Theme count + lifecycle breakdown
news-tracker cluster recompute-centroids                      # Recalculate centroids from embeddings

# Standalone daily clustering (cron-friendly)
news-tracker daily-clustering [--date YYYY-MM-DD] [--dry-run]
```

### Semantic Search

```bash
news-tracker vector-search "NVIDIA AI demand" --limit 10
news-tracker vector-search "semiconductor supply chain" \
  --platform twitter --platform reddit \
  --ticker NVDA --ticker AMD \
  --min-authority 0.6 --threshold 0.7
```

### Backtesting

```bash
news-tracker backtest run --start 2025-01-01 --end 2025-06-30 \
  [--strategy swing|position] [--top-n 10] [--horizon 5]

news-tracker backtest plot --run-id <id> [--output-dir ./backtest_plots]
```

### Causal Graph

```bash
news-tracker graph seed    # Seed ~50 nodes + 100 edges (supply chain, competition, tech deps)
```

### Drift Detection

```bash
news-tracker drift check-quick    # Embedding KL divergence only (hourly cron)
news-tracker drift check-daily    # All 4 checks (daily cron)
news-tracker drift report         # Verbose weekly report
```

### Global Options

```bash
news-tracker --debug <command>    # Enable debug logging
news-tracker <command> --help     # Show help for any command
```

## Feature Flags

All features are opt-in (disabled by default) to allow flexible deployment. Enable via environment variables:

| Feature | Env Variable | Description |
|---------|-------------|-------------|
| NER | `NER_ENABLED=true` | Named entity recognition (spaCy + EntityRuler + fastcoref) |
| Keywords | `KEYWORDS_ENABLED=true` | TextRank keyword extraction |
| Events | `EVENTS_ENABLED=true` | SVO event extraction with time normalization |
| Clustering | `CLUSTERING_ENABLED=true` | BERTopic theme discovery + real-time assignment |
| Volume Metrics | `VOLUME_METRICS_ENABLED=true` | EMA-based volume and velocity tracking |
| Ranking | `RANKING_ENABLED=true` | Theme actionability ranking (swing/position strategies) |
| Graph | `GRAPH_ENABLED=true` | Causal supply chain graph |
| Propagation | `PROPAGATION_ENABLED=true` | Sentiment propagation through causal graph |
| Alerts | `ALERTS_ENABLED=true` | Alert triggers (volume spike, sentiment shift, new theme) |
| Notifications | `NOTIFICATIONS_ENABLED=true` | Webhook + Slack notification channels |
| Backtest | `BACKTEST_ENABLED=true` | Historical backtesting engine |
| Scoring | `SCORING_ENABLED=true` | LLM compellingness scoring (rule → GPT → Claude) |
| Security Master | `SECURITY_MASTER_ENABLED=true` | DB-backed ticker/company database with fuzzy search |
| Feedback | `FEEDBACK_ENABLED=true` | User quality ratings for themes/alerts/documents |
| Authority | `AUTHORITY_ENABLED=true` | Bayesian source authority scoring with time decay |
| Drift Detection | `DRIFT_ENABLED=true` | Embedding, fragmentation, sentiment, and stability drift checks |
| Tracing | `TRACING_ENABLED=true` | OpenTelemetry distributed tracing (OTLP gRPC) |
| WebSocket Alerts | `WS_ALERTS_ENABLED=true` | Real-time alert streaming via WebSocket |
| Rate Limiting | `RATE_LIMIT_ENABLED=true` | Per-endpoint rate limiting (slowapi) |

## API

The FastAPI server exposes 39 endpoints across 15 categories, with optional API key authentication, CORS, request timeouts, correlation ID tracking, and per-endpoint rate limiting.

### Authentication

All endpoints (except `/health`) require an `X-API-KEY` header when `API_KEYS` is configured. Leave `API_KEYS` empty for development mode (no auth).

### Endpoints

#### Embeddings & NLP

| Method | Path | Description |
|--------|------|-------------|
| POST | `/embed` | Generate embeddings (auto/finbert/minilm model selection) |
| POST | `/sentiment` | Document + optional entity-level sentiment analysis |
| POST | `/ner` | Batch NER entity extraction (feature-gated) |
| POST | `/keywords` | Batch TextRank keyword extraction (feature-gated) |
| POST | `/events/extract` | SVO event extraction from text (feature-gated) |

#### Search

| Method | Path | Description |
|--------|------|-------------|
| POST | `/search/similar` | Semantic similarity search with filters (platform, ticker, authority, date range) |

#### Documents

| Method | Path | Description |
|--------|------|-------------|
| GET | `/documents` | List with filters (platform, ticker, query, sort, pagination) |
| GET | `/documents/stats` | Aggregate stats (counts, coverage, date range) |
| GET | `/documents/{id}` | Full detail (content, entities, keywords, events) |

#### Themes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/themes` | List themes (lifecycle_stage filter, pagination) |
| GET | `/themes/ranked` | Ranked by actionability (strategy, max_tier, limit) |
| GET | `/themes/{id}` | Theme detail (optional centroid via `?include_centroid=true`) |
| GET | `/themes/{id}/documents` | Documents assigned to theme (platform, authority filters) |
| GET | `/themes/{id}/sentiment` | Aggregated sentiment with exponential decay weighting |
| GET | `/themes/{id}/metrics` | Daily metrics time series |
| GET | `/themes/{id}/events` | Events via ticker overlap (dedup, investment_signal) |

#### Alerts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/alerts` | List (severity, trigger_type, theme_id, acknowledged filters) |
| PATCH | `/alerts/{id}/acknowledge` | Mark alert as acknowledged |

#### Causal Graph

| Method | Path | Description |
|--------|------|-------------|
| GET | `/graph/nodes` | List graph nodes (optional node_type filter) |
| GET | `/graph/nodes/{id}/subgraph` | Subgraph around a node (depth param, 1-5) |
| POST | `/graph/propagate` | Sentiment propagation through causal edges |

#### Entities

| Method | Path | Description |
|--------|------|-------------|
| GET | `/entities` | List entities with search/filter/sort (feature-gated: `ner_enabled`) |
| GET | `/entities/stats` | Aggregate entity stats (total, by_type, docs_with_entities) |
| GET | `/entities/trending` | Entities with mention spikes (recent vs baseline) |
| GET | `/entities/{type}/{normalized}` | Entity detail (stats, platforms, graph link) |
| GET | `/entities/{type}/{normalized}/documents` | Documents mentioning entity |
| GET | `/entities/{type}/{normalized}/cooccurrence` | Co-occurring entities (Jaccard similarity) |
| GET | `/entities/{type}/{normalized}/sentiment` | Aggregate sentiment + trend |
| POST | `/entities/{type}/{normalized}/merge` | Merge entity into another (admin) |

#### Securities

| Method | Path | Description |
|--------|------|-------------|
| GET | `/securities` | List securities with filters (feature-gated: `security_master_enabled`) |
| POST | `/securities` | Create a new security |
| PUT | `/securities/{ticker}/{exchange}` | Update a security |
| DELETE | `/securities/{ticker}/{exchange}` | Deactivate (soft delete) a security |

#### Feedback

| Method | Path | Description |
|--------|------|-------------|
| POST | `/feedback` | Submit quality rating for theme/alert/document |
| GET | `/feedback/stats` | Aggregated feedback statistics by entity type |

#### WebSocket

| Method | Path | Description |
|--------|------|-------------|
| WS | `/ws/alerts` | Real-time alert stream (severity, theme_id, api_key query params) |

#### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Three-tier status: unhealthy (DB down) → degraded (Redis down) → healthy |

### Rate Limiting

When `RATE_LIMIT_ENABLED=true`, endpoints are rate-limited per client IP:

| Scope | Default | Applies To |
|-------|---------|-----------|
| `RATE_LIMIT_DEFAULT` | 60/min | Most endpoints |
| `RATE_LIMIT_EMBED` | 30/min | `/embed` |
| `RATE_LIMIT_SENTIMENT` | 30/min | `/sentiment`, theme sentiment |
| `RATE_LIMIT_SEARCH` | 60/min | `/search/similar`, document stats |
| `RATE_LIMIT_GRAPH` | 30/min | `/graph/*` |
| `RATE_LIMIT_ENTITIES` | 60/min | `/entities/*` |
| `RATE_LIMIT_ADMIN` | 30/min | Entity merge, security CRUD |

## Web UI

A React single-page application provides a full dashboard for exploring data, themes, alerts, and system health.

**Stack:** React 18 + TypeScript + Vite + Tailwind CSS + React Query + Zustand

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | System overview, key metrics, recent activity |
| Search | `/search` | Semantic search with filter controls and result cards |
| Documents | `/documents` | Browse, filter, and inspect ingested documents |
| Document Detail | `/documents/:id` | Full document view with entities, keywords, events, sentiment |
| Theme Explorer | `/themes` | List themes by lifecycle stage, search, pagination |
| Theme Detail | `/themes/:id` | Theme documents, sentiment, metrics, events, graph links |
| Alert Center | `/alerts` | Alerts with severity/trigger filters, acknowledge actions |
| Causal Graph | `/graph` | Interactive graph visualization, node detail, propagation |
| Entity Explorer | `/entities` | Entity list with search, type filter, trending view |
| Entity Detail | `/entities/:type/:name` | Entity stats, documents, co-occurrence, sentiment, merge |
| Securities | `/securities` | Security master CRUD (create, edit, deactivate) |
| Monitoring | `/monitoring` | Drift detection results and system health |
| Embed Playground | `/playground/embed` | Test embedding endpoint with model selection |
| Sentiment Playground | `/playground/sentiment` | Test sentiment analysis |
| NER Playground | `/playground/ner` | Test entity extraction |
| Keywords Playground | `/playground/keywords` | Test keyword extraction |
| Events Playground | `/playground/events` | Test event extraction |
| Settings | `/settings` | Configuration and preferences |

All pages are lazy-loaded for optimal bundle splitting. The UI uses a persistent dark theme.

## Processing Pipeline

### Embedding Service

Generates vector representations using a tiered model approach:

| Model | Dimensions | Use Case | Selection Criteria |
|-------|------------|----------|-------------------|
| **FinBERT** | 768 | Financial domain, longer text | Default for most content |
| **MiniLM** | 384 | General purpose, fast | Twitter posts and short text (<300 chars) |

Features: automatic model selection, lazy loading, CUDA/MPS/CPU auto-detection, chunking for documents >512 tokens with overlap and mean pooling, Redis content-hash caching, async processing via Redis Streams.

### Sentiment Analysis

ProsusAI/finbert classifies documents as positive, negative, or neutral with confidence scores. Entity-level sentiment extracts context windows around each mentioned entity for per-entity analysis. Runs as a dedicated worker consuming from `sentiment_queue`.

### Named Entity Recognition

spaCy transformer model + custom EntityRuler for domain-specific patterns + fastcoref coreference resolution (resolves "the chipmaker" → "Samsung" before NER runs). Extracts five entity types:

| Type | Examples |
|------|----------|
| **TICKER** | $NVDA, $AMD, $INTC |
| **COMPANY** | Nvidia → NVIDIA, Taiwan Semiconductor → TSM |
| **PRODUCT** | H100, A100, Snapdragon, GeForce RTX |
| **TECHNOLOGY** | HBM3E, CoWoS, 3nm, EUV lithography |
| **METRIC** | $5.6 billion, 10% YoY, 51% margin |

### Keyword Extraction

TextRank via rapid-textrank extracts top-N keywords per document with relevance scores.

### Event Extraction

Regex-based Subject-Verb-Object pattern extraction with time normalization and theme linking. Surfaces investment-relevant events (launches, earnings, partnerships, supply disruptions).

### Authority Scoring

Bayesian authority model with Beta prior smoothing `(correct + α) / (total + α + β)`, exponential time decay, 30-day probation ramp for new sources, and tier-based base weights (anonymous 1.0, verified 5.0, research 10.0).

## Theme Clustering

Dual-path clustering combines batch discovery with real-time assignment:

- **Batch (daily):** BERTopic (UMAP + HDBSCAN + c-TF-IDF) discovers themes from recent documents. Runs via `daily-clustering` CLI command.
- **Real-time:** ClusteringWorker assigns incoming documents to existing themes via pgvector HNSW cosine similarity. Three-tier assignment: strong (centroid update), weak (assign only), or new candidate.

Theme lifecycle stages: `emerging` → `active` → `mature` → `declining` → `dead`. Centroids update via EMA: `(1 - lr) * old + lr * new`.

The ranking service scores themes by actionability for swing or position trading strategies, producing tiered output (Tier 1/2/3).

## Causal Graph

A directed graph models semiconductor supply chain relationships:

- **Node types:** company, technology, product, market_segment
- **Edge types:** supplies_to, competes_with, depends_on, manufactures, develops
- **Traversal:** Recursive CTE with cycle detection (`NOT node_id = ANY(path)`)
- **Propagation:** BFS sentiment propagation through causal edges, with edge-type-aware sign and weight. `competes_with` edges flip impact direction.

Seed data covers ~50 nodes and 100+ edges across foundries, fabless designers, equipment suppliers, and technology dependencies.

## Alerts & Notifications

Stateless trigger functions evaluate conditions (volume spike, sentiment shift, new theme, authority change) and persist alerts with severity levels and deduplication via Redis SET NX.

Notification dispatch supports Webhook and Slack channels, each wrapped in a circuit breaker (CLOSED → OPEN → HALF_OPEN → CLOSED). Failures never block alert persistence; a Redis fallback queue handles retries.

Real-time alerts stream to WebSocket clients via Redis pub/sub fan-out (`alerts:broadcast` channel). Each uvicorn worker subscribes independently.

## Backtesting

The backtest engine evaluates theme ranking strategies against historical price data:

1. **Point-in-time queries** filter on `fetched_at` (ingestion time), not `timestamp` (publication time), to prevent look-ahead bias
2. **Theme ranking** runs per-day using only data available at that point
3. **Forward returns** measure price change over configurable horizons
4. **Metrics:** Sharpe ratio, max drawdown, hit rate, cumulative returns

Visualization generates matplotlib charts: cumulative returns, drawdown, return scatter, and correlation heatmap.

## Drift Detection

Four monitoring checks detect model and data drift:

| Check | Metric | Frequency |
|-------|--------|-----------|
| **Embedding** | KL divergence on L2 norm distributions | Hourly |
| **Fragmentation** | Cluster count and singleton ratio | Daily |
| **Sentiment** | Z-score deviation from baseline distribution | Daily |
| **Centroid Stability** | Cosine distance of theme centroids over time | Daily |

## Observability

- **Tracing:** OpenTelemetry with OTLP gRPC exporter to Jaeger. W3C `traceparent` propagated through Redis Streams via `BaseRedisQueue._trace_fields()`.
- **Metrics:** Prometheus scraping with auto-provisioned Grafana dashboards. Workers expose `/metrics` on configurable ports.
- **Logging:** structlog with correlation ID contextvars. `X-Request-ID` / `X-Correlation-ID` headers auto-propagate through request lifecycle.
- **Health:** Three-tier status — `unhealthy` (DB down) → `degraded` (Redis down) → `healthy` — with per-subsystem `ComponentHealth` detail.

## Infrastructure

### Docker Compose Services

```bash
docker compose up -d    # Start all services
```

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL + pgvector | 5432 | Document storage, vector search (HNSW indexes) |
| Redis | 6379 | Streams (queues), pub/sub (alerts), caching |
| Prometheus | 9090 | Metrics scraping + alert rules |
| AlertManager | 9093 | Alert routing |
| Grafana | 3000 | Auto-provisioned dashboards (login: admin/admin) |
| Jaeger | 16686 (UI), 4317 (OTLP gRPC) | Distributed tracing |
| news-tracker-api | 8001 | FastAPI application server |

### Database Migrations

Migrations live in `migrations/` and run automatically via `init-db`. For manual application:

```bash
psql $DATABASE_URL -f migrations/001_initial_schema.sql
psql $DATABASE_URL -f migrations/001_embedding_vector_768.sql
# ... through 013_authority.sql
```

## Configuration

Settings are managed via environment variables (Pydantic BaseSettings). Copy `.env.example` to `.env` and configure:

### Infrastructure

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/news_tracker
REDIS_URL=redis://localhost:6379/0
API_HOST=0.0.0.0
API_PORT=8001
API_KEYS=key1,key2            # Comma-separated (empty = dev mode, no auth)
CORS_ORIGINS=*                 # Comma-separated origins
REQUEST_TIMEOUT_SECONDS=30.0
```

### Data Sources

```bash
# Twitter
TWITTER_BEARER_TOKEN=...
TWITTER_API_KEY=...
TWITTER_API_SECRET=...

# Reddit
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...

# News APIs (single key)
FINNHUB_API_KEY=...
NEWSAPI_API_KEY=...
ALPHA_VANTAGE_API_KEY=...

# News APIs (comma-separated for key rotation)
NEWSFILTER_API_KEYS=key1,key2,key3
MARKETAUX_API_KEYS=...
FINLIGHT_API_KEYS=...

# Substack
SUBSTACK_COOKIE=...

# Sotwe (Twitter fallback)
SOTWE_ENABLED=true
SOTWE_USERNAMES=user1,user2
SOTWE_RATE_LIMIT=10
```

### ML Models

```bash
# Embedding
EMBEDDING_MODEL_NAME=ProsusAI/finbert
MINILM_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_USE_FP16=true
EMBEDDING_DEVICE=auto            # auto | cpu | cuda | mps
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_TTL_HOURS=168

# Sentiment
SENTIMENT_MODEL_NAME=ProsusAI/finbert
SENTIMENT_BATCH_SIZE=16
SENTIMENT_USE_FP16=true
SENTIMENT_DEVICE=auto
SENTIMENT_CACHE_ENABLED=true
SENTIMENT_ENABLE_ENTITY_SENTIMENT=true
SENTIMENT_ENTITY_CONTEXT_WINDOW=100

# NER
NER_SPACY_MODEL=en_core_web_trf   # en_core_web_trf (accurate) or en_core_web_sm (fast)
NER_FUZZY_THRESHOLD=85
NER_ENABLE_COREFERENCE=true
NER_CONFIDENCE_THRESHOLD=0.5
COREF_MIN_LENGTH=500               # Skip coreference for short content
COREF_DEVICE=cpu
```

### Processing

```bash
SPAM_THRESHOLD=0.7
DUPLICATE_THRESHOLD=0.85
```

### Worker Resilience

```bash
WORKER_MAX_CONSECUTIVE_FAILURES=10
WORKER_BACKOFF_BASE_DELAY=2.0
WORKER_BACKOFF_MAX_DELAY=120.0
```

### Observability

```bash
TRACING_ENABLED=true
OTEL_SERVICE_NAME=news-tracker
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### WebSocket Alerts

```bash
WS_ALERTS_ENABLED=true
WS_ALERTS_MAX_CONNECTIONS=100
WS_ALERTS_HEARTBEAT_SECONDS=30
```

Each opt-in feature has its own `ENV_PREFIX_*` namespace for fine-grained configuration. See `src/config/settings.py` for the full reference.

## Development

```bash
# Install all dependencies (including test/dev)
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_embedding/test_service.py -v

# Skip integration tests (requires running services)
uv run pytest tests/ -v -m "not integration"

# Syntax check a module
python3 -m py_compile src/module/file.py

# Build Docker image
docker build -t news-tracker .
```

### Frontend Development

The frontend requires Node.js 18+ (22 recommended).

```bash
cd frontend
npm install                      # Install dependencies
npx vite                         # Dev server (default :5173)
npx tsc --noEmit                 # Type check (zero output = success)
npx eslint .                     # Lint
npx vite build                   # Production build
```

### Project Structure

Tests mirror the source tree: `tests/test_<module>/test_<file>.py`. Every service module follows the convention of `config.py`, `service.py`, and `__init__.py` with `__all__` exports.

## License

MIT
