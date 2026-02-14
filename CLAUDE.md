# CLAUDE.md

## Build & Development

```bash
uv sync --extra dev                       # Install deps (always use --extra dev)
uv run pytest tests/ -v                   # Run all tests
uv run pytest tests/test_X/test_Y.py -v   # Single test file
uv run pytest tests/ -v -m "not integration"  # Skip integration tests
python3 -m py_compile src/module/file.py  # Syntax check

docker compose up -d                      # Start PostgreSQL, Redis, Prometheus, Grafana, Jaeger, AlertManager + API
docker build -t news-tracker .            # Build Docker image
uv run news-tracker health                # Check all dependencies
uv run news-tracker init-db               # Initialize database schema
uv run news-tracker run-once --mock       # Single ingestion cycle (mock data)
uv run news-tracker serve                 # Start FastAPI server
uv run news-tracker daily-clustering --date 2026-02-05  # Batch clustering
uv run news-tracker graph seed            # Seed causal graph
uv run news-tracker vector-search "query" --limit 10    # Semantic search
uv run news-tracker cleanup --days 90     # Remove old documents
uv run news-tracker backtest run --start 2025-01-01 --end 2025-06-30  # Run backtest
uv run news-tracker backtest run --start 2025-01-01 --end 2025-06-30 --strategy position --horizon 20
uv run news-tracker drift check-quick     # Embedding drift only (hourly)
uv run news-tracker drift check-daily     # All 4 drift checks (daily)
uv run news-tracker drift report          # Weekly verbose report
```

Test markers: `@pytest.mark.performance` (benchmarks), `@pytest.mark.integration` (requires running services).

## Frontend (Node.js)

### Node.js Setup

System Node (`/usr/local/bin/node`) is v14 — **too old** for the frontend toolchain. Always use nvm Node v22:

```bash
# ✅ CORRECT: Prefix PATH with nvm node, then use local binaries
export PATH="/Users/admin/.nvm/versions/node/v22.18.0/bin:$PATH"
cd frontend && node_modules/.bin/tsc --noEmit    # Type check
cd frontend && node_modules/.bin/vite               # Dev server (:5151)

# ❌ WRONG: npx/npm from system PATH uses Node v14 and crashes
npx tsc --noEmit   # "Cannot find module 'node:path'"
```

### Frontend Commands

```bash
# Always from frontend/ with nvm Node on PATH
npm install                          # Install dependencies
node_modules/.bin/vite               # Dev server (default :5151)
node_modules/.bin/tsc --noEmit       # Type check (zero output = success)
node_modules/.bin/eslint .           # Lint
node_modules/.bin/vite build         # Production build
```

### Frontend Tips

- **Stack**: React 18 + TypeScript + Vite + Tailwind CSS + React Query + Zustand
- **Path alias**: `@/` maps to `frontend/src/` (configured in `tsconfig.json`)
- **API client**: `src/api/http.ts` — axios instance with `/api` baseURL, auth interceptor, correlation IDs
- **Query keys**: All in `src/api/queryKeys.ts` — use the factory functions, never hand-craft key arrays
- **Hooks pattern**: `src/api/hooks/use*.ts` — one hook per API endpoint, typed request/response interfaces co-located
- **Shared utilities**: `src/lib/constants.ts` (PLATFORMS, colors), `src/lib/formatters.ts` (timeAgo, pct, truncate, latency), `src/lib/utils.ts` (cn for Tailwind merge)
- **Domain components**: `src/components/domain/` — reusable cards/panels, always export a Skeleton variant for loading states
- **Pages**: `src/pages/` — one default export per route, lazy-loaded in `App.tsx`
- **Dark theme**: Always on (`<html class="dark">`), use `text-foreground`, `bg-card`, `border-border` etc.

## Architecture

Semiconductor news ingestion pipeline. Data flows through:

```
Adapters → Redis Streams → Processing → PostgreSQL + pgvector
                                ↓
              embedding_queue / sentiment_queue / clustering_queue
                                ↓
              EmbeddingWorker / SentimentWorker / ClusteringWorker
                                ↓
                           FastAPI REST API
```

### Layer Map

| Layer | Path | Key Classes |
|-------|------|-------------|
| Ingestion | `src/ingestion/` | `BaseAdapter` (extend for new platforms), `DocumentQueue`, `HTTPClient` |
| Processing | `src/ingestion/` + `src/services/` | `Preprocessor`, `SpamDetector`, `TickerExtractor`, `Deduplicator` |
| NER | `src/ner/` | `NERService` (spaCy + EntityRuler + rapidfuzz) |
| Keywords | `src/keywords/` | `KeywordsService` (TextRank via rapid-textrank) |
| Events | `src/event_extraction/` | `PatternExtractor` (regex SVO), `TimeNormalizer`, `EventThemeLinker` |
| Authority | `src/authority/` | `AuthorityService` (Bayesian scoring + time decay + probation), `AuthorityRepository`, `AuthorityProfile` |
| Embedding | `src/embedding/` | `EmbeddingService` (FinBERT 768-dim + MiniLM 384-dim), `EmbeddingWorker` |
| Sentiment | `src/sentiment/` | `SentimentService` (ProsusAI/finbert), `SentimentWorker` |
| Clustering | `src/clustering/` | `BERTopicService` (batch), `ClusteringWorker` (real-time), `run_daily_clustering()` |
| Themes | `src/themes/` | `ThemeRepository`, `LifecycleClassifier`, `VolumeMetricsService`, `ThemeRankingService` |
| Graph | `src/graph/` | `GraphRepository` (recursive CTE), `CausalGraph`, `SentimentPropagation` (BFS impact), `seed_data.py` |
| Alerts | `src/alerts/` | `AlertService`, `triggers.py` (stateless functions), `AlertRepository` |
| Notifications | `src/alerts/` | `NotificationDispatcher`, `WebhookChannel`, `SlackChannel`, `CircuitBreaker` |
| Backtest | `src/backtest/` | `BacktestEngine`, `BacktestMetrics`, `PointInTimeService`, `PriceDataFeed`, `BacktestRunRepository` |
| Visualization | `src/backtest/` | `BacktestVisualizer` (matplotlib charts: cumulative returns, drawdown, scatter, heatmap) |
| Scoring | `src/scoring/` | `CompellingnessService` (3-tier: rule→GPT→Claude), `LLMClient`, `GenericCircuitBreaker` |
| Security Master | `src/security_master/` | `SecurityMasterService` (cached DB-backed tickers), `SecurityMasterRepository` (pg_trgm fuzzy search), `Security` dataclass |
| Sources | `src/sources/` | `SourcesService` (TTL-cached per-platform lookups), `SourcesRepository` (CRUD + `get_active_by_platform()`), `Source` dataclass |
| Tracing | `src/observability/tracing.py` | `setup_tracing` (OTLP exporter), `traced` (span context manager), `inject_trace_context`/`extract_trace_context` (Redis Streams propagation), `add_trace_context` (structlog processor) |
| Monitoring | `src/monitoring/` | `DriftService` (4 checks: embedding KL, fragmentation, sentiment z-score, centroid stability), `DriftConfig`, `DriftReport` |
| Feedback | `src/feedback/` | `FeedbackRepository` (create, list_by_entity, get_stats), `Feedback` dataclass, `FeedbackConfig` |
| WS Alerts | `src/alerts/broadcaster.py` + `src/api/routes/ws_alerts.py` | `AlertBroadcaster` (Redis pub/sub fan-out to WebSocket clients), `/ws/alerts` endpoint |
| Storage | `src/storage/` | `Database` (asyncpg), `DocumentRepository` (includes entity queries: `list_entities`, `get_entity_detail`, `get_entity_sentiment`, `get_trending_entities`, `get_cooccurring_entities`, `merge_entity`) |
| Queues | `src/queues/` | `BaseRedisQueue` (XREADGROUP + XAUTOCLAIM), `ExponentialBackoff`, `QueueConfig` |
| API | `src/api/` | FastAPI with `routes/`, correlation ID middleware, `TimeoutMiddleware`, `rate_limit.py` (slowapi) |

### Data Schema

`NormalizedDocument` (`src/ingestion/schemas.py`) is the canonical schema: id, platform, content, author info, quality scores (spam, bot, authority, tickers, entities, keywords), engagement, embedding (768 + 384 dim), sentiment (JSONB), theme_ids.

## Critical Rules

### Rate Limiting MUST happen at I/O boundary

```python
# ✅ CORRECT: Rate limit before HTTP request in _fetch_raw()
async def _fetch_raw(self):
    for ticker in tickers:
        await self._rate_limiter.acquire()  # Before HTTP call
        response = await client.get(url, params={"ticker": ticker})
        for article in response.json():
            yield article  # Yields are free — no rate limit here

# ❌ WRONG: Rate limit per-item after yield (N× slowdown for batch APIs)
```

APIs return batches (50-100 items). Rate limiting per-item = N× slowdown.

### Config classes use Pydantic v2 style

```python
model_config = SettingsConfigDict(env_prefix="SERVICE_", ...)  # ✅
class Config: ...  # ❌ Never use this
```

### FastAPI route order matters

`/themes/ranked` MUST register before `/themes/{theme_id}` — otherwise "ranked" gets captured as a theme_id.

### Ranking z-score shift

`max(0, zscore + 2)` before `** alpha` prevents complex numbers from negative base with fractional exponent.

## Key Patterns

- **Worker Supervisor Loop**: `start()` wraps `_connect_dependencies()` + `_process_loop()` in a retry loop with `ExponentialBackoff`; `_cleanup()` between retries
- **Queue Exponential Backoff**: `consume()` uses `ExponentialBackoff` from `QueueConfig` on error, with dedicated `_reconnect()` for Redis connection failures
- **Adapter Pattern**: Extend `BaseAdapter`, implement `_fetch_raw()` (with rate limiting) and `_transform()`
- **Lazy Model Loading**: All ML services defer model loading until first use
- **Redis SET NX Idempotency**: Prevents reprocessing in ClusteringWorker and alert dedup
- **Atomic Array Merge**: `ARRAY(SELECT DISTINCT unnest(a || $2))` for concurrent theme assignment
- **EMA Centroid Update**: `(1 - lr) * old + lr * new` adapts themes with O(1) cost
- **Three-Tier Assignment**: Strong (centroid update) / Weak (assign only) / New candidate
- **Batch Cosine Similarity**: `emb_norm @ centroid_norm.T` — numpy matrix multiply, no Python loops
- **Dual-Path Clustering**: Real-time (pgvector HNSW per-doc) vs Batch (numpy per-day)
- **Stateless Services**: LifecycleClassifier, VolumeMetricsService, trigger functions — no instance state, trivially testable
- **Phase-Resilient Errors**: Each daily clustering phase has independent try/except
- **Content-Hash Caching**: `emb:{model}:{sha256}` keys for embedding and sentiment cache
- **Tiered Model Selection**: MiniLM for Twitter + short (<300 chars), FinBERT otherwise
- **DB vs Clustering Schemas**: `Theme` (persistence) vs `ThemeCluster` (in-memory BERTopic)
- **Deterministic Theme IDs**: `theme_{sha256(sorted_topic_words)[:12]}`
- **EntityRuler Before NER**: spaCy EntityRuler runs before statistical NER for domain pattern priority
- **Pre-NER Coref Resolution**: fastcoref resolves text ("The chipmaker" → "Samsung") BEFORE NER runs, gated by `coref_min_length` (500 chars) to skip short content
- **Bayesian Authority Smoothing**: `(correct + alpha) / (total + alpha + beta)` Beta prior prevents new sources with 1/1 accuracy from outranking established sources
- **Authority Probation Ramp**: `min(1.0, days_active / 30)` linearly gates new sources over their first 30 days
- **Tier-Based Base Weight**: anonymous (1.0) / verified (5.0) / research (10.0) multiplied into authority formula
- **Recursive CTE Traversal**: Graph uses `WITH RECURSIVE` + cycle detection via `NOT node_id = ANY(path)`
- **Composite PK Edges**: `(source, target, relation)` allows multiple relation types between same nodes
- **Bidirectional Competition**: `competes_with` requires explicit A→B and B→A edges
- **Idempotent Upserts**: Edges, metrics, and seed data all use `ON CONFLICT DO UPDATE`
- **Soft Delete Themes**: `deleted_at` column enables point-in-time queries; `AND deleted_at IS NULL` on all live queries
- **Deterministic Model Versions**: `mv_{sha256(config_json)[:12]}` — idempotent, same config = same version ID
- **Point-in-Time Queries**: Filter on `fetched_at` (ingestion time), not `timestamp` (publication time), to prevent look-ahead bias
- **Circuit Breaker Decorator**: `CircuitBreaker` wraps any `NotificationChannel` transparently (CLOSED→OPEN→HALF_OPEN→CLOSED)
- **Graceful Notification Degradation**: Dispatcher failures never block alert persistence; Redis fallback queue for retries
- **Edge-Type-Aware BFS**: Sentiment propagation multiplies edge weights (with sign) per hop; `competes_with` (-0.3) flips impact direction
- **First Arrival Wins**: Propagation uses shallowest path to determine impact — deeper paths to already-reached nodes are skipped
- **Transparent Cache Injection**: `init_security_master()` populates module-level caches in `tickers.py`; existing consumers use cached DB data with zero code changes
- **Composite PK Securities**: `(ticker, exchange)` identifies securities across exchanges (e.g., Samsung on KRX vs US ADR)
- **pg_trgm Fuzzy Search**: GIN trigram index on `securities.name` enables typo-tolerant company lookup via `similarity()`
- **Redis Pub/Sub Fan-Out**: `alerts:broadcast` channel for WS alert push; each uvicorn worker subscribes independently
- **WS Auth via Query Param**: `?api_key=...` since browsers can't set custom headers on WebSocket upgrade
- **W3C Traceparent in Redis Streams**: `traceparent` field injected on XADD, extracted on XREADGROUP via `BaseRedisQueue._trace_fields()` — propagates trace context across worker boundaries
- **SimpleSpanProcessor for Tests**: Custom exporter path uses synchronous export; production uses `BatchSpanProcessor` for efficiency
- **Correlation ID Middleware**: `X-Request-ID` / `X-Correlation-ID` header → structlog contextvars + OTel span attribute; auto-generated UUID if absent
- **Health Endpoint Three-Tier Status**: `unhealthy` (DB down) → `degraded` (Redis down) → `healthy` (all green); `ComponentHealth` model per subsystem

## Docker Compose Services

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL + pgvector | 5432 | Document storage, vector search |
| Redis | 6379 | Queues, caching, pub/sub |
| Prometheus | 9090 | Metrics scraping + alert rules |
| AlertManager | 9093 | Alert routing (webhook placeholder) |
| Grafana | 3000 | Auto-provisioned dashboards (admin/admin) |
| Jaeger | 16686 (UI), 4317 (OTLP gRPC) | Distributed tracing |
| news-tracker-api | 8001 | Application API (built from Dockerfile) |

## Configuration

Settings in `src/config/settings.py` (Pydantic BaseSettings, env var overrides).

### Infrastructure
`DATABASE_URL`, `REDIS_URL`, `api_host` (0.0.0.0), `api_port` (8001), `api_keys` (comma-separated, empty = dev mode), `cors_origins` (`*`, comma-separated), `cors_allow_credentials` (true), `request_timeout_seconds` (30.0), `worker_max_consecutive_failures` (10), `worker_backoff_base_delay` (2.0), `worker_backoff_max_delay` (120.0)

### Opt-In Features (all `false` by default)

| Feature | Flag | Env Prefix | Key Settings |
|---------|------|------------|--------------|
| NER | `ner_enabled` | `NER_*` | `ner_spacy_model` (en_core_web_trf), semantic linking, `coref_min_length` (500), `coref_device` (cpu) |
| Keywords | `keywords_enabled` | `KEYWORDS_*` | `top_n` (10), `min_score` |
| Events | `events_enabled` | `EVENTS_*` | `min_confidence`, `max_events_per_doc` |
| Clustering | `clustering_enabled` | `CLUSTERING_*` | HDBSCAN/UMAP params, assignment thresholds |
| Volume Metrics | `volume_metrics_enabled` | `VOLUME_*` | decay, windows, thresholds, EMA spans |
| Ranking | `ranking_enabled` | `RANKING_*` | `default_strategy` (swing/position), tier percentiles |
| Graph | `graph_enabled` | `GRAPH_*` | `max_traversal_depth`, `default_confidence` |
| Propagation | `propagation_enabled` | `GRAPH_*` | `propagation_default_decay` (0.7), `propagation_max_depth` (3), `propagation_min_impact`, edge type weights |
| Alerts | `alerts_enabled` | `ALERTS_*` | dedup TTL, daily rate limits, trigger thresholds |
| Notifications | `notifications_enabled` | `NOTIFICATIONS_*` | retry attempts/delays, circuit breaker threshold/recovery, queue TTL |
| Backtest | `backtest_enabled` | `BACKTEST_*` | price cache TTL, forward horizons, yfinance rate limit |
| Scoring | `scoring_enabled` | `SCORING_*` | LLM API keys, tier thresholds, budget caps, circuit breaker, cache TTL |
| Security Master | `security_master_enabled` | `SECURITY_MASTER_*` | `cache_ttl_seconds` (300), `fuzzy_threshold` (0.3), `seed_on_init` (True) |
| Sources | `sources_enabled` | `SOURCES_*` | `cache_ttl_seconds` (300), `seed_on_init` (True) |
| Feedback | `feedback_enabled` | `FEEDBACK_*` | `max_comment_length` (2000) |
| Authority | `authority_enabled` | `AUTHORITY_*` | `prior_alpha` (2.0), `prior_beta` (5.0), `decay_lambda` (0.01), `probation_days` (30), tier weights |
| Drift Detection | `drift_enabled` | `DRIFT_*` | KL thresholds, fragmentation limits, sentiment z-score, stability cosine distance |
| Tracing | `tracing_enabled` | (top-level) | `otel_service_name` (news-tracker), `otel_exporter_otlp_endpoint` (OTLP gRPC endpoint) |
| WS Alerts | `ws_alerts_enabled` | (top-level) | `ws_alerts_max_connections` (100), `ws_alerts_heartbeat_seconds` (30) |
| Rate Limiting | `rate_limit_enabled` | (top-level) | `rate_limit_default` (60/minute), `rate_limit_embed` (30/min), `rate_limit_sentiment` (30/min), `rate_limit_search` (60/min), `rate_limit_graph` (30/min), `rate_limit_entities` (60/min), `rate_limit_admin` (30/min) |

### Other Config
- Tickers/companies: `src/config/tickers.py`
- Twitter accounts: `src/config/twitter_accounts.py`
- Sotwe: `SOTWE_ENABLED` (true), `SOTWE_USERNAMES`, `SOTWE_RATE_LIMIT` (10)
- News API keys: `NEWSFILTER_API_KEYS`, `MARKETAUX_API_KEYS`, `FINLIGHT_API_KEYS` (comma-separated for rotation)
- Processing: `spam_threshold` (0.7), `duplicate_threshold` (0.85)
- Embedding: `embedding_model_name` (ProsusAI/finbert), `minilm_model_name` (all-MiniLM-L6-v2), cache TTL 168h
- Sentiment: `sentiment_model_name` (ProsusAI/finbert), entity context window 100 chars

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /embed | Auto model selection embedding |
| POST | /sentiment | Document + optional entity sentiment |
| POST | /ner | Batch NER entity extraction (feature-gated) |
| POST | /keywords | Batch keyword extraction via TextRank (feature-gated) |
| POST | /events/extract | Single-text SVO event extraction (feature-gated) |
| POST | /search/similar | Semantic search with filters |
| GET | /themes | List (lifecycle_stage filter, pagination) |
| GET | /themes/ranked | Ranked by actionability (strategy, max_tier, limit) |
| GET | /themes/{id} | Detail (optional centroid via `?include_centroid=true`) |
| GET | /themes/{id}/documents | Documents (platform, authority filters) |
| GET | /themes/{id}/sentiment | Aggregated sentiment (exponential decay) |
| GET | /themes/{id}/metrics | Daily metrics time series |
| GET | /themes/{id}/events | Events via ticker overlap (dedup, investment_signal) |
| GET | /documents | List with filters (platform, ticker, q, sort, pagination) |
| GET | /documents/stats | Aggregate stats (counts, coverage, date range) |
| GET | /documents/{id} | Full detail (content, entities, keywords, events) |
| GET | /alerts | List (severity, trigger_type, theme_id, acknowledged) |
| PATCH | /alerts/{id}/acknowledge | Mark alert as acknowledged |
| GET | /graph/nodes | List graph nodes (optional node_type filter) |
| GET | /graph/nodes/{id}/subgraph | Subgraph around a node (depth param) |
| POST | /graph/propagate | Sentiment propagation through causal graph |
| POST | /feedback | Submit quality rating for theme/alert/document |
| GET | /feedback/stats | Aggregated feedback statistics by entity type |
| WS | /ws/alerts | Real-time alert stream (severity, theme_id, api_key query params) |
| GET | /entities | List entities with search/filter/sort (feature-gated: `ner_enabled`) |
| GET | /entities/stats | Aggregate entity stats (total, by_type, docs with entities) |
| GET | /entities/trending | Entities with mention spikes (recent vs baseline) |
| GET | /entities/{type}/{normalized} | Entity detail (stats, platforms, graph link) |
| GET | /entities/{type}/{normalized}/documents | Documents mentioning entity |
| GET | /entities/{type}/{normalized}/cooccurrence | Co-occurring entities (Jaccard) |
| GET | /entities/{type}/{normalized}/sentiment | Aggregate sentiment + trend |
| POST | /entities/{type}/{normalized}/merge | Merge entity into another |
| GET | /securities | List securities with filters (feature-gated: `security_master_enabled`) |
| POST | /securities | Create a new security |
| PUT | /securities/{ticker}/{exchange} | Update a security |
| DELETE | /securities/{ticker}/{exchange} | Deactivate (soft delete) a security |
| GET | /sources | List sources with filters (feature-gated: `sources_enabled`) |
| POST | /sources | Create a new source |
| POST | /sources/bulk | Bulk-create sources for a single platform (max 500) |
| PUT | /sources/{platform}/{identifier} | Update a source |
| DELETE | /sources/{platform}/{identifier} | Deactivate (soft delete) a source |
| GET | /health | Service status |

## Module Conventions

- Every service: `config.py`, `service.py`, `__init__.py` with `__all__` exports
- Tests mirror src: `tests/test_<module>/test_<file>.py`
- New features: add opt-in flag to `src/config/settings.py`, use `SettingsConfigDict(env_prefix="FEATURE_")`
- DB migrations: `migrations/NNN_description.sql`
