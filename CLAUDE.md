# CLAUDE.md

## Build & Development

```bash
uv sync --extra dev                       # Install deps (always use --extra dev)
uv run pytest tests/ -v                   # Run all tests
uv run pytest tests/test_X/test_Y.py -v   # Single test file
uv run pytest tests/ -v -m "not integration"  # Skip integration tests
python3 -m py_compile src/module/file.py  # Syntax check

docker compose up -d                      # Start PostgreSQL, Redis, Prometheus, Grafana
uv run news-tracker health                # Check all dependencies
uv run news-tracker init-db               # Initialize database schema
uv run news-tracker run-once --mock       # Single ingestion cycle (mock data)
uv run news-tracker serve                 # Start FastAPI server
uv run news-tracker daily-clustering --date 2026-02-05  # Batch clustering
uv run news-tracker graph seed            # Seed causal graph
uv run news-tracker vector-search "query" --limit 10    # Semantic search
uv run news-tracker cleanup --days 90     # Remove old documents
```

Test markers: `@pytest.mark.performance` (benchmarks), `@pytest.mark.integration` (requires running services).

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
| Embedding | `src/embedding/` | `EmbeddingService` (FinBERT 768-dim + MiniLM 384-dim), `EmbeddingWorker` |
| Sentiment | `src/sentiment/` | `SentimentService` (ProsusAI/finbert), `SentimentWorker` |
| Clustering | `src/clustering/` | `BERTopicService` (batch), `ClusteringWorker` (real-time), `run_daily_clustering()` |
| Themes | `src/themes/` | `ThemeRepository`, `LifecycleClassifier`, `VolumeMetricsService`, `ThemeRankingService` |
| Graph | `src/graph/` | `GraphRepository` (recursive CTE), `CausalGraph`, `seed_data.py` |
| Alerts | `src/alerts/` | `AlertService`, `triggers.py` (stateless functions), `AlertRepository` |
| Backtest | `src/backtest/` | `PointInTimeService`, `PriceDataFeed`, `ModelVersionRepository`, `BacktestRunRepository` |
| Storage | `src/storage/` | `Database` (asyncpg), `DocumentRepository` |
| API | `src/api/` | FastAPI with `routes/` (embed, sentiment, search, themes, events, alerts, health) |

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
- **Recursive CTE Traversal**: Graph uses `WITH RECURSIVE` + cycle detection via `NOT node_id = ANY(path)`
- **Composite PK Edges**: `(source, target, relation)` allows multiple relation types between same nodes
- **Bidirectional Competition**: `competes_with` requires explicit A→B and B→A edges
- **Idempotent Upserts**: Edges, metrics, and seed data all use `ON CONFLICT DO UPDATE`
- **Soft Delete Themes**: `deleted_at` column enables point-in-time queries; `AND deleted_at IS NULL` on all live queries
- **Deterministic Model Versions**: `mv_{sha256(config_json)[:12]}` — idempotent, same config = same version ID
- **Point-in-Time Queries**: Filter on `fetched_at` (ingestion time), not `timestamp` (publication time), to prevent look-ahead bias

## Configuration

Settings in `src/config/settings.py` (Pydantic BaseSettings, env var overrides).

### Infrastructure
`DATABASE_URL`, `REDIS_URL`, `api_host` (0.0.0.0), `api_port` (8000), `api_keys` (comma-separated, empty = dev mode)

### Opt-In Features (all `false` by default)

| Feature | Flag | Env Prefix | Key Settings |
|---------|------|------------|--------------|
| NER | `ner_enabled` | `NER_*` | `ner_spacy_model` (en_core_web_trf), semantic linking |
| Keywords | `keywords_enabled` | `KEYWORDS_*` | `top_n` (10), `min_score` |
| Events | `events_enabled` | `EVENTS_*` | `min_confidence`, `max_events_per_doc` |
| Clustering | `clustering_enabled` | `CLUSTERING_*` | HDBSCAN/UMAP params, assignment thresholds |
| Volume Metrics | `volume_metrics_enabled` | `VOLUME_*` | decay, windows, thresholds, EMA spans |
| Ranking | `ranking_enabled` | `RANKING_*` | `default_strategy` (swing/position), tier percentiles |
| Graph | `graph_enabled` | `GRAPH_*` | `max_traversal_depth`, `default_confidence` |
| Alerts | `alerts_enabled` | `ALERTS_*` | dedup TTL, daily rate limits, trigger thresholds |
| Backtest | `backtest_enabled` | `BACKTEST_*` | price cache TTL, forward horizons, yfinance rate limit |

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
| POST | /search/similar | Semantic search with filters |
| GET | /themes | List (lifecycle_stage filter, pagination) |
| GET | /themes/ranked | Ranked by actionability (strategy, max_tier, limit) |
| GET | /themes/{id} | Detail (optional centroid via `?include_centroid=true`) |
| GET | /themes/{id}/documents | Documents (platform, authority filters) |
| GET | /themes/{id}/sentiment | Aggregated sentiment (exponential decay) |
| GET | /themes/{id}/metrics | Daily metrics time series |
| GET | /themes/{id}/events | Events via ticker overlap (dedup, investment_signal) |
| GET | /alerts | List (severity, trigger_type, theme_id, acknowledged) |
| GET | /health | Service status |

## Module Conventions

- Every service: `config.py`, `service.py`, `__init__.py` with `__all__` exports
- Tests mirror src: `tests/test_<module>/test_<file>.py`
- New features: add opt-in flag to `src/config/settings.py`, use `SettingsConfigDict(env_prefix="FEATURE_")`
- DB migrations: `src/storage/migrations/NNN_description.sql`
