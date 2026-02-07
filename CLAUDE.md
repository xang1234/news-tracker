# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies (uses uv for package management)
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_ingestion/test_preprocessing.py::TestTickerExtractor::test_cashtag_extraction -v

# Syntax check without running
python3 -m py_compile src/ingestion/base_adapter.py

# Start infrastructure (PostgreSQL, Redis, Prometheus, Grafana)
docker compose up -d

# CLI commands
uv run news-tracker ingest --mock      # Run ingestion with mock data
uv run news-tracker process            # Run processing service
uv run news-tracker worker --mock      # Run both services together
uv run news-tracker health             # Check all dependencies
uv run news-tracker init-db            # Initialize database schema
uv run news-tracker run-once --mock    # Single cycle for testing
uv run news-tracker run-once --mock --with-sentiment  # Include sentiment analysis
uv run news-tracker serve              # Start embedding API server
uv run news-tracker sentiment-worker   # Run sentiment analysis worker
uv run news-tracker clustering-worker  # Run clustering worker for theme assignment
uv run news-tracker daily-clustering                   # Daily batch clustering (today)
uv run news-tracker daily-clustering --date 2026-02-05 # Batch clustering for specific date
uv run news-tracker daily-clustering --dry-run         # Preview without running
uv run news-tracker cluster fit --days 30              # Discover themes via BERTopic
uv run news-tracker cluster run --date 2026-02-05      # Daily clustering (same as daily-clustering)
uv run news-tracker cluster backfill --start 2026-01-01 --end 2026-01-31  # Backfill date range
uv run news-tracker cluster merge --dry-run            # Preview theme merges
uv run news-tracker cluster status                     # Show theme summary
uv run news-tracker cluster recompute-centroids        # Recompute all centroids from docs
uv run news-tracker graph seed                         # Seed causal graph with semiconductor supply chain
uv run news-tracker vector-search "query" --limit 10  # Semantic search
uv run news-tracker cleanup --days 90  # Remove old documents (storage management)

# NER testing
uv run pytest tests/test_ner/ -v           # Run all NER tests
uv run pytest tests/test_ner/ -v -m "not integration"  # Skip integration tests (no spaCy model needed)
python -m spacy download en_core_web_trf   # Download transformer model for NER
python -m spacy download en_core_web_sm    # Download smaller model (faster, lower accuracy)
```

## Architecture Overview

This is a multi-platform financial data ingestion pipeline for tracking semiconductor news. Data flows through four stages:

```
Adapters → Redis Streams → Processing Pipeline → PostgreSQL
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            embedding_queue  sentiment_queue   (direct store)
            (Redis Stream)   (Redis Stream)
                    │               │
                    ▼               ▼
            EmbeddingWorker  SentimentWorker
                    │               │
            ┌───────┴───────┐       │
            ▼               ▼       ▼
    FinBERT (768-dim)  MiniLM   FinBERT Sentiment
    Long/financial     (384)    (pos/neg/neutral)
                    │               │
                    └───────┬───────┘
                            ▼
                    PostgreSQL + pgvector
                    (embedding, sentiment JSONB)

                            ┌─────────────┐
                            │ FastAPI     │
                            │ /embed      │
                            │ /sentiment  │◄── External clients
                            │ /search     │
                            │ /health     │
                            └─────────────┘
```

### Key Components

**Ingestion Layer** (`src/ingestion/`):
- `BaseAdapter`: Abstract base with error handling and preprocessing. Subclasses implement `_fetch_raw()` and `_transform()`. **Note**: Subclasses must call `self._rate_limiter.acquire()` before each HTTP request in `_fetch_raw()`.
- Platform adapters: `TwitterAdapter`, `RedditAdapter`, `SubstackAdapter`, `NewsAdapter`, `MockAdapter`
- `TwitterAdapter`: Uses Twitter API v2 when `TWITTER_BEARER_TOKEN` is set, falls back to Sotwe.com scraping otherwise
- `SotweClient`: Sotwe.com scraper using `curl_cffi` (browser impersonation) and Node.js (NUXT JS parsing)
- `DocumentQueue`: Redis Streams wrapper with consumer groups, DLQ support
- `HTTPClient`: Async HTTP client with exponential backoff, retry on 429/5xx, and API key rotation (`http_client.py`)
- All adapters output `NormalizedDocument` (defined in `schemas.py`)

**Processing Layer** (`src/ingestion/` + `src/services/`):
- `Preprocessor`: Orchestrates spam detection, bot detection, ticker extraction, NER (optional)
- `SpamDetector`: Rule-based scoring (0.0-1.0) with ML-ready signal structure
- `TickerExtractor`: Cashtags, company names, fuzzy matching against `src/config/tickers.py`
- `Deduplicator`: MinHash LSH with 3-word shingles, configurable threshold

**NER Layer** (`src/ner/`):
- `NERConfig`: Pydantic settings for spaCy model, fuzzy threshold, coreference, batch size, semantic linking
- `FinancialEntity`: Dataclass for extracted entities with type, normalized form, confidence, metadata
- `NERService`: Named entity recognition using spaCy + EntityRuler patterns + fastcoref
- `patterns/`: JSONL files for domain-specific EntityRuler patterns (companies, products, technologies, metrics)
- Entity types: `TICKER`, `COMPANY`, `PRODUCT`, `TECHNOLOGY`, `METRIC`
- Lazy model loading, fuzzy matching via rapidfuzz, optional coreference resolution
- `link_entities_to_theme_semantic()`: Embedding-based disambiguation using cosine similarity (requires EmbeddingService)

**Keywords Layer** (`src/keywords/`):
- `KeywordsConfig`: Pydantic settings for top_n, language, min_score, max_text_length
- `ExtractedKeyword`: Dataclass for extracted keywords with text, score, rank, lemma, count, metadata
- `KeywordsService`: TextRank-based keyword extraction using rapid-textrank library
- Lazy model loading, graceful error handling, opt-in activation via `KEYWORDS_ENABLED=true`

**Event Extraction Layer** (`src/event_extraction/`):
- `EventExtractionConfig`: Pydantic settings (prefix `EVENTS_`) for extractor_version, min_confidence, max_events_per_doc
- `EventType`: Literal type for event categories: `capacity_expansion`, `capacity_constraint`, `product_launch`, `product_delay`, `price_change`, `guidance_change`
- `EventRecord`: Dataclass for extracted events with SVO structure (actor, action, object), time_ref, quantity, tickers, confidence, span offsets
- `PatternExtractor`: Regex-based event extractor with lazy-compiled patterns, named capture groups, ticker context linking, confidence scoring
- `TimeNormalizer`: Stateless normalizer for temporal references (Q1-Q4, H1-H2, relative refs, month names)
- Opt-in activation via `EVENTS_ENABLED=true`, follows NER/Keywords service pattern
- DB: `events` table with B-tree on event_type/doc_id/created_at, GIN on tickers; `events_extracted` JSONB column on documents
- `EventThemeLinker`: Stateless linker (static methods) for associating events with themes via ticker overlap
  - `link_events_to_theme(events, theme)`: Filters events by ticker set intersection with theme's `top_tickers`, adds `theme_id`
  - `deduplicate_events(events)`: Composite key `(actor, action, object, time_ref)`, keeps earliest, tracks `source_doc_ids`, +0.05 confidence per source (capped at 1.0)
- `ThemeWithEvents`: Summary dataclass with `event_counts`, `investment_signal()` → `supply_increasing | supply_decreasing | product_momentum | product_risk | None`

**Causal Graph Layer** (`src/graph/`):
- `GraphConfig`: Pydantic settings (prefix `GRAPH_`) for max_traversal_depth, default_confidence
- `CausalNode`: Dataclass for graph nodes with node_type (ticker, theme, technology) and metadata
- `CausalEdge`: Dataclass for directed edges with relation type, confidence, source_doc_ids provenance
- `GraphRepository`: asyncpg CRUD + recursive CTE traversals (downstream, upstream, path finding, subgraph extraction)
- `CausalGraph`: High-level service with depth clamping, config-driven defaults, ensure_node convenience
- Node types: `ticker`, `theme`, `technology`
- Relation types: `depends_on`, `supplies_to`, `competes_with`, `drives`, `blocks`
- Composite PK on edges: `(source, target, relation)` — same nodes can have multiple relationship types
- Recursive CTEs with cycle detection via path array (`NOT node_id = ANY(path)`)
- `get_downstream(node, max_depth)` / `get_upstream(node, max_depth)`: BFS traversal returning `(node_id, depth)` tuples
- `find_path(source, target, max_depth)`: Shortest path via recursive CTE with `ORDER BY depth LIMIT 1`
- `get_subgraph(node, depth)`: Extracts local neighborhood with both nodes and edges
- Idempotent upsert for edges with `source_doc_ids` array merge via `DISTINCT unnest()`
- ON DELETE CASCADE from nodes to edges
- Opt-in activation via `graph_enabled` in settings.py
- DB: migration `005_add_causal_graph_tables.sql` with B-tree indexes on edges(source) and edges(target)
- `seed_data.py`: Static semiconductor supply chain seed with 51 nodes (30 tickers, 13 technologies, 8 themes) and 149 edges
- `seed_graph(database)`: Async function to populate graph, idempotent via ON CONFLICT upserts
- `SEED_VERSION`: Integer constant for tracking seed data revisions
- Node IDs: tickers use exchange symbols (`NVDA`, `TSM`), non-US companies use readable IDs (`SK_HYNIX`, `SAMSUNG`)
- Edge categories: foundry supply, equipment supply, memory supply, EDA/IP supply, competition, technology deps, demand drivers
- CLI: `news-tracker graph seed` command for one-step graph population

**Clustering Layer** (`src/clustering/`):
- `ClusteringConfig`: Pydantic settings for UMAP, HDBSCAN, c-TF-IDF, assignment thresholds, Redis queue
- `ClusteringQueue`: Redis Streams wrapper for clustering jobs (follows `BaseRedisQueue[ClusteringJob]` pattern)
- `ClusteringJob`: Dataclass with `document_id`, `embedding_model`, `message_id`, `retry_count`
- `ThemeCluster`: Dataclass for discovered themes with deterministic IDs, centroid embeddings, topic words, serialization
- `BERTopicService`: Sync fit() runs UMAP → HDBSCAN → c-TF-IDF on pre-computed FinBERT embeddings to discover themes
- `BERTopicService.transform()`: Incremental assignment of new documents to existing themes via cosine similarity against centroids
  - Three-tier: strong (>= assign threshold, EMA centroid update), weak (>= new threshold, no update), new candidate (buffered)
  - `_new_theme_candidates`: List of `(doc_id, embedding)` pairs for documents below similarity_threshold_new
  - `updated_at` field on ThemeCluster tracks when centroid was last updated via EMA
- `BERTopicService.merge_similar_themes()`: Greedy pairwise merge of themes whose centroids exceed `similarity_threshold_merge` (0.85)
  - Survivor = larger `document_count`, gets weighted centroid, combined topic words, merged doc IDs
  - Re-keys `_themes` dict with regenerated theme IDs after merge
  - Returns `[(merged_from_id, merged_into_id)]` pairs
- `BERTopicService.check_new_themes()`: Detects emerging themes from outlier candidate documents
  - Runs lightweight HDBSCAN on candidate embeddings (no UMAP — pool too small)
  - Overlap check: skips clusters whose centroid is too similar to existing themes
  - TF-IDF keyword extraction via `_extract_keywords_tfidf()` for topic representation
  - New themes get `metadata={"lifecycle_stage": "emerging"}`
  - Clears processed doc_ids from `_new_theme_candidates` buffer
- Theme IDs: `theme_{sha256(sorted_topic_words)[:12]}` for cross-run stability
- Outlier documents (BERTopic topic -1) excluded from theme assignments
- Deferred imports for heavy dependencies (bertopic, hdbscan, umap-learn)
- Configurable via `CLUSTERING_*` environment variables (e.g., `CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE=20`)
- `clustering_enabled` (false) in settings.py for opt-in activation
- `ClusteringWorker`: Real-time per-document theme assignment via pgvector HNSW (unlike batch BERTopicService)
  - Consumes from `clustering_queue` Redis Stream, finds similar centroids via `ThemeRepository.find_similar()`
  - Assigns document `theme_ids`, EMA centroid update, atomic `document_count + 1`
  - Idempotency: `idempotent:cluster:{doc_id}:{model}` Redis SET NX with 7-day TTL
  - Skips MiniLM-only documents (theme centroids are 768-dim FinBERT)
  - CLI: `news-tracker clustering-worker [--batch-size N] [--metrics-port 8002]`
- `DailyClusteringResult`: Dataclass summarizing a batch run (counts, errors, elapsed time)
- `run_daily_clustering(database, target_date, config)`: Offline batch pipeline (10 phases):
  1. Fetch docs via `get_with_embeddings_since()` (lightweight projection, 6 fields)
  2. Batch cosine similarity via numpy matrix multiply `emb_norm @ centroid_norm.T`
  3. Three-tier assignment: strong (EMA centroid update), weak (assign only), unassigned
  4. Persist assignments via `update_themes()` atomic array merge
  5. Detect emerging themes from unassigned candidates via `BERTopicService.check_new_themes()`
  6. Compute daily metrics (sentiment_score, avg_authority, bullish_ratio) via `ThemeMetrics`
  7. Weekly Monday merge via `BERTopicService.merge_similar_themes()` with DB cleanup
  - Helper functions: `_batch_cosine_similarity()`, `_theme_to_cluster()`, `_cluster_to_theme()`, `_aggregate_sentiment_metrics()`
  - CLI: `news-tracker daily-clustering [--date YYYY-MM-DD] [--dry-run]`
  - Designed for cron: `0 4 * * * news-tracker daily-clustering`

**Themes Layer** (`src/themes/`):
- `Theme`: Dataclass mapping 1:1 to the `themes` DB table (distinct from in-memory `ThemeCluster`)
- `ThemeMetrics`: Dataclass mapping 1:1 to the `theme_metrics` daily time-series table
- `ThemeRepository`: CRUD, vector search, batch centroid, and metrics operations with pgvector centroid storage
- `VALID_LIFECYCLE_STAGES`: `{"emerging", "accelerating", "mature", "fading"}`
- `create()`, `get_by_id()`, `get_all()` with optional lifecycle_stage filter, `update()` with field allowlist, `delete()`
- `update_centroid()`: Dedicated fast-path for centroid updates (fixed SQL, no JSONB, no RETURNING)
- `find_similar(centroid, limit, threshold)`: HNSW cosine similarity search on theme centroids, returns `List[Tuple[Theme, float]]`
- `get_centroids_batch(theme_ids)`: Bulk centroid fetch with in-memory TTL cache (600s default), cache invalidated on update/delete
- `add_metrics(ThemeMetrics)`: Idempotent upsert of daily metrics (ON CONFLICT DO UPDATE on `(theme_id, date)`)
- `get_metrics_range(theme_id, start, end)`: Time-series query ordered by date ascending for trend computation
- Module-level helpers: `_row_to_theme()`, `_row_to_metrics()`, `_parse_centroid()`, `_centroid_to_pgvector()`
- JSONB fields (`top_entities`, `metadata`) serialized with `json.dumps()` on write, `json.loads()` on read
- TEXT[] fields (`top_keywords`, `top_tickers`) passed as Python lists (asyncpg handles natively)
- DB trigger handles `updated_at` — no explicit SET needed
- `LifecycleClassifier`: Rule-based lifecycle stage classification using 3-day sliding window of ThemeMetrics
  - `classify(theme, metrics_history)` → `(stage, confidence)` with normalized least-squares velocity/volume trends
  - `detect_transition(theme, new_stage, confidence)` → `Optional[LifecycleTransition]` for stage changes
  - `_compute_trend(values)`: Least-squares slope normalized by mean for scale-independent growth rate
  - Classification cascade: emerging (low docs + positive velocity) → accelerating (high velocity trend) → fading (negative velocity) → mature (default)
  - Confidence scores: 0.5 for insufficient data, 0.6-1.0 for confident classifications
- `LifecycleTransition`: Dataclass for detected stage changes with alertability
  - `is_alertable` / `alert_message`: Properties checking against `ALERTABLE_TRANSITIONS` lookup table
  - Key transitions: emerging→accelerating ("gaining momentum"), accelerating→mature ("peaking"), *→fading ("losing momentum")
- Integrated into `run_daily_clustering()` as Phase 11 after metrics computation
- `VolumeMetricsConfig`: Pydantic settings (prefix `VOLUME_`) for decay, windows, thresholds, EMA spans
- `VolumeMetricsService`: Stateless volume metrics computation (follows LifecycleClassifier pattern)
  - `compute_weighted_volume(docs, reference_time)`: platform weight * recency decay * authority
  - `compute_volume_zscore(current, history)`: (current - mean) / std with min_history_days guard
  - `compute_velocity(zscores)`: short EMA - long EMA for momentum detection
  - `compute_acceleration(velocities)`: delta of last two velocity values
  - `detect_volume_anomaly(zscore)` → `"surge" | "collapse" | None` via threshold comparison
  - `compute_for_theme(theme_id, target_date)`: async orchestrator fetching docs + history, returns ThemeMetrics
  - Duck-typed doc input: accepts any object with `.timestamp`, `.platform`, `.authority_score`
- `DEFAULT_PLATFORM_WEIGHTS`: `{"twitter": 1.0, "reddit": 5.0, "news": 20.0, "substack": 100.0}`
- `volume_metrics_enabled` (false) in settings.py for opt-in activation
- `ThemeRankingService`: Stateless theme ranking engine for trading actionability (follows LifecycleClassifier/VolumeMetricsService pattern)
  - `RankingConfig`: Pydantic settings (prefix `RANKING_`) for default_strategy, tier percentiles, min_zscore, compellingness fallback
  - `RankedTheme`: Dataclass with theme_id, theme ref, score, tier (1/2/3), components breakdown
  - `RankingStrategy`: Literal["swing", "position"] for strategy selection
  - `compute_score(theme, metrics, strategy)`: Core formula `(volume_component ** alpha) * (compellingness ** beta) * lifecycle_multiplier`
    - volume_component: `max(0, zscore + 2) ** alpha` — shift prevents negative base with fractional exponent
    - compellingness: `metadata.get("compellingness", 5.0) ** beta` — future LLM scorer fills this
    - lifecycle_multiplier: `{"emerging": 1.5, "accelerating": 1.2, "mature": 0.8, "fading": 0.3}`
  - `rank_themes(themes, metrics_map, strategy)`: Score all, sort descending, assign tiers, filter by min_score
  - `_assign_tiers(ranked, metrics_map)`: Tier 1 = top 5% AND (zscore >= 2.0 OR accelerating), Tier 2 = top 20%, Tier 3 = rest
  - `get_actionable(strategy, max_tier)`: Async orchestrator fetching themes + latest metrics from ThemeRepository
  - `STRATEGY_CONFIGS`: `{"swing": {"alpha": 0.6, "beta": 0.4}, "position": {"alpha": 0.4, "beta": 0.6}}`
- `ranking_enabled` (false) in settings.py for opt-in activation
- Ranking settings can be overridden via `RANKING_*` environment variables (e.g., `RANKING_DEFAULT_STRATEGY=position`)

**Storage Layer** (`src/storage/`):
- `Database`: asyncpg connection pool with transaction context managers
- `DocumentRepository`: CRUD operations, batch upserts, full-text search, similarity search, `update_themes()` array merge, `get_with_embeddings_since()` lightweight projection for batch clustering, `get_documents_by_theme()` with dynamic SQL filtering, `get_events_by_tickers()` GIN-indexed array overlap query on events table

**Embedding Layer** (`src/embedding/`):
- `EmbeddingConfig`: Pydantic settings for model, batching, caching, queue configuration
- `EmbeddingService`: Multi-model embedding (FinBERT 768-dim, MiniLM 384-dim) with lazy loading, chunking, caching
- `EmbeddingQueue`: Redis Streams wrapper for async embedding job processing
- `EmbeddingWorker`: Consumes queue, selects model by platform/length, generates embeddings, updates DB, enqueues to clustering_queue (if `clustering_enabled`)
- `ModelType`: Enum for type-safe model selection (FINBERT, MINILM)

**Sentiment Layer** (`src/sentiment/`):
- `SentimentConfig`: Pydantic settings for model, caching, entity sentiment, queue configuration
- `SentimentService`: FinBERT-based sentiment classification with lazy loading, caching, entity-level analysis, metrics instrumentation
- `SentimentQueue`: Redis Streams wrapper for async sentiment job processing
- `SentimentWorker`: Consumes queue, generates document/entity sentiment, updates DB, records metrics
- Label mapping: `{0: "positive", 1: "negative", 2: "neutral"}` (ProsusAI/finbert)
- Entity sentiment uses context windows around entity mentions for aspect-based analysis
- Prometheus metrics: `sentiment_analyzed_total`, `sentiment_latency_seconds`, `sentiment_cache_hits/misses`, `sentiment_queue_depth`

**VectorStore Layer** (`src/vectorstore/`):
- `base.py`: Abstract `VectorStore` class, `VectorSearchResult` and `VectorSearchFilter` dataclasses
- `config.py`: `VectorStoreConfig` Pydantic settings for search defaults and authority weights
- `pgvector_store.py`: `PgVectorStore` implementation wrapping DocumentRepository with dynamic SQL filtering
- `manager.py`: `VectorStoreManager` orchestrates embed + store + search + cleanup operations, computes authority scores
- `VectorSearchFilter` supports: platforms, tickers, theme_ids, min_authority_score, exclude_ids, timestamp_after, timestamp_before
- `cleanup_old_documents(days_to_keep)`: Removes documents older than threshold for storage management

**API Layer** (`src/api/`):
- `app.py`: FastAPI application factory with structlog integration
- `auth.py`: X-API-KEY header authentication with dev mode bypass
- `models.py`: Pydantic request/response models (EmbedRequest, EmbedResponse, SearchRequest, SearchResponse, SentimentRequest, SentimentResponse, ThemeItem, ThemeListResponse, ThemeDetailResponse, ThemeDocumentItem, ThemeDocumentsResponse, ThemeSentimentResponse, ThemeMetricsItem, ThemeMetricsResponse, ThemeEventItem, ThemeEventsResponse, RankedThemeItem, RankedThemesResponse)
- `dependencies.py`: Dependency injection for EmbeddingService, SentimentService, Redis, VectorStoreManager, ThemeRepository, DocumentRepository, SentimentAggregator, and ThemeRankingService
- `routes/embed.py`: POST /embed endpoint with auto model selection
- `routes/sentiment.py`: POST /sentiment endpoint with optional entity-level analysis
- `routes/search.py`: POST /search/similar endpoint for semantic search with filters
- `routes/health.py`: GET /health endpoint for service status
- `routes/themes.py`: Theme REST API endpoints:
  - GET /themes — list themes with lifecycle_stage filter, pagination, optional centroid
  - GET /themes/{theme_id} — single theme detail with optional centroid
  - GET /themes/{theme_id}/documents — documents in a theme with platform/authority filters
  - GET /themes/{theme_id}/sentiment — aggregated sentiment with exponential decay weighting
  - GET /themes/{theme_id}/metrics — daily metrics time series with date range
- `routes/themes.py` (ranked endpoint):
  - GET /themes/ranked — ranked themes by actionability with strategy/max_tier/limit params, returns RankedThemesResponse
- `routes/events.py`: Event endpoints for theme-linked event retrieval:
  - GET /themes/{theme_id}/events — events linked via ticker overlap with dedup, event_type/days/limit filters, investment_signal

**Services** (`src/services/`):
- `IngestionService`: Runs adapters concurrently, publishes to queue
- `ProcessingService`: Consumes batches, applies pipeline, stores to DB, queues for embedding and sentiment

### Configuration

Settings in `src/config/settings.py` use Pydantic BaseSettings with env var overrides:
- `DATABASE_URL`, `REDIS_URL` for infrastructure
- `TWITTER_BEARER_TOKEN`, `REDDIT_CLIENT_ID`, etc. for APIs
- `SOTWE_ENABLED` (true), `SOTWE_USERNAMES` (comma-separated, overrides defaults), `SOTWE_RATE_LIMIT` (10) for Sotwe fallback
- `NEWSFILTER_API_KEYS`, `MARKETAUX_API_KEYS`, `FINLIGHT_API_KEYS` for new news sources (comma-separated for key rotation)
- `max_http_retries` (3), `max_backoff_seconds` (60.0) for HTTP retry configuration
- `spam_threshold` (0.7), `duplicate_threshold` (0.85) for processing
- `embedding_model_name` (ProsusAI/finbert), `minilm_model_name` (sentence-transformers/all-MiniLM-L6-v2) for embedding models
- `embedding_batch_size` (32), `embedding_device` (auto), `embedding_cache_enabled` (true), `embedding_cache_ttl_hours` (168)
- `api_host` (0.0.0.0), `api_port` (8000), `api_keys` (comma-separated, empty = dev mode) for embedding API
- `vectorstore_default_limit` (10), `vectorstore_default_threshold` (0.7) for search defaults
- `vectorstore_centroid_limit` (100), `vectorstore_centroid_threshold` (0.5) for theme/cluster queries
- `sentiment_model_name` (ProsusAI/finbert), `sentiment_batch_size` (16), `sentiment_device` (auto) for sentiment analysis
- `sentiment_cache_enabled` (true), `sentiment_cache_ttl_hours` (168) for sentiment caching
- `sentiment_enable_entity_sentiment` (true), `sentiment_entity_context_window` (100) for entity-level analysis
- `sentiment_stream_name` (sentiment_queue), `sentiment_consumer_group` (sentiment_workers) for queue config
- `ner_enabled` (false), `ner_spacy_model` (en_core_web_trf) for NER configuration
- `NER_ENABLE_SEMANTIC_LINKING` (false), `NER_SEMANTIC_SIMILARITY_THRESHOLD` (0.5), `NER_SEMANTIC_BASE_SCORE` (0.6) for embedding-based entity-theme linking
- `keywords_enabled` (false), `keywords_top_n` (10) for keyword extraction configuration
- Keywords settings can be overridden via `KEYWORDS_*` environment variables (e.g., `KEYWORDS_TOP_N=15`, `KEYWORDS_MIN_SCORE=0.01`)
- `events_enabled` (false) for event extraction in preprocessing
- Event extraction settings can be overridden via `EVENTS_*` environment variables (e.g., `EVENTS_MIN_CONFIDENCE=0.6`, `EVENTS_MAX_EVENTS_PER_DOC=50`)
- `clustering_enabled` (false), `clustering_stream_name` (clustering_queue), `clustering_consumer_group` (clustering_workers) for BERTopic clustering
- Clustering settings can be overridden via `CLUSTERING_*` environment variables (e.g., `CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE=20`, `CLUSTERING_UMAP_N_COMPONENTS=15`)
- `volume_metrics_enabled` (false) for volume metrics computation
- Volume metrics settings can be overridden via `VOLUME_*` environment variables (e.g., `VOLUME_DECAY_FACTOR=0.5`, `VOLUME_SURGE_THRESHOLD=4.0`)
- `ranking_enabled` (false) for theme ranking engine actionability scoring
- Ranking settings can be overridden via `RANKING_*` environment variables (e.g., `RANKING_DEFAULT_STRATEGY=position`, `RANKING_TIER_1_PERCENTILE=0.10`)
- `graph_enabled` (false) for causal graph supply chain modeling
- Graph settings can be overridden via `GRAPH_*` environment variables (e.g., `GRAPH_MAX_TRAVERSAL_DEPTH=5`, `GRAPH_DEFAULT_CONFIDENCE=1.0`)

Semiconductor tickers and company mappings are in `src/config/tickers.py`.

Curated Twitter accounts for Sotwe fallback are in `src/config/twitter_accounts.py` (analysts, companies, market accounts).

NER settings can be overridden via `NER_*` environment variables (e.g., `NER_SPACY_MODEL=en_core_web_sm`, `NER_ENABLE_SEMANTIC_LINKING=true`).

### Data Schema

`NormalizedDocument` is the canonical schema throughout the pipeline:
- Identity: `id`, `platform`, `url`
- Content: `content`, `content_type`, `title`
- Author: `author_id`, `author_name`, `author_verified`, `author_followers`
- Quality: `spam_score`, `bot_probability`, `authority_score`, `tickers_mentioned`, `entities_mentioned`, `keywords_extracted`
- Engagement: `likes`, `shares`, `comments`, `views`
- Embedding: `embedding` (FinBERT 768-dim), `embedding_minilm` (MiniLM 384-dim) - generated async by EmbeddingWorker
- Sentiment: `sentiment` (JSONB with label, confidence, scores, entity_sentiments) - generated async by SentimentWorker
- Clustering: `theme_ids` - assigned by clustering service

### Critical Implementation Rules

#### Rate Limiting in Adapters

**Rule**: Rate limiting MUST be applied at the I/O boundary (before HTTP calls), NOT at the data transformation layer (after yields).

```python
# ✅ CORRECT: Rate limit before HTTP request
async def _fetch_raw(self):
    for ticker in tickers:
        await self._rate_limiter.acquire()  # Before HTTP call
        response = await client.get(url, params={"ticker": ticker})
        for article in response.json():
            yield article  # No rate limit here - yields are free

# ❌ WRONG: Rate limit after yield (in base class fetch loop)
async def fetch(self):
    async for raw in self._fetch_raw():
        await self._rate_limiter.acquire()  # Too late! API call already happened
        yield self._transform(raw)
```

**Why it matters**: APIs return batches (NewsAPI: 50 articles, Twitter: 100 tweets). Rate limiting per-item instead of per-request causes N× slowdown where N = batch size. A 50-article response should take 1 rate limit wait, not 50.

**Implementation**: Each adapter's `_fetch_raw()` must call `await self._rate_limiter.acquire()` before every HTTP request. The base class `fetch()` method does NOT rate limit—it only handles transformation and error handling.

### Patterns

- **Adapter Pattern**: New platforms extend `BaseAdapter`, implement `_fetch_raw()` (with rate limiting before HTTP calls) and `_transform()`
- **Rate Limiting at I/O Boundary**: Rate limiters protect external APIs, so they must guard HTTP calls directly—not downstream data processing. See "Critical Implementation Rules" above.
- **Consumer Groups**: Redis Streams enable horizontal scaling of processing workers
- **Singleton Ticker Extractor**: `get_ticker_extractor()` in `base_adapter.py` avoids circular imports
- **Dataclass Config**: `ArticleSourceConfig` in `news_adapter.py` consolidates source-specific transformation logic
- **HTTP Infrastructure Layer**: `HTTPClient` separates retry/backoff concerns from business logic in adapters
- **API Key Rotation**: `APIKeyRotator` enables round-robin rotation of comma-separated keys to distribute rate limits
- **Lazy Model Loading**: `EmbeddingService` defers model loading until first embed call to save memory
- **Model Registry Pattern**: Dict-based model/tokenizer storage keyed by `ModelType` enum for multi-model support
- **Content-Hash Caching**: Embeddings cached by SHA256 hash with model prefix (`emb:finbert:{hash}`, `emb:minilm:{hash}`)
- **Async Embedding Pipeline**: Decoupled from main processing via `embedding_queue` Redis Stream
- **Tiered Model Selection**: Worker selects MiniLM for Twitter + short content (<300 chars), FinBERT otherwise
- **Dependency Injection**: FastAPI `Depends()` for service instantiation and authentication
- **Strategy Pattern**: `VectorStore` ABC enables swapping backends (pgvector → Pinecone) without changing application code
- **Dynamic SQL Builder**: `PgVectorStore.search()` constructs parameterized queries with variable filter combinations
- **Log-Scaled Scoring**: Authority score uses `log(value+1)/log(scale)` to compress power-law distributions (followers, engagement)
- **Orchestration Layer**: `VectorStoreManager` combines EmbeddingService + VectorStore for unified embed+search API
- **Timestamp Range Queries**: Hybrid search combining vector similarity with temporal filtering for time-sensitive news
- **EntityRuler Before NER**: spaCy EntityRuler runs before statistical NER for domain-specific pattern priority
- **Opt-in NER**: NER is disabled by default (`enable_ner=False`) to avoid memory overhead when not needed
- **Semantic Theme Linking**: `link_entities_to_theme_semantic()` uses MiniLM embeddings + cosine similarity for robust conglomerate disambiguation (e.g., "Samsung Electronics" vs "Samsung Galaxy")
- **Async Sentiment Pipeline**: Decoupled from main processing via `sentiment_queue` Redis Stream, follows EmbeddingWorker pattern
- **Entity Context Windows**: Entity-level sentiment extracts text windows around entity mentions for aspect-based classification
- **Cache Isolation**: Document-level and entity-level sentiment use separate cache strategies to prevent cache poisoning
- **RED Metrics Pattern**: Sentiment metrics follow Rate (analyzed_total), Errors (errors_total), Duration (latency_seconds) pattern for observability
- **Multi-Source Fallback**: TwitterAdapter uses API when available, falls back to Sotwe scraping (follows NewsAdapter pattern)
- **Browser Impersonation**: `curl_cffi` with `impersonate="chrome"` bypasses Cloudflare protection for Sotwe
- **NUXT Data Extraction**: Tweet data extracted directly from embedded NUXT JavaScript using regex (no Node.js required)
- **Per-Run Model Creation**: BERTopicService creates a fresh model per `fit()` call (training artifact, not reused), unlike inference services that cache models
- **Deterministic Theme IDs**: SHA256 hash of sorted topic words ensures stable IDs across re-fits with the same topic words
- **Batch Cosine Similarity**: `transform()` uses normalized matrix multiply `embeddings @ centroids.T` for O(n_docs × n_themes) similarity without Python loops
- **Three-Tier Assignment**: Strong/weak/new-candidate routing based on cosine similarity thresholds controls centroid drift and new theme detection
- **EMA Centroid Update**: `centroid = (1 - lr) * centroid + lr * embedding` adapts themes to evolving content with O(1) per-document cost
- **Greedy Pairwise Merge**: `merge_similar_themes()` processes pairs in descending similarity order with `merged_set` to prevent chain merges in a single pass
- **Lightweight TF-IDF Keywords**: `_extract_keywords_tfidf()` uses sklearn CountVectorizer + TfidfTransformer for small candidate clusters, avoiding full BERTopic overhead
- **Mockable Sub-Clustering**: `_create_mini_clusterer()` follows `_create_model()` pattern — deferred HDBSCAN import wrapped in a method for easy test mocking
- **Conditional Fan-Out**: EmbeddingWorker enqueues to clustering_queue only when `clustering_enabled=True`, with soft failure (warning log, no embedding failure) to keep clustering as a non-critical downstream enrichment
- **Separate Centroid Fast-Path**: `ThemeRepository.update_centroid()` is a dedicated method for the hot-path centroid EMA update — fixed SQL with `execute()`, no JSONB, no RETURNING, minimal overhead
- **DB-Layer vs Clustering-Layer Schemas**: `Theme` (persistence, DB columns) vs `ThemeCluster` (in-memory, BERTopic fields) cleanly separates concerns at the domain boundary
- **TTL Cache without External Deps**: `_TTLCache` in ThemeRepository uses a simple dict + `time.monotonic()` expiry instead of adding `cachetools` dependency — minimal code for key→(value, expiry) with lazy eviction
- **Cache-Through Batch Reads**: `get_centroids_batch()` checks cache first, fetches only uncached IDs in a single `ANY($1)` query, populates cache on read, invalidates on write — read-heavy pattern for ClusteringWorker
- **Idempotent Metrics Upsert**: `add_metrics()` uses `ON CONFLICT (theme_id, date) DO UPDATE SET` for safe re-runs of daily batch jobs
- **pgvector for Real-Time Assignment**: ClusteringWorker uses `ThemeRepository.find_similar()` (HNSW index) for O(log n) per-document lookup, not BERTopicService (batch UMAP + HDBSCAN)
- **Redis SET NX Idempotency**: `idempotent:cluster:{doc_id}:{model}` with 7-day TTL prevents reprocessing; key is deleted if embedding missing so re-queue succeeds
- **Atomic Array Merge**: `update_themes()` uses `ARRAY(SELECT DISTINCT unnest(theme_ids || $2))` for safe concurrent theme assignment without read-modify-write races
- **Atomic Counter Increment**: `document_count = document_count + 1` raw SQL avoids stale-read increment races between concurrent workers
- **Projection Query for Batch Efficiency**: `get_with_embeddings_since()` returns 6 fields as dicts instead of full `NormalizedDocument` (24+ fields) to minimize memory for 50k-doc batch clustering
- **Numpy Batch Similarity**: Daily job uses `emb_norm @ centroid_norm.T` for O(n_docs × n_themes) similarity in a single matrix multiply — no Python loops, handles 50k × 500 in ~10ms
- **Dual-Path Clustering**: Real-time `ClusteringWorker` (pgvector HNSW, per-document) vs offline `run_daily_clustering` (numpy batch, per-day) serve different latency/throughput tradeoffs
- **BERTopicService Reuse**: Daily job populates `_themes` dict from DB `Theme` records and sets `_initialized=True` to reuse `merge_similar_themes()` and `check_new_themes()` without reimplementing
- **Phase-Resilient Error Handling**: Each phase (fetch, assign, centroid update, metrics, lifecycle, merge) has independent try/except — failures are logged to `result.errors` without aborting the job
- **Stateless Classifier**: `LifecycleClassifier` has no instance state — takes Theme + metrics, returns stage + confidence. Trivially testable without mocking DB
- **Stateless Volume Service**: `VolumeMetricsService` pure methods accept duck-typed docs (any object with `.timestamp`, `.platform`, `.authority_score`) — no DB mocking needed for unit tests
- **EMA-Based Velocity**: Short EMA minus long EMA on z-scores detects momentum changes faster than SMA, with configurable spans for sensitivity tuning
- **Platform-Weighted Volume**: `DEFAULT_PLATFORM_WEIGHTS` scales document contributions by platform signal quality (Substack 100× vs Twitter 1×), combined with exponential recency decay and authority score
- **Persisted Weighted Volume**: Raw `weighted_volume` stored in `theme_metrics` so z-score history can reference 30-day rolling volumes without expensive doc re-fetches
- **Normalized Trend Analysis**: `_compute_trend()` divides raw least-squares slope by `abs(mean)` so thresholds work across themes with different volume scales
- **Alertable Transition Lookup**: `ALERTABLE_TRANSITIONS` dict maps `(from, to)` tuples to messages — O(1) lookup, easy to extend without modifying class logic
- **Opt-In Centroid Serialization**: Theme API excludes 768-float centroid (~6KB) by default; `?include_centroid=true` opts in. `_theme_to_item()` centralizes Theme→ThemeItem conversion
- **Sentiment Endpoint Chaining**: `/themes/{id}/sentiment` chains ThemeRepository (existence check) → DocumentRepository (lightweight sentiment rows) → SentimentAggregator (sync CPU-only) for clean separation of DB/compute concerns
- **Dynamic SQL for Theme Documents**: `get_documents_by_theme()` uses incremental `param_idx` builder pattern with optional platform/min_authority filters, matching existing repository style
- **SVO Event Extraction**: `PatternExtractor` uses named regex groups (`?P<actor>`, `?P<action>`, `?P<object>`) to extract Subject-Verb-Object triplets from financial text
- **Lazy Pattern Compilation**: `_build_patterns()` compiles regex on first `.patterns` access; subsequent accesses reuse the compiled dict
- **Overlap Deduplication**: `_overlaps()` prevents the same text span from generating duplicate events across different event type patterns
- **Additive Confidence Scoring**: Base 0.7 per regex match, +0.1 for actor/ticker/quantity signals, capped at 1.0
- **Chain-of-Responsibility TimeNormalizer**: Each `_try_*` method returns `None` on non-match, falling through to the next — easy to extend with new time patterns
- **Opt-in Event Extraction**: Events disabled by default (`events_enabled=False`) to avoid overhead when not needed, follows NER/Keywords pattern
- **Query-Time Ticker Linkage**: `get_events_by_tickers()` uses `&&` (array overlap) operator + GIN index for on-the-fly event-theme association — no FK column or schema migration needed
- **Over-Fetch Then Dedup**: Events endpoint fetches `limit * 3` rows, deduplicates in Python, truncates to `limit` — accounts for cross-document event redundancy
- **Composite-Key Dedup**: `(actor, action, object, time_ref)` lowercased composite key for event deduplication; +0.05 confidence per confirming source, capped at 1.0
- **Directional Investment Signal**: `ThemeWithEvents.investment_signal()` compares supply-side vs product-side event counts to derive `supply_increasing | supply_decreasing | product_momentum | product_risk`
- **Recursive CTE Graph Traversal**: `get_downstream()` / `get_upstream()` use `WITH RECURSIVE` with cycle detection via `NOT node_id = ANY(path)` for O(1) roundtrip multi-hop traversal
- **Depth-Clamped Traversal**: `CausalGraph` clamps caller-supplied depth to `config.max_traversal_depth` to prevent runaway recursive queries
- **Idempotent Edge Upsert**: `add_edge()` uses `ON CONFLICT (source, target, relation) DO UPDATE` with `DISTINCT unnest()` for `source_doc_ids` array merge
- **Two-Layer Graph Architecture**: `GraphRepository` (raw SQL, testable with mock DB) vs `CausalGraph` (config defaults, depth clamping, convenience methods) separates persistence from business logic
- **Composite PK for Multi-Relation Edges**: `PRIMARY KEY (source, target, relation)` allows multiple relationship types between the same node pair (e.g., TSMC supplies_to AND competes_with Samsung)
- **Versioned Seed Data**: `SEED_VERSION` constant enables tracking graph data revisions; `seed_graph()` is idempotent via underlying ON CONFLICT upserts
- **Frozen Dataclass Definitions**: Seed uses `@dataclass(frozen=True)` helper types (`_NodeDef`, `_EdgeDef`) to prevent accidental mutation of static domain data
- **Categorized Edge Lists**: Edges split into domain categories (foundry, equipment, memory, EDA, competition, technology, demand) for maintainability and selective testing
- **Bidirectional Competition Edges**: Competition relationships are explicitly bidirectional (A→B and B→A) since `competes_with` is symmetric but the directed graph requires both edges
- **Multiplicative Ranking Score**: `volume ** alpha * compellingness ** beta * lifecycle_multiplier` — zero in any factor collapses the total, preventing low-quality themes from ranking high on volume alone
- **Z-Score Shift by +2**: `max(0, zscore + 2)` before exponentiation prevents complex numbers from negative base with fractional exponent, maps z=-2 to 0, z=0 to 2, z=3 to 5
- **Strategy-Specific Exponents**: Swing (alpha=0.6 volume-biased) vs Position (alpha=0.4 compellingness-biased) — exponents are sublinear so diminishing returns on extreme values
- **Percentile-Based Tiers**: Top 5%/20% of sorted list scales naturally with theme count, unlike absolute thresholds
- **Tier 1 Gate**: Top percentile alone isn't enough — must also pass z-score >= 2.0 OR be accelerating, preventing low-activity themes from reaching Tier 1 by luck of small population
- **Compellingness Fallback**: Default 5.0 when metadata missing, designed for future LLM Compellingness Scorer (Feature 7.1) to fill `theme.metadata["compellingness"]`
- **Route Order for Static Paths**: `/themes/ranked` must be registered before `/themes/{theme_id}` to prevent FastAPI path parameter from capturing "ranked" as a theme_id

### Testing

Test markers in `pyproject.toml`:
- `@pytest.mark.performance`: Performance benchmarks (upsert throughput, search latency)
- `@pytest.mark.integration`: Integration tests requiring running services

```bash
# Run all tests
uv run pytest tests/ -v

# Run only performance tests
uv run pytest tests/ -v -m performance

# Skip integration tests
uv run pytest tests/ -v -m "not integration"

# Manual NER verification
uv run python -c "
from src.ner import NERService
svc = NERService()
entities = svc.extract_sync('Nvidia announced HBM3E support for H200 GPUs')
for e in entities:
    print(f'{e.type}: {e.text} -> {e.normalized}')
"

# Semantic theme linking (requires embedding models)
uv run python -c "
import asyncio
from src.embedding.service import EmbeddingService
from src.ner import NERService

async def main():
    emb_svc = EmbeddingService()
    ner_svc = NERService(embedding_service=emb_svc)
    entities = ner_svc.extract_sync('Nvidia announced HBM3E for H200')
    scores = await ner_svc.link_entities_to_theme_semantic(
        entities, ['AI accelerator', 'deep learning']
    )
    for name, score in scores.items():
        print(f'{name}: {score:.2f}')

asyncio.run(main())
"

# Sotwe fallback testing
uv run pytest tests/test_ingestion/test_sotwe.py -v -m "not integration"  # Unit tests
uv run pytest tests/test_ingestion/test_sotwe.py -v -m integration        # Integration (requires Node.js)

# Sentiment testing
uv run pytest tests/test_sentiment/ -v     # Run all sentiment tests

# Manual sentiment verification
uv run python -c "
import asyncio
from src.sentiment.service import SentimentService

async def main():
    svc = SentimentService()
    result = await svc.analyze('NVIDIA stock surged 10% on strong AI demand')
    print(f'Label: {result[\"label\"]} ({result[\"confidence\"]:.2f})')
    print(f'Scores: {result[\"scores\"]}')
    await svc.close()

asyncio.run(main())
"

# Keywords testing
uv run pytest tests/test_keywords/ -v     # Run all keyword tests

# Manual keywords verification
uv run python -c "
from src.keywords import KeywordsService
svc = KeywordsService()
keywords = svc.extract_sync('Nvidia announced new GPU architecture with HBM3E memory')
for kw in keywords:
    print(f'{kw.rank}. {kw.text} (score: {kw.score:.3f})')
"

# Event extraction testing
uv run pytest tests/test_event_extraction/ -v              # Run all event extraction tests
uv run pytest tests/test_event_extraction/test_patterns.py -v              # Run pattern extractor tests
uv run pytest tests/test_event_extraction/test_normalizer.py -v            # Run time normalizer tests
uv run pytest tests/test_event_extraction/test_schemas.py -v               # Run schema tests
uv run pytest tests/test_event_extraction/test_preprocessor_integration.py -v  # Run integration tests
uv run pytest tests/test_event_extraction/test_theme_integration.py -v         # Run event-theme integration tests

# Clustering testing
uv run pytest tests/test_clustering/ -v   # Run all clustering tests (schema + service + config + worker + daily job)
uv run pytest tests/test_clustering/test_service.py -v -k "Transform"  # Run only transform tests
uv run pytest tests/test_clustering/test_service.py -v -k "Merge or CheckNew"  # Run merge + new theme tests
uv run pytest tests/test_clustering/test_worker.py -v   # Run clustering worker tests
uv run pytest tests/test_clustering/test_daily_job.py -v  # Run daily batch clustering tests
uv run pytest tests/test_clustering/test_bertopic_service.py -v -m integration   # Integration tests (real BERTopic)
uv run pytest tests/test_clustering/test_bertopic_service.py -v -m performance   # Performance benchmarks (slow)

# Themes testing
uv run pytest tests/test_themes/ -v              # Run all theme tests (schema + repository + lifecycle + search + metrics)
uv run pytest tests/test_themes/test_repository.py -v -k "Update"  # Run only update tests
uv run pytest tests/test_themes/test_repository.py -v -k "FindSimilar or GetCentroidsBatch"  # Run vector search tests
uv run pytest tests/test_themes/test_repository.py -v -k "Metrics"  # Run metrics time-series tests
uv run pytest tests/test_themes/test_lifecycle.py -v              # Run lifecycle classifier tests
uv run pytest tests/test_themes/test_lifecycle.py -v -k "Classify"  # Run only classification tests
uv run pytest tests/test_themes/test_lifecycle.py -v -k "Transition"  # Run transition detection tests
uv run pytest tests/test_themes/test_metrics.py -v               # Run volume metrics tests
uv run pytest tests/test_themes/test_metrics.py -v -k "WeightedVolume"   # Run weighted volume tests
uv run pytest tests/test_themes/test_metrics.py -v -k "Zscore"          # Run z-score tests
uv run pytest tests/test_themes/test_metrics.py -v -k "Velocity or Acceleration"  # Run velocity/acceleration tests
uv run pytest tests/test_themes/test_metrics.py -v -k "Anomaly"         # Run anomaly detection tests
uv run pytest tests/test_themes/test_metrics.py -v -k "ComputeForTheme" # Run orchestrator tests
uv run pytest tests/test_themes/test_ranking.py -v              # Run ranking engine tests
uv run pytest tests/test_themes/test_ranking.py -v -k "ComputeScore"   # Run scoring formula tests
uv run pytest tests/test_themes/test_ranking.py -v -k "RankThemes"     # Run sorting/filtering tests
uv run pytest tests/test_themes/test_ranking.py -v -k "AssignTiers"    # Run tier assignment tests
uv run pytest tests/test_themes/test_ranking.py -v -k "GetActionable"  # Run orchestrator tests

# Graph testing
uv run pytest tests/test_graph/ -v                         # Run all graph tests
uv run pytest tests/test_graph/test_schemas.py -v          # Run schema validation tests
uv run pytest tests/test_graph/test_storage.py -v          # Run repository CRUD + traversal tests
uv run pytest tests/test_graph/test_causal_graph.py -v     # Run high-level service tests
uv run pytest tests/test_graph/test_storage.py -v -k "Downstream or Upstream"  # Run traversal tests
uv run pytest tests/test_graph/test_storage.py -v -k "FindPath"               # Run path finding tests
uv run pytest tests/test_graph/test_storage.py -v -k "Subgraph"               # Run subgraph extraction tests
uv run pytest tests/test_graph/test_seed_data.py -v                            # Run seed data tests
uv run pytest tests/test_graph/test_seed_data.py -v -k "Integrity"            # Run data integrity tests
uv run pytest tests/test_graph/test_seed_data.py -v -k "Coverage"             # Run domain coverage tests
uv run pytest tests/test_graph/test_seed_data.py -v -k "SeedGraphFunction"    # Run seed function tests

# CLI testing
uv run pytest tests/test_cli/ -v                         # Run all CLI tests
uv run pytest tests/test_cli/test_cluster.py -v          # Run cluster command tests
uv run pytest tests/test_cli/test_cluster.py -v -k "Fit"       # Run fit subcommand tests
uv run pytest tests/test_cli/test_cluster.py -v -k "Backfill"  # Run backfill tests
uv run pytest tests/test_cli/test_cluster.py -v -k "Merge"     # Run merge tests
uv run pytest tests/test_cli/test_graph.py -v              # Run graph CLI command tests

# API testing
uv run pytest tests/test_api/ -v                        # Run all API tests
uv run pytest tests/test_api/test_themes.py -v           # Run theme endpoint tests
uv run pytest tests/test_api/test_themes.py -v -k "ListThemes"    # Run list endpoint tests
uv run pytest tests/test_api/test_themes.py -v -k "GetTheme and not Documents"  # Run detail endpoint tests
uv run pytest tests/test_api/test_themes.py -v -k "Documents"     # Run documents endpoint tests
uv run pytest tests/test_api/test_themes.py -v -k "Sentiment"     # Run sentiment endpoint tests
uv run pytest tests/test_api/test_themes.py -v -k "Metrics"       # Run metrics endpoint tests
uv run pytest tests/test_api/test_themes.py -v -k "Ranked"        # Run ranked endpoint tests
uv run pytest tests/test_api/test_events.py -v                    # Run event endpoint tests
uv run pytest tests/test_api/test_events.py -v -k "deduplication"        # Run dedup tests
uv run pytest tests/test_api/test_events.py -v -k "investment_signal"    # Run signal tests
```
