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

**Storage Layer** (`src/storage/`):
- `Database`: asyncpg connection pool with transaction context managers
- `DocumentRepository`: CRUD operations, batch upserts, full-text search, similarity search, `update_themes()` array merge

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
- `models.py`: Pydantic request/response models (EmbedRequest, EmbedResponse, SearchRequest, SearchResponse, SentimentRequest, SentimentResponse)
- `dependencies.py`: Dependency injection for EmbeddingService, SentimentService, Redis, and VectorStoreManager
- `routes/embed.py`: POST /embed endpoint with auto model selection
- `routes/sentiment.py`: POST /sentiment endpoint with optional entity-level analysis
- `routes/search.py`: POST /search/similar endpoint for semantic search with filters
- `routes/health.py`: GET /health endpoint for service status

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
- `clustering_enabled` (false), `clustering_stream_name` (clustering_queue), `clustering_consumer_group` (clustering_workers) for BERTopic clustering
- Clustering settings can be overridden via `CLUSTERING_*` environment variables (e.g., `CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE=20`, `CLUSTERING_UMAP_N_COMPONENTS=15`)

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

# Clustering testing
uv run pytest tests/test_clustering/ -v   # Run all clustering tests (schema + service + config + worker)
uv run pytest tests/test_clustering/test_service.py -v -k "Transform"  # Run only transform tests
uv run pytest tests/test_clustering/test_service.py -v -k "Merge or CheckNew"  # Run merge + new theme tests
uv run pytest tests/test_clustering/test_worker.py -v   # Run clustering worker tests

# Themes testing
uv run pytest tests/test_themes/ -v              # Run all theme tests (schema + repository + search + metrics)
uv run pytest tests/test_themes/test_repository.py -v -k "Update"  # Run only update tests
uv run pytest tests/test_themes/test_repository.py -v -k "FindSimilar or GetCentroidsBatch"  # Run vector search tests
uv run pytest tests/test_themes/test_repository.py -v -k "Metrics"  # Run metrics time-series tests
```
