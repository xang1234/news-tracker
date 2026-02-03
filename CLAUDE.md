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
uv run news-tracker serve              # Start embedding API server
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
                                    ▼ publish document_id
                            embedding_queue (Redis Stream)
                                    │
                                    ▼ consume
                            EmbeddingWorker
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            FinBERT (768-dim)               MiniLM (384-dim)
            Long/financial text             Short social posts
                    │                               │
                    ▼                               ▼
            embedding column                embedding_minilm column
                            PostgreSQL

                            ┌─────────────┐
                            │ FastAPI     │
                            │ /embed      │◄── External clients
                            │ /health     │
                            └─────────────┘
```

### Key Components

**Ingestion Layer** (`src/ingestion/`):
- `BaseAdapter`: Abstract base with rate limiting, error handling. Subclasses implement `_fetch_raw()` and `_transform()`.
- Platform adapters: `TwitterAdapter`, `RedditAdapter`, `SubstackAdapter`, `NewsAdapter`, `MockAdapter`
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

**Storage Layer** (`src/storage/`):
- `Database`: asyncpg connection pool with transaction context managers
- `DocumentRepository`: CRUD operations, batch upserts, full-text search, similarity search

**Embedding Layer** (`src/embedding/`):
- `EmbeddingConfig`: Pydantic settings for model, batching, caching, queue configuration
- `EmbeddingService`: Multi-model embedding (FinBERT 768-dim, MiniLM 384-dim) with lazy loading, chunking, caching
- `EmbeddingQueue`: Redis Streams wrapper for async embedding job processing
- `EmbeddingWorker`: Consumes queue, selects model by platform/length, generates embeddings, updates DB
- `ModelType`: Enum for type-safe model selection (FINBERT, MINILM)

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
- `models.py`: Pydantic request/response models (EmbedRequest, EmbedResponse, SearchRequest, SearchResponse)
- `dependencies.py`: Dependency injection for EmbeddingService, Redis, and VectorStoreManager
- `routes/embed.py`: POST /embed endpoint with auto model selection
- `routes/search.py`: POST /search/similar endpoint for semantic search with filters
- `routes/health.py`: GET /health endpoint for service status

**Services** (`src/services/`):
- `IngestionService`: Runs adapters concurrently, publishes to queue
- `ProcessingService`: Consumes batches, applies pipeline, stores to DB

### Configuration

Settings in `src/config/settings.py` use Pydantic BaseSettings with env var overrides:
- `DATABASE_URL`, `REDIS_URL` for infrastructure
- `TWITTER_BEARER_TOKEN`, `REDDIT_CLIENT_ID`, etc. for APIs
- `NEWSFILTER_API_KEYS`, `MARKETAUX_API_KEYS`, `FINLIGHT_API_KEYS` for new news sources (comma-separated for key rotation)
- `max_http_retries` (3), `max_backoff_seconds` (60.0) for HTTP retry configuration
- `spam_threshold` (0.7), `duplicate_threshold` (0.85) for processing
- `embedding_model_name` (ProsusAI/finbert), `minilm_model_name` (sentence-transformers/all-MiniLM-L6-v2) for embedding models
- `embedding_batch_size` (32), `embedding_device` (auto), `embedding_cache_enabled` (true), `embedding_cache_ttl_hours` (168)
- `api_host` (0.0.0.0), `api_port` (8000), `api_keys` (comma-separated, empty = dev mode) for embedding API
- `vectorstore_default_limit` (10), `vectorstore_default_threshold` (0.7) for search defaults
- `vectorstore_centroid_limit` (100), `vectorstore_centroid_threshold` (0.5) for theme/cluster queries
- `ner_enabled` (false), `ner_spacy_model` (en_core_web_trf) for NER configuration
- `NER_ENABLE_SEMANTIC_LINKING` (false), `NER_SEMANTIC_SIMILARITY_THRESHOLD` (0.5), `NER_SEMANTIC_BASE_SCORE` (0.6) for embedding-based entity-theme linking

Semiconductor tickers and company mappings are in `src/config/tickers.py`.

NER settings can be overridden via `NER_*` environment variables (e.g., `NER_SPACY_MODEL=en_core_web_sm`, `NER_ENABLE_SEMANTIC_LINKING=true`).

### Data Schema

`NormalizedDocument` is the canonical schema throughout the pipeline:
- Identity: `id`, `platform`, `url`
- Content: `content`, `content_type`, `title`
- Author: `author_id`, `author_name`, `author_verified`, `author_followers`
- Quality: `spam_score`, `bot_probability`, `authority_score`, `tickers_mentioned`, `entities_mentioned`
- Engagement: `likes`, `shares`, `comments`, `views`
- Embedding: `embedding` (FinBERT 768-dim), `embedding_minilm` (MiniLM 384-dim) - generated async by EmbeddingWorker
- Clustering: `theme_ids` - assigned by clustering service

### Patterns

- **Adapter Pattern**: New platforms extend `BaseAdapter`, get rate limiting and preprocessing free
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
```
