# News Tracker

Multi-platform financial data ingestion framework for tracking semiconductor and tech news.

## Architecture

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
                    └───────────────┬───────────────┘
                                    ▼
                            PostgreSQL + pgvector
                            (HNSW indexes)
                                    │
                            ┌───────┴───────┐
                            ▼               ▼
                    ┌─────────────┐  ┌─────────────┐
                    │ FastAPI     │  │ CLI         │
                    │ /embed      │  │ vector-     │
                    │ /search     │  │ search      │
                    │ /health     │  └─────────────┘
                    └─────────────┘
```

## Data Sources

| Platform | API/Method | Rate Limit | Content Type |
|----------|------------|------------|--------------|
| **Twitter** | Twitter API v2 (Bearer Token) | 30 req/min | Posts with cashtags, financial influencers |
| **Reddit** | Reddit OAuth API | 60 req/min | Posts from r/wallstreetbets, r/stocks, r/semiconductors, etc. |
| **Substack** | RSS feeds (public) | 10 req/min | Newsletter articles from SemiAnalysis, Stratechery, Asianometry |
| **News APIs** | Finnhub, NewsAPI, Alpha Vantage, Newsfilter, Marketaux, Finlight | 60 req/min | Financial news with multi-source fallback |

## Ingestion Methods

### Twitter
- Queries posts containing tracked semiconductor tickers ($NVDA, $AMD, $INTC, etc.)
- Filters by verified accounts and engagement thresholds
- Extracts cashtags and maps company mentions to tickers

### Reddit
- Monitors financial subreddits: wallstreetbets, stocks, investing, semiconductors, AMD_Stock, nvidia, intel
- Fetches hot posts per subreddit
- Extracts tickers from natural language (Reddit doesn't use cashtags)

### Substack
- Polls RSS feeds from curated publications:
  - SemiAnalysis (semiconductor deep dives)
  - Stratechery (tech business analysis)
  - Asianometry (tech and economics)
  - Doomberg (commodities and energy)
- Parses HTML content, extracts clean text

### News APIs
- **Finnhub** (primary): Company-specific financial news by ticker
- **NewsAPI** (fallback): Broader keyword-based news search
- **Alpha Vantage** (tertiary): News sentiment with ticker relevance scores
- **Newsfilter.io**: Real-time SEC filings and financial news (POST, Bearer token auth)
- **Marketaux**: Global financial news with entity recognition (GET, query param auth)
- **Finlight.me**: AI-curated financial news with sentiment analysis (POST, X-API-KEY auth)
- Deduplicates across sources, applies source authority weighting (WSJ/Bloomberg/Reuters ranked higher)
- New sources support multiple API keys with round-robin rotation and exponential backoff on rate limits

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Start infrastructure
docker compose up -d

# Initialize database
uv run news-tracker init-db

# Run with mock data (for testing)
uv run news-tracker worker --mock

# Run with real APIs (requires API keys in .env)
uv run news-tracker worker
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required for real data
TWITTER_BEARER_TOKEN=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
FINNHUB_API_KEY=...

# Optional news sources (single key)
NEWSAPI_API_KEY=...
ALPHA_VANTAGE_API_KEY=...

# Optional news sources (comma-separated for key rotation)
NEWSFILTER_API_KEYS=key1,key2,key3
MARKETAUX_API_KEYS=...
FINLIGHT_API_KEYS=...

# HTTP retry configuration
MAX_HTTP_RETRIES=3
MAX_BACKOFF_SECONDS=60.0

# Embedding service
EMBEDDING_MODEL_NAME=ProsusAI/finbert
EMBEDDING_MINILM_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_USE_FP16=true
EMBEDDING_DEVICE=auto
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_TTL_HOURS=168

# Embedding API
API_HOST=0.0.0.0
API_PORT=8000
API_KEYS=key1,key2  # Comma-separated valid keys (empty = dev mode, no auth)
```

## Embedding Service

The embedding service generates vector representations using a tiered model approach:

| Model | Dimensions | Use Case | Selection Criteria |
|-------|------------|----------|-------------------|
| **FinBERT** | 768 | Financial domain, longer text | Default for most content |
| **MiniLM** | 384 | General purpose, fast | Twitter posts < 300 chars |

### Features

- **Tiered model selection** - Automatically selects model based on platform and content length
- **Lazy model loading** - Each model loads on first use, saving memory
- **Automatic device detection** - Uses CUDA > MPS > CPU
- **Long document handling** - Chunks documents >512 tokens with overlapping windows and mean pooling
- **Redis caching** - Content-hash based caching with model prefix avoids recomputation
- **Async processing** - Decoupled from main pipeline via Redis Streams
- **Similarity search** - HNSW indexes for efficient cosine similarity queries

### Database Migration

For existing installations, run migrations to set up embedding columns:

```bash
# FinBERT 768-dim column
psql $DATABASE_URL -f migrations/001_embedding_vector_768.sql

# MiniLM 384-dim column
psql $DATABASE_URL -f migrations/002_add_minilm_embedding.sql
```

### Similarity Search

Query similar documents using pgvector:

```sql
-- FinBERT similarity search (768-dim)
SELECT id, platform, title,
       1 - (embedding <=> $1) AS similarity
FROM documents
WHERE embedding IS NOT NULL
  AND 1 - (embedding <=> $1) >= 0.7
ORDER BY embedding <=> $1
LIMIT 10;

-- MiniLM similarity search (384-dim)
SELECT id, platform, title,
       1 - (embedding_minilm <=> $1) AS similarity
FROM documents
WHERE embedding_minilm IS NOT NULL
  AND 1 - (embedding_minilm <=> $1) >= 0.7
ORDER BY embedding_minilm <=> $1
LIMIT 10;
```

## Embedding API

A FastAPI server provides HTTP access to the embedding service for external clients.

### Start the API Server

```bash
uv run news-tracker serve --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/embed` | Generate embeddings for batch of texts (1-100) |
| GET | `/health` | Service health check with model and cache status |

### Example Usage

```bash
# Generate embeddings (auto model selection)
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{"texts": ["NVIDIA reports record Q4 earnings"], "model": "auto"}'

# Force specific model
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{"texts": ["Short tweet"], "model": "minilm"}'

# Health check (no auth required)
curl http://localhost:8000/health
```

### Response Format

```json
{
  "embeddings": [[0.123, -0.456, ...]],
  "model_used": "finbert",
  "dimensions": 768,
  "latency_ms": 45.23
}
```

## Vector Search

Semantic search over documents using the VectorStore abstraction layer built on pgvector.

### Authority Score

Documents are scored for authority (0.0-1.0) based on:

| Component | Max Score | Calculation |
|-----------|-----------|-------------|
| Verified Author | +0.2 | Boolean bonus |
| Follower Count | +0.3 | Log-scaled, caps at 1M followers |
| Engagement | +0.3 | Log-scaled (likes + 2×shares + comments) |
| Inverse Spam | +0.2 | 0.2 × (1 - spam_score) |

### CLI Usage

```bash
# Basic semantic search
uv run news-tracker vector-search "NVIDIA AI demand" --limit 5

# With filters
uv run news-tracker vector-search "semiconductor supply chain" \
  --platform twitter \
  --platform reddit \
  --ticker NVDA \
  --ticker AMD \
  --min-authority 0.6

# Cleanup old documents (storage management)
uv run news-tracker cleanup --days 90              # Delete docs older than 90 days
uv run news-tracker cleanup --days 30 --dry-run   # Preview without deleting
```

### API Endpoint

```bash
# Semantic search with filters
curl -X POST http://localhost:8001/search/similar \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{
    "query": "NVIDIA datacenter revenue growth",
    "limit": 10,
    "threshold": 0.7,
    "platforms": ["twitter", "news"],
    "tickers": ["NVDA"],
    "min_authority_score": 0.5,
    "timestamp_after": "2026-01-01T00:00:00Z",
    "timestamp_before": "2026-02-01T00:00:00Z"
  }'
```

### Search Response

```json
{
  "results": [
    {
      "document_id": "twitter_123456",
      "score": 0.92,
      "platform": "twitter",
      "title": null,
      "content_preview": "NVIDIA reports record Q4...",
      "author_name": "analyst_pro",
      "author_verified": true,
      "tickers": ["NVDA"],
      "authority_score": 0.78,
      "timestamp": "2026-02-03T10:30:00Z"
    }
  ],
  "total": 1,
  "latency_ms": 45.2
}
```

### Filter Options

| Filter | Type | Description |
|--------|------|-------------|
| `platforms` | string[] | Filter by platform (twitter, reddit, news, substack) |
| `tickers` | string[] | Filter by mentioned ticker symbols |
| `theme_ids` | string[] | Filter by theme cluster IDs |
| `min_authority_score` | float | Minimum authority score (0.0-1.0) |
| `timestamp_after` | datetime | Filter to documents created after this time (ISO 8601) |
| `timestamp_before` | datetime | Filter to documents created before this time (ISO 8601) |
| `threshold` | float | Minimum similarity score (default: 0.7) |
| `limit` | int | Maximum results (default: 10, max: 100) |

## Named Entity Recognition (NER)

Optional NER extraction identifies financial entities beyond simple ticker symbols.

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| **TICKER** | Stock ticker symbols | $NVDA, $AMD, $INTC |
| **COMPANY** | Company names (normalized to standard form) | Nvidia → NVIDIA, Taiwan Semiconductor → TSM |
| **PRODUCT** | Hardware products | H100, A100, Snapdragon, GeForce RTX |
| **TECHNOLOGY** | Technical terms | HBM3E, CoWoS, 3nm, EUV lithography |
| **METRIC** | Financial metrics | $5.6 billion, 10% YoY, 51% margin |

### Enable NER

NER is disabled by default to save memory (~500MB for transformer model). Enable via environment variable:

```bash
# In .env
NER_ENABLED=true
NER_SPACY_MODEL=en_core_web_trf  # High accuracy (default)
# NER_SPACY_MODEL=en_core_web_sm  # Faster, lower memory

# Download the spaCy model
python -m spacy download en_core_web_trf
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NER_ENABLED` | false | Enable NER extraction in preprocessing |
| `NER_SPACY_MODEL` | en_core_web_trf | spaCy model (trf=transformer, sm=small) |
| `NER_FUZZY_THRESHOLD` | 85 | Fuzzy matching threshold (0-100) |
| `NER_ENABLE_COREFERENCE` | true | Resolve "the company" → entity |
| `NER_CONFIDENCE_THRESHOLD` | 0.5 | Minimum confidence for entities |
| `NER_ENABLE_SEMANTIC_LINKING` | false | Enable embedding-based theme linking |
| `NER_SEMANTIC_SIMILARITY_THRESHOLD` | 0.5 | Minimum cosine similarity for theme match |
| `NER_SEMANTIC_BASE_SCORE` | 0.6 | Base weight for semantic similarity score |

### Example Output

```python
from src.ner import NERService

service = NERService()
entities = service.extract_sync("Nvidia announced HBM3E support for H200 GPUs")

for e in entities:
    print(f"{e.type}: {e.text} → {e.normalized}")

# Output:
# COMPANY: Nvidia → NVIDIA
# TECHNOLOGY: HBM3E → HBM3E
# PRODUCT: H200 → H200
```

### Extracted Entity Schema

Entities are stored in the `entities_mentioned` JSONB column:

```json
{
  "text": "Nvidia",
  "type": "COMPANY",
  "normalized": "NVIDIA",
  "start": 0,
  "end": 6,
  "confidence": 0.95,
  "metadata": {"ticker": "NVDA"}
}
```

### Semantic Theme Linking

For robust entity disambiguation (e.g., distinguishing "Samsung Electronics" from "Samsung Galaxy"), use embedding-based semantic similarity:

```python
from src.embedding.service import EmbeddingService
from src.ner import NERService

# Initialize with embedding service for semantic linking
embedding_svc = EmbeddingService()
ner_svc = NERService(embedding_service=embedding_svc)

# Extract entities
entities = ner_svc.extract_sync("Nvidia announced HBM3E support for H200")

# Calculate semantic relevance to theme keywords
scores = await ner_svc.link_entities_to_theme_semantic(
    entities,
    ["AI accelerator", "deep learning", "graphics processing"]
)
# {'NVIDIA': 0.82, 'HBM3E': 0.71, 'H200': 0.65}
```

The scoring formula combines semantic similarity with domain-specific bonuses:
- **Semantic similarity**: Cosine similarity between entity and theme embeddings (weighted by `semantic_base_score`)
- **Semiconductor ticker bonus**: +0.2 for entities with known semiconductor ticker symbols
- **Tech entity bonus**: +0.1 for TECHNOLOGY and PRODUCT entity types

This is useful for:
- **Conglomerate disambiguation**: "Samsung" in a semiconductor context vs. consumer electronics
- **Theme clustering**: Grouping entities by topic relevance
- **Search ranking**: Prioritizing entities that match user intent

## Storage Management

The cleanup command removes old documents to prevent unbounded database growth:

```bash
# Delete documents older than 90 days (default)
uv run news-tracker cleanup

# Custom retention period
uv run news-tracker cleanup --days 30

# Preview deletion count without actually deleting
uv run news-tracker cleanup --days 30 --dry-run
```

Programmatic cleanup is also available via `VectorStoreManager.cleanup_old_documents(days_to_keep=90)`.
