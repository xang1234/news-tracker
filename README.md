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
    "min_authority_score": 0.5
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
| `threshold` | float | Minimum similarity score (default: 0.7) |
| `limit` | int | Maximum results (default: 10, max: 100) |
