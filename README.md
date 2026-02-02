# News Tracker

Multi-platform financial data ingestion framework for tracking semiconductor and tech news.

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
```
