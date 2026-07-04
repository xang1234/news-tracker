# Design: Broaden ingestion with an RSS/Atom Feed Adapter

- **Date:** 2026-05-21
- **Status:** Approved (design); pending implementation plan
- **Author:** David Ten (with Claude Code)

## 1. Context & goal

The `news-tracker` pipeline should ingest the same breadth of news/data shown on
[tessara.tech/signals](https://tessara.tech/signals) — a scored AI/semiconductor
intelligence feed. Inspection of the Tessara Signals page showed it draws on
sources the repo does **not** currently cover: semiconductor trade press
(EE Times, JW Insights), general tech press (The Verge), and first-party company
announcements.

The repo today ingests via four adapters — `NewsAdapter` (6 financial news
APIs: Finnhub, NewsAPI, Alpha Vantage, Newsfilter, Marketaux, Finlight),
`TwitterAdapter`, `RedditAdapter`, `SubstackAdapter`. None consume **RSS/Atom
feeds**, which is the publishing format for trade press, tech blogs, and company
newsrooms.

**Goal:** add one generic RSS/Atom feed adapter that broadens source coverage to
semiconductor trade press, company/IR newsrooms, and general tech press, feeding
documents into the existing pipeline with no downstream changes.

## 2. Scope

### In scope
- A single generic `FeedAdapter` that consumes a curated list of RSS/Atom feeds.
- Three source categories: semiconductor trade press, company newsrooms / IR,
  general tech press.
- Hybrid content depth: use the feed-supplied body when available; fetch and
  extract the article page for feeds flagged as truncated (`full_text=True`).
- A new `Platform.RSS` enum value, settings, `IngestionService` wiring, frontend
  platform label, and tests.

### Out of scope (explicitly)
- arXiv / research preprints (not selected by the user).
- Signal scoring (1–10 badges), catalyst tags, Rumor/Deep-Dive sub-types,
  Priority/Watch/Feed tiers — the Tessara *presentation* layer.
- The Memory / Compute "desks" (regime scores, constraint boards, sector
  baskets) and earnings-transcript ingestion.
- DB-backed runtime feed management via `SourcesService` — see Phase 2.

## 3. Approach decision

| # | Approach | Verdict |
|---|----------|---------|
| 1 | Generic `FeedAdapter` + config-module feed list | **Chosen** |
| 2 | Generic `FeedAdapter` + DB-backed feeds via `SourcesService` | Deferred (Phase 2) |
| 3 | One adapter class per publication | Rejected — duplication |

**Chosen: #1.** It mirrors two existing patterns: `NewsAdapter`'s "one adapter,
many sources" dispatch, and the static config modules `twitter_accounts.py` /
`tickers.py`. It ships fast and is fully version-controlled. Approach #2 (manage
feeds in the `sources` table, exposed through the `/sources` API) is a clean
follow-up if runtime feed management is wanted; it needs a migration for
per-feed metadata (`full_text`, `category`) and `/sources` UI work.

**Key property — idempotency for free:** the deterministic doc ID
`{platform}_{stable_hash(url)}` means re-listing the same article across polls
produces the same ID, so the downstream `Deduplicator` + DB upsert collapse
repeats. The adapter therefore needs only *in-cycle* dedup and no persistent
state or DB access.

## 4. Components

### 4.1 `src/config/feeds.py` (new)

A static config module mirroring `twitter_accounts.py`.

```python
@dataclass(frozen=True)
class Feed:
    slug: str          # stable id  -> "eetimes"
    name: str          # display    -> "EE Times"
    url: str           # feed URL
    category: str      # "trade_press" | "company_ir" | "company_tech" | "tech_press"
    full_text: bool = False   # True -> fetch + extract the article page
    enabled: bool = True
```

Plus `FEEDS: list[Feed]` seeded with a focused semiconductor catalog. Current
starter set:

- **trade_press:** Semiconductor Engineering, EE Times, Semiconductor Digest,
  Semiconductor Today, SemiWiki
- **company_ir:** NVIDIA, AMD, Intel, Micron, KLA, Samsung press/newsrooms
- **company_tech:** NVIDIA Technical Blog and Samsung semiconductor/newsroom
  feeds

Per-feed `full_text` is set based on whether the feed ships full bodies; feeds
that ship truncated `<description>` excerpts get `full_text=True`.

### 4.2 `src/ingestion/feed_adapter.py` (new)

`FeedAdapter(BaseAdapter)`:

- `__init__(self, feeds: list[Feed] | None = None, rate_limit: int = ...)`:
  `feeds` defaults to the module-level `FEEDS` from `src/config/feeds.py`;
  accepting it as a parameter lets tests inject fixture feeds without patching
  the config module.
- `platform` property → `Platform.RSS`.
- `_fetch_raw()` async generator:
  1. For each enabled `Feed` (filtered to `feed.enabled`):
     - `await self._rate_limiter.acquire()` (before the HTTP call — I/O boundary).
     - Conditional GET of the feed URL, sending cached `ETag` / `Last-Modified`
       (held in an in-memory dict keyed by feed slug, process lifetime). A `304`
       response skips the feed entirely.
     - `feedparser.parse()` the response body.
     - For each entry: skip if older than `rss_recency_days`, if already yielded
       this process (per-feed seen-GUID set, in-memory), or if the per-feed item
       cap `rss_max_items_per_feed` is reached.
     - Resolve content: if `feed.full_text` **and** `rss_full_text_enabled` →
       `await self._rate_limiter.acquire()` → GET the article URL →
       `trafilatura.extract()`; on failure fall back to the feed body. Else use
       `content:encoded` / `description` from the entry.
     - Yield a raw dict carrying the feedparser entry, the resolving `Feed`, and
       the resolved content string.
  2. In-cycle URL dedup via a `_seen_urls` set (cleared at the start of each
     `_fetch_raw()` call), consistent with `NewsAdapter`.
  3. Per-feed `try/except` and per-article `try/except` so one bad feed or one
     unreachable article page never aborts the run.
- `_transform(raw)` → `NormalizedDocument | None`:
  - `id = f"rss_{stable_hash(url)}"`
  - `platform = Platform.RSS`, `content_type = "article"`
  - `url`, `title` from the entry; `timestamp` parsed from entry
    `published`/`updated` (fallback `fetched_at` if absent)
  - `author_id = feed.slug`, `author_name = feed.name` (RSS rarely carries a
    per-article author identity; the publication is the author)
  - `content` = resolved content string (cleaned via `clean_text()`)
  - `tickers_mentioned` via `extract_tickers()` on title + content
  - `raw_data` = original entry dict
  - Returns `None` (never raises) on missing URL / empty content / unparseable
    timestamp — consistent with the `BaseAdapter` contract.
- `health_check()`: GET one feed URL, return whether it parses.

### 4.3 Schema — `src/ingestion/schemas.py`

Add `RSS = "rss"` to the `Platform` enum. **No DB migration needed** — the
`documents.platform` column is `TEXT` (per `migrations/001_initial_schema.sql`).

### 4.4 Configuration — `src/config/settings.py`

New top-level `Settings` fields, consistent with the existing
`news_rate_limit` / `twitter_rate_limit` style (env prefix `RSS_`):

| Field | Default | Purpose |
|-------|---------|---------|
| `rss_enabled` | `True` | Master enable for the feed adapter |
| `rss_rate_limit` | `20` | Requests per minute (feed + article fetches) |
| `rss_max_items_per_feed` | `50` | Cap on items processed per feed per cycle |
| `rss_recency_days` | `7` | Ignore entries older than this |
| `rss_fetch_timeout` | `15.0` | Per-request timeout (seconds) |
| `rss_full_text_enabled` | `True` | Global kill-switch for the article-page fetch path |

### 4.5 Wiring — `src/services/ingestion_service.py`

In `_create_adapters()`, register `FeedAdapter` keyed by `Platform.RSS` when
`settings.rss_enabled` is true and the enabled-feed list is non-empty:

```python
if settings.rss_enabled and any(f.enabled for f in FEEDS):
    adapters[Platform.RSS] = FeedAdapter(rate_limit=settings.rss_rate_limit)
```

The adapter then runs as its own concurrent task in `IngestionService.start()`,
polled at `settings.poll_interval_seconds`, exactly like every other adapter.

### 4.6 Frontend — `frontend/src/lib/constants.ts`

Add an `rss` entry to the `PLATFORMS` record (label `"RSS"` + a Tailwind color
class) so feed-sourced documents render with a platform badge.

### 4.7 Dependencies — `pyproject.toml`

Add:
- `feedparser` — universal pure-Python RSS/Atom parser; handles real-world feed
  variants.
- `trafilatura` — HTML main-content extraction for the full-text path.

Both are pure-Python / lxml-wheel based; neither pulls in `llvmlite`/`numba`, so
the documented macOS x86_64 / Python 3.11 build constraint does not apply.

## 5. Data flow

```
FeedAdapter.fetch()
  -> NormalizedDocument (platform="rss", content_type="article")
  -> DocumentQueue -> Redis Streams
  -> [existing] Preprocessor -> SpamDetector -> Deduplicator
  -> NER / Keywords / Events / Embedding / Sentiment / Clustering
  -> PostgreSQL + pgvector
```

Feed documents are ordinary `NormalizedDocument`s. **No downstream service
changes** are required — the only consumers that need to know about the new
platform value are the frontend label map (cosmetic) and any future
platform-specific routing.

## 6. Error handling

- **Per-feed isolation:** each feed fetch is wrapped in `try/except`; a failing
  feed is logged and skipped, the run continues.
- **Per-article isolation:** a failed article-page fetch or extraction in the
  full-text path falls back to the feed-supplied excerpt; it never aborts the
  entry.
- **`_transform()` never raises:** returns `None` for invalid data, per the
  `BaseAdapter` contract; `BaseAdapter.fetch()` counts it as filtered.
- **Rate limiting at the I/O boundary:** `_rate_limiter.acquire()` is called
  once before each HTTP request (feed fetch and article fetch), never per
  yielded item — per the project's "rate limit at I/O boundary" rule.

## 7. Testing

New file `tests/test_ingestion/test_feed_adapter.py`:

- Parse a fixture **RSS 2.0** feed and a fixture **Atom** feed → correct
  `NormalizedDocument`s.
- `_transform()` field correctness: id format, platform, content_type, title,
  timestamp parsing, author = publication, ticker extraction.
- Full-text path: `full_text=True` feed with a mocked article-page response →
  `trafilatura` extraction used; mocked failure → falls back to excerpt.
- Recency filtering: entries older than `rss_recency_days` are dropped.
- Per-feed cap: no more than `rss_max_items_per_feed` items per feed.
- In-cycle dedup: a URL listed twice yields one document.
- Malformed entry (missing URL / empty content) → `_transform()` returns `None`.
- Rate limiter is acquired once per HTTP request.

Tests follow the existing async-generator consumption pattern in
`tests/test_ingestion/test_adapters.py`.

## 8. Phase 2 (future, not in this work)

- DB-backed feed management: add `rss` to `SourcesService` /
  `SOURCE_PLATFORMS`, with a migration adding `full_text` / `category`
  metadata to the `sources` table, so feeds can be added/edited at runtime via
  the `/sources` API and admin UI.
- arXiv adapter (research preprints) if research coverage is later wanted.
