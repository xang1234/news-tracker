# News Tracker

News Tracker is a producer-side intelligence system for semiconductors: it ingests multi-source evidence, resolves claims into assertions, builds lane outputs (narrative, filing, structural, backtest), and publishes manifest-keyed objects for downstream consumers.

## Why Teams Use This

- Reduce time-to-insight from raw documents to explainable intelligence artifacts.
- Compare narrative momentum vs filing-confirmed adoption before making decisions.
- Surface second-order exposure paths with auditable structural rationale.
- Replay runs and publication state point-in-time through manifests and lineage.

## Quick Start (Utility-First)

### 1) Boot Local Stack

```bash
uv sync --extra dev
docker compose up -d
uv run news-tracker init-db
```

### 2) Create Reproducible Mock State

```bash
uv run news-tracker run-once --mock
uv run news-tracker graph seed

# Optional (for non-empty Themes view screenshots)
uv run news-tracker run-once --mock --with-embeddings
uv run news-tracker daily-clustering --date YYYY-MM-DD
```

### 3) Run API + Frontend

```bash
uv run news-tracker serve
cd frontend && npm install && npm run dev
```

### 4) Capture UI Evidence with Playwright

```bash
FRONTEND_PORT=5151   # If occupied, use the actual Vite port from startup logs
mkdir -p output/playwright
npx --yes playwright screenshot "http://localhost:${FRONTEND_PORT}/" output/playwright/dashboard.png
npx --yes playwright screenshot "http://localhost:${FRONTEND_PORT}/themes" output/playwright/themes.png
npx --yes playwright screenshot "http://localhost:${FRONTEND_PORT}/graph" output/playwright/graph.png
```

## Core Operator Views

### Dashboard (System + Lane Signal)

![Dashboard overview](output/playwright/dashboard.png)

Use this for quick triage:

- Is ingestion/processing moving?
- Are lane health states publish-ready?
- Are queue backlogs stable?

### Themes (Narrative Discovery Surface)

![Themes view](output/playwright/themes.png)

Use this to inspect theme volume, lifecycle stage, and ranking-oriented context. If empty, run clustering jobs and confirm ingestion throughput.

### Graph (Structural Exposure Surface)

![Graph view](output/playwright/graph.png)

Use this to explore two complementary graph layers:

- Manual causal graph (`causal_nodes` / `causal_edges`): node types `ticker`, `theme`, `technology`; edge relations `depends_on`, `supplies_to`, `competes_with`, `drives`, `blocks`.
- Assertion-derived structural relations (`src/graph/structural.py`): broader concept predicates (for example `customer_of`, `uses_technology`, `component_of`) with sign, freshness, corroboration, and assertion lineage. During `news-tracker graph sync`, these map into causal-edge relations for traversal APIs.

## Q88 Producer Boundary (Authoritative Architecture)

The system is documented around this producer contract:

1. Source documents and filings produce evidence claims.
2. Claims are resolved into assertions (current-belief layer).
3. Assertions feed lane-specific computations.
4. Lane outputs are published as manifest-keyed objects.
5. Downstream consumers read published surfaces, not WIP tables.

Key publication concepts:

- `news_intel.lane_runs`: lane execution lifecycle.
- `intel_pub.manifests`: versioned publication units.
- `intel_pub.manifest_pointers`: active serving pointer per lane.
- `intel_pub.published_objects`: published, lineaged payloads.
- `intel_pub.read_model`: stable consumer read surface.

## Data Science / ML Techniques by Lane

| Lane | Techniques | Primary Outputs | Why It Matters |
|---|---|---|---|
| Narrative | Component scoring: attention, corroboration, confirmation, novelty/persistence | Narrative run payloads, rollups, signals | Distinguishes real cross-platform momentum from noise |
| Filing | Section-weighted adoption scoring, fact alignment, temporal consistency, divergence classification | Adoption payloads, divergence alerts, issuer summaries | Tests whether narrative claims are operationally reflected in filings |
| Structural | Assertion-derived typed relations, 1/2-hop path scoring, basket assembly | Path explanations, basket summaries, structural relations | Surfaces second-order beneficiaries/risks with traceable rationale |
| Backtest | Point-in-time replay over published states | Backtest/evaluation artifacts | Validates decision utility under historical constraints |

### Narrative Lane Scoring

Narrative scoring is decomposed into four inspectable components:

- Attention: velocity + acceleration + doc mass.
- Corroboration: platform spread + source diversity + spread speed.
- Confirmation: authority alignment + crowd agreement.
- Novelty/Persistence: recency decay vs duration persistence.

Composite scores are weighted and capped, then exposed for ranking/explanation workflows.

### Filing Lane Scoring

Filing adoption score combines:

- Section coverage
- Section depth
- Fact alignment
- Temporal consistency

Divergence logic classifies structured reason codes such as:

- `narrative_without_filing`
- `filing_without_narrative`
- `adverse_drift`
- `contradictory_drift`
- `lagging_adoption`

### Structural Lane Scoring

Structural path scoring is explanation-first, not opaque graph embedding:

- Edge score: `confidence * freshness * corroboration`
- Path score: product of edge scores with hop decay
- Path sign: product of edge signs

Path outputs keep decomposed factors and assertion lineage.

## Explainability: “Why Did This Surface?”

This system is designed so surfaced outputs can be audited without recomputing everything live.

### Assertion-Level Explainability

Resolved assertions expose confidence context via:

- top-level fields: `support_count`, `contradiction_count`, `source_diversity`, `valid_from`, `valid_to`, `first_seen_at`, `last_evidence_at`
- `metadata.breakdown`: `base`, `freshness`, `diversity`, `support_ratio`, `review_bonus`

### Structural Path Explainability

Published path explanations include:

- `hops`, `path_score`, `path_sign`
- `confidence_product`, `freshness_product`, `corroboration_product`, `hop_decay`
- `assertion_ids` and edge predicate sequence

### Filing Divergence Explainability

Divergence payloads include reason code, severity, human-readable summary, and structured evidence fields for UI and audit workflows.

## Data Flow (Practical)

1. Adapters fetch and normalize source content.
2. Processing pipeline runs spam filtering, deduplication, extraction/enrichment.
3. Lane logic computes narrative/filing/structural outputs.
4. Publish layer creates/updates manifests and object state.
5. Consumers query published objects/read-model surfaces.

## Datasource Coverage

Current source families include:

- Social/news ingestion: X/Twitter, Reddit, Substack, and news adapters.
- RSS/Atom ingestion: a generic `FeedAdapter` ingests curated RSS 2.0 and Atom feeds as normal `NormalizedDocument` article records with source category, authority, full-text mode, feed health, conditional GET lineage, recency limits, per-feed caps, and per-run dedupe. The seeded catalog covers first-party company/IR and technical feeds from NVIDIA, AMD, Intel, Micron, KLA, and Samsung; semiconductor trade feeds from Semiconductor Engineering, EE Times, Semiconductor Digest, Semiconductor Today, and SemiWiki; and broad technology feeds from Tom's Hardware, The Verge, TechCrunch, and Ars Technica. Sources seeding converts the static catalog into `platform="rss"` rows with `metadata.url`, `metadata.category`, `metadata.authority`, and `metadata.full_text`, so operators can list, create, or deactivate RSS feeds through the existing Sources API without code changes. RSS health is exposed through `GET /sources/rss/health` and the Settings sources table, including active/stale/failing/inactive status, last fetch time, recent document count, and latest error. Run `uv run python scripts/validate_feeds.py` to catch malformed, duplicate, empty, or dead feed URLs before deploying catalog changes.
- SEC filing lane: EDGAR filing search/fetch providers with centralized SEC fair-access policy.
- Security master: ticker, exchange, alias, FIGI, and SEC issuer identifiers. Seeded semiconductor securities now carry SEC CIKs where available, SEC issuer names, former issuer-name slots, external identifier maps, and identifier-lineage records so Company Facts and submissions ingestion can audit how a ticker was mapped to an SEC issuer.
- Nasdaq Trader symbol directories: the free Nasdaq Trader `nasdaqlisted.txt` and `otherlisted.txt` reference files can be ingested into the security master with `uv run news-tracker ingest-nasdaq-trader`. The parser preserves market category, listing-exchange code/name, financial status, round-lot size, ETF flags, test-issue flags, ACT/CQS/Nasdaq symbols, file-creation timestamps, and unexpected extra columns under `external_identifiers.nasdaq_trader`. Current non-test issues remain active, test issues are explicitly inactive, and previously Nasdaq-sourced symbols missing from the latest files are deactivated without deleting SEC CIKs or curated aliases.
- SEC structured fundamentals: official `data.sec.gov` Submissions and XBRL Company Facts JSON for tracked issuer CIKs, cached by issuer, payload hash, source URL, and accession-number lineage. The provider uses declared SEC User-Agent headers, per-second fair-access rate limiting, retry handling for transient SEC failures, and exposes the SEC nightly bulk archive URLs for backfills.
- SEC filing-delta events: point-in-time deltas derived from cached Company Facts for revenue, inventory, capex, R&D, gross-margin compression, and restatement/amended-filing changes. Events preserve accession, fact, unit, period, filed-date, fetched-at, source payload hash, and availability lineage for backtests and publication manifests.
- SEC ownership events: Form 4, Schedule 13D/G, and 13F-HR filings can be parsed into structured `sec_ownership_events` rows instead of ambiguous news documents. Form 4 non-derivative and derivative transactions preserve transaction codes, acquired/disposed flags, prices, post-transaction holdings, and derivative underlying shares. Schedule 13D/G rows capture beneficial-ownership percentages and share counts, while 13F rows capture CUSIP-keyed quarterly positions, value, share counts, and point-in-time position deltas when a prior quarter is available. Events retain accession, filer, issuer mapping status, filing date, availability, source URL, amendment flags, and raw metadata for downstream guardrails.
- Market-structure stress events: FINRA daily short-sale volume files and SEC fails-to-deliver files can be ingested with `uv run news-tracker ingest-market-structure` into structured `market_structure_events` rows. FINRA rows retain trade date, reporting facility, short-sale volume, short-exempt volume, total volume, ratio signals, and point-in-time availability based on FINRA's same-day post-close publication window. SEC fails-to-deliver rows retain settlement date, CUSIP, issuer description, fail quantity, prior close price, notional estimate, source file lineage, and fetch-time availability. Derived fields distinguish short-volume ratios, fails-to-deliver notional thresholds, and persistence streaks without treating either source as short-interest position data or evidence of abusive short selling.
- Market-plumbing publication and replay: ownership and market-structure events can be converted into alert/read-model payloads with distinct labels for insider ownership, activist ownership, institutional holdings, short-volume anomalies, and fails-to-deliver anomalies. Short-volume and FTD payloads carry visible interpretation guardrails, while point-in-time replay delegates to repository `available_at` filters instead of event dates alone.
- SEC delta publication and replay: filing manifests can publish SEC fact-delta payloads as `filing_fact` objects with stable reason codes, while narrative and structural payloads can attach SEC fact lineage as corroborating or contradictory evidence. Backtests can replay SEC delta availability through point-in-time `available_at` queries.
- Innovation identity mapping: the concept registry supports patent/research alias resolution for issuer names, subsidiaries, acquired entities, labs, research institutions, and abbreviations before PatentsView, OpenAlex, or arXiv events are joined to securities. Alias rows retain confidence, source attribution, review status, review notes, and metadata so ambiguous assignee or affiliation matches remain auditable instead of being collapsed into a single company too early.
- USPTO patent innovation signals: PatentsView-transition ingestion uses the free USPTO Open Data Portal patent/application search API when configured with an ODP API key, and can also normalize PatentsView/ODP bulk snapshot rows for offline backfills. Patent records preserve assignees, application/grant dates, CPC/IPC classifications, source lineage, fetched-at timestamps, snapshot staleness metadata, and family-level dedupe so duplicate applications and grants do not double-count the same invention. Linked signals require issuer-to-security mappings, retain ambiguous assignee candidates with confidence/review metadata, and attach configured CPC/IPC classes to innovation themes as slower-moving structural evidence.
- Research innovation signals: OpenAlex Works metadata and arXiv Atom search feeds can be normalized into research/preprint records for curated semiconductor, AI infrastructure, materials, and EDA topics. Records retain titles, abstracts, authors, institutions, OpenAlex topics, arXiv categories, DOI/arXiv identifiers, publication dates, URLs, fetched-at timestamps, provider lineage, cursor/start pagination metadata, rate-limit hook usage, and duplicate DOI/arXiv suppression. Linked signals require institution-to-issuer and issuer-to-security mappings, skip weak issuer matches below the configured confidence floor, and map curated topics/categories to themes as slower-moving innovation evidence. Published structural and narrative outputs expose patents/research in separate `innovation_evidence` arrays with conservative low/medium confidence labels so users can inspect technology momentum without conflating it with news/social momentum or near-term catalysts.
- Macro and supply-chain factors: FRED, BLS, BEA, Treasury Fiscal Data, Federal Reserve CSV, EIA, and Census sources for point-in-time ranking/backtest context.

## API Surfaces

- Infrastructure/publish endpoints: `src/api/routes/intel.py`
- User-facing intelligence endpoints: `src/api/routes/intel_surface.py`
- Graph endpoints: `src/api/routes/graph.py`

Start docs:

- API docs: `http://localhost:8001/docs`

## CLI Reference (Most Used)

```bash
# Core runtime
news-tracker serve
news-tracker worker
news-tracker init-db
news-tracker health

# Mock ingestion and cleanup
news-tracker run-once --mock
news-tracker cleanup --days 90 --dry-run

# Clustering and ranking workflows
news-tracker daily-clustering --date YYYY-MM-DD
news-tracker cluster status

# Graph and monitoring
news-tracker graph seed
news-tracker drift check-quick
news-tracker drift check-daily
news-tracker drift report

# Backtesting
news-tracker backtest run --start YYYY-MM-DD --end YYYY-MM-DD --strategy swing
news-tracker backtest plot --run-id <id>
```

## Configuration Essentials

```bash
# Infrastructure
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/news_tracker
REDIS_URL=redis://localhost:6379/0
API_KEYS=key1,key2

# X/Twitter ingestion
TWITTER_BEARER_TOKEN=...
TWITTER_XUI_ENABLED=false
XUI_INSTALL=false

# Model selection
EMBEDDING_MODEL_NAME=ProsusAI/finbert
SENTIMENT_MODEL_NAME=ProsusAI/finbert
NER_SPACY_MODEL=en_core_web_trf

# Processing thresholds
SPAM_THRESHOLD=0.7
DUPLICATE_THRESHOLD=0.85

# RSS/Atom ingestion (opt in)
RSS_ENABLED=false
RSS_RATE_LIMIT=20
RSS_MAX_ITEMS_PER_FEED=50
RSS_RECENCY_DAYS=7
RSS_FULL_TEXT_ENABLED=true

# Validate the curated RSS/Atom source catalog
uv run python scripts/validate_feeds.py --skip-live
uv run python scripts/validate_feeds.py

# Example DB-backed RSS source row
platform=rss
identifier=semiconductor-engineering
metadata.url=https://semiengineering.com/feed/
metadata.category=trade_press
metadata.authority=specialist
metadata.full_text=true

# Ingest free Nasdaq Trader listed-security reference files
uv run news-tracker ingest-nasdaq-trader
uv run news-tracker ingest-nasdaq-trader \
  --nasdaq-listed-file /path/to/nasdaqlisted.txt \
  --other-listed-file /path/to/otherlisted.txt

# Ingest local FINRA daily short-volume and SEC fails-to-deliver files
uv run news-tracker ingest-market-structure \
  --finra-short-volume-file /path/to/CNMSshvolYYYYMMDD.txt \
  --sec-fails-file /path/to/cnsfailsYYYYMMa.txt \
  --fetched-at 2026-06-15T00:00:00Z
```

The official X API is the primary Twitter/X ingestion path when `TWITTER_BEARER_TOKEN` is
configured. The private xui path is an explicit fallback: set `TWITTER_XUI_ENABLED=true`
at runtime and `XUI_INSTALL=true` at image build time to use it.

Feature flags are opt-in and grouped by subsystem (`*_ENABLED`). See `src/config/settings.py` for full settings.

## Observability and Reliability

- Tracing: OpenTelemetry with OTLP export.
- Metrics: Prometheus + Grafana dashboards.
- Logging: structured logs with request correlation.
- Lane health semantics: freshness, quality, quarantine, publish readiness.

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run pytest tests/ -v -m "not integration"
```

Project layout highlights:

- `src/contracts/intelligence/`: producer contract definitions
- `src/publish/`: manifest/pointer/object lifecycle and export
- `src/assertions/`: aggregation, derived edges, recompute
- `src/narrative/`, `src/filing/`, `src/graph/`: lane-specific methods
- `src/api/`: REST/WebSocket routes
- `frontend/`: React app and domain views

## Roadmap / In Progress

- Tightening end-to-end lane publication orchestration across all lanes.
- Expanding parity between lane payload producers and `intel_surface` consumer expectations.
- Continuing migration of mixed UX reads to published-object surfaces where WIP table access still exists.
- Improving screenshot/data fixtures for richer non-empty local demo states.
- Continuing contract-level hardening and replay validations for operational cutover.

## License

MIT
