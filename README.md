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

# Model selection
EMBEDDING_MODEL_NAME=ProsusAI/finbert
SENTIMENT_MODEL_NAME=ProsusAI/finbert
NER_SPACY_MODEL=en_core_web_trf

# Processing thresholds
SPAM_THRESHOLD=0.7
DUPLICATE_THRESHOLD=0.85
```

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
