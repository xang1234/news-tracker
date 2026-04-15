-- Migration 029: Reconcile the runtime schema with the current application model.
--
-- This migration is intentionally idempotent and non-destructive so it can
-- safely bring legacy databases up to the schema expected by the application
-- without replaying historical destructive migrations.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS authority_score REAL,
    ADD COLUMN IF NOT EXISTS embedding_minilm vector(384),
    ADD COLUMN IF NOT EXISTS entities_mentioned JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS keywords_extracted JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS events_extracted JSONB NOT NULL DEFAULT '[]';

CREATE INDEX IF NOT EXISTS idx_documents_authority_score
    ON documents(authority_score DESC)
    WHERE authority_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_documents_platform_authority
    ON documents(platform, authority_score DESC)
    WHERE authority_score IS NOT NULL AND embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_documents_entities
    ON documents USING GIN(entities_mentioned);

CREATE INDEX IF NOT EXISTS idx_documents_keywords
    ON documents USING GIN(keywords_extracted);

CREATE INDEX IF NOT EXISTS idx_documents_events_extracted
    ON documents USING GIN(events_extracted);

CREATE INDEX IF NOT EXISTS idx_documents_embedding_minilm_hnsw
    ON documents
    USING hnsw (embedding_minilm vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS themes (
    theme_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    centroid        vector(768) NOT NULL,
    top_keywords    TEXT[] NOT NULL DEFAULT '{}',
    top_tickers     TEXT[] NOT NULL DEFAULT '{}',
    top_entities    JSONB NOT NULL DEFAULT '[]',
    document_count  INTEGER NOT NULL DEFAULT 0,
    lifecycle_stage TEXT NOT NULL DEFAULT 'emerging'
        CHECK (lifecycle_stage IN ('emerging', 'accelerating', 'mature', 'fading')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}',
    deleted_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_themes_centroid_hnsw
    ON themes
    USING hnsw (centroid vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_themes_top_keywords
    ON themes USING GIN(top_keywords);

CREATE INDEX IF NOT EXISTS idx_themes_lifecycle_stage
    ON themes(lifecycle_stage);

CREATE INDEX IF NOT EXISTS idx_themes_updated_at
    ON themes(updated_at DESC);

DROP TRIGGER IF EXISTS update_themes_updated_at ON themes;
CREATE TRIGGER update_themes_updated_at
    BEFORE UPDATE ON themes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE theme_metrics
    ADD COLUMN IF NOT EXISTS weighted_volume REAL;

CREATE TABLE IF NOT EXISTS events (
    event_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL,
    actor               TEXT,
    action              TEXT NOT NULL,
    object              TEXT,
    time_ref            TEXT,
    quantity            TEXT,
    tickers             TEXT[] NOT NULL DEFAULT '{}',
    confidence          REAL NOT NULL DEFAULT 0.0,
    span_start          INTEGER NOT NULL,
    span_end            INTEGER NOT NULL,
    extractor_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_event_type
    ON events(event_type);

CREATE INDEX IF NOT EXISTS idx_events_tickers
    ON events USING GIN(tickers);

CREATE INDEX IF NOT EXISTS idx_events_doc_id
    ON events(doc_id);

CREATE INDEX IF NOT EXISTS idx_events_created_at
    ON events(created_at DESC);

CREATE TABLE IF NOT EXISTS sources (
    platform     TEXT NOT NULL,
    identifier   TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (platform, identifier)
);

CREATE INDEX IF NOT EXISTS idx_sources_platform
    ON sources(platform);

CREATE INDEX IF NOT EXISTS idx_sources_platform_active
    ON sources(platform, is_active) WHERE is_active = TRUE;

DROP TRIGGER IF EXISTS update_sources_updated_at ON sources;
CREATE TRIGGER update_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS securities (
    ticker      TEXT NOT NULL,
    exchange    TEXT NOT NULL DEFAULT 'US',
    name        TEXT NOT NULL DEFAULT '',
    aliases     TEXT[] NOT NULL DEFAULT '{}',
    sector      TEXT NOT NULL DEFAULT '',
    country     TEXT NOT NULL DEFAULT 'US',
    currency    TEXT NOT NULL DEFAULT 'USD',
    figi        TEXT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, exchange)
);

CREATE INDEX IF NOT EXISTS idx_securities_ticker
    ON securities(ticker);

CREATE INDEX IF NOT EXISTS idx_securities_name_trgm
    ON securities USING GIN (name gin_trgm_ops);

CREATE UNIQUE INDEX IF NOT EXISTS idx_securities_figi
    ON securities(figi) WHERE figi IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_securities_active
    ON securities(is_active) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_securities_sector
    ON securities(sector);

DROP TRIGGER IF EXISTS update_securities_updated_at ON securities;
CREATE TRIGGER update_securities_updated_at
    BEFORE UPDATE ON securities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
