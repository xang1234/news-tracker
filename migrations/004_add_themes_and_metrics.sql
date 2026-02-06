-- Migration 004: Add themes and theme_metrics tables
-- Persists clustering results from BERTopicService for theme tracking,
-- similarity search on theme centroids, and daily metric time series.

-- ============================================================
-- Table: themes
-- Stores discovered topic clusters with centroid embeddings
-- ============================================================
CREATE TABLE IF NOT EXISTS themes (
    -- Identity (deterministic: theme_{sha256(sorted_topic_words)[:12]})
    theme_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,           -- e.g. "gpu_nvidia_architecture"
    description     TEXT,                    -- optional human-readable summary

    -- Centroid embedding (FinBERT 768-dim mean of assigned documents)
    centroid        vector(768) NOT NULL,

    -- Topic representation
    top_keywords    TEXT[] NOT NULL DEFAULT '{}',    -- ranked topic words
    top_tickers     TEXT[] NOT NULL DEFAULT '{}',    -- most-mentioned tickers
    top_entities    JSONB NOT NULL DEFAULT '[]',     -- entity objects with scores

    -- Stats
    document_count  INTEGER NOT NULL DEFAULT 0,
    lifecycle_stage TEXT NOT NULL DEFAULT 'emerging'
        CHECK (lifecycle_stage IN ('emerging', 'accelerating', 'mature', 'fading')),

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Flexible storage (bertopic_topic_id, merged_from, etc.)
    metadata        JSONB NOT NULL DEFAULT '{}'
);

-- HNSW index for centroid similarity search (find related themes)
CREATE INDEX IF NOT EXISTS idx_themes_centroid_hnsw
    ON themes
    USING hnsw (centroid vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for keyword array containment queries
CREATE INDEX IF NOT EXISTS idx_themes_top_keywords
    ON themes USING GIN(top_keywords);

-- Filter by lifecycle stage
CREATE INDEX IF NOT EXISTS idx_themes_lifecycle_stage
    ON themes(lifecycle_stage);

-- Recency queries
CREATE INDEX IF NOT EXISTS idx_themes_updated_at
    ON themes(updated_at DESC);

-- Reuse the update_updated_at_column() trigger function from 001
DROP TRIGGER IF EXISTS update_themes_updated_at ON themes;
CREATE TRIGGER update_themes_updated_at
    BEFORE UPDATE ON themes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Table: theme_metrics
-- Daily time-series metrics per theme for trend analysis
-- ============================================================
CREATE TABLE IF NOT EXISTS theme_metrics (
    theme_id        TEXT NOT NULL REFERENCES themes(theme_id) ON DELETE CASCADE,
    date            DATE NOT NULL,

    -- Volume & sentiment
    document_count  INTEGER NOT NULL DEFAULT 0,
    sentiment_score REAL,           -- aggregate sentiment (-1.0 to 1.0)
    volume_zscore   REAL,           -- standard deviations from mean volume
    velocity        REAL,           -- rate of volume change
    acceleration    REAL,           -- rate of velocity change

    -- Quality signals
    avg_authority   REAL,           -- mean authority_score of documents
    bullish_ratio   REAL,           -- fraction of positive sentiment docs

    PRIMARY KEY (theme_id, date)
);

-- Time-range scans across all themes
CREATE INDEX IF NOT EXISTS idx_theme_metrics_date
    ON theme_metrics(date);
