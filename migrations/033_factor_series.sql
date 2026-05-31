-- Migration 033: Add factor registry and point-in-time observations
--
-- Adds:
--   1. factor_series: curated registry for macro and supply-chain series
--   2. factor_observations: provider values with availability/fetch lineage

CREATE TABLE IF NOT EXISTS factor_series (
    factor_id        TEXT PRIMARY KEY,
    provider         TEXT NOT NULL,
    external_id      TEXT NOT NULL,
    name             TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    units            TEXT NOT NULL,
    cadence          TEXT NOT NULL CHECK (
        cadence IN ('daily', 'weekly', 'monthly', 'quarterly', 'annual', 'irregular')
    ),
    release_lag_days INTEGER NOT NULL DEFAULT 0 CHECK (release_lag_days >= 0),
    relevance_tags   TEXT[] NOT NULL DEFAULT '{}',
    required_credentials TEXT[] NOT NULL DEFAULT '{}',
    source_url       TEXT,
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    metadata         JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (factor_id, units)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_factor_series_provider_external_id
    ON factor_series(provider, external_id);

CREATE INDEX IF NOT EXISTS idx_factor_series_provider
    ON factor_series(provider);

CREATE INDEX IF NOT EXISTS idx_factor_series_relevance_tags
    ON factor_series USING GIN(relevance_tags);

CREATE INDEX IF NOT EXISTS idx_factor_series_active
    ON factor_series(is_active) WHERE is_active = TRUE;

CREATE TABLE IF NOT EXISTS factor_observations (
    factor_id        TEXT NOT NULL REFERENCES factor_series(factor_id) ON DELETE CASCADE,
    observation_date DATE NOT NULL,
    value            DOUBLE PRECISION,
    units            TEXT NOT NULL,
    available_at     TIMESTAMPTZ NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revision         TEXT NOT NULL DEFAULT '',
    missing_reason   TEXT,
    metadata         JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (factor_id, observation_date, available_at, revision),
    FOREIGN KEY (factor_id, units) REFERENCES factor_series(factor_id, units)
        ON DELETE CASCADE,
    CHECK (
        (value IS NOT NULL AND missing_reason IS NULL)
        OR (value IS NULL AND missing_reason IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_factor_observations_as_of
    ON factor_observations(factor_id, observation_date, available_at DESC, fetched_at DESC);

CREATE INDEX IF NOT EXISTS idx_factor_observations_fetched_at
    ON factor_observations(fetched_at DESC);

DROP TRIGGER IF EXISTS update_factor_series_updated_at ON factor_series;
CREATE TRIGGER update_factor_series_updated_at
    BEFORE UPDATE ON factor_series
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
