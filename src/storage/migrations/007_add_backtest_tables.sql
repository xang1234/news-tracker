-- Migration 007: Add backtest infrastructure tables
--
-- Adds:
--   1. deleted_at column on themes (soft deletes for point-in-time queries)
--   2. model_versions table (tracks embedding + clustering config snapshots)
--   3. backtest_runs table (audit log for backtest executions)
--   4. price_cache table (OHLCV cache keyed by ticker + date)
--   5. Index on documents.fetched_at (point-in-time document queries)

-- 1. Soft delete support for themes
ALTER TABLE themes ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

-- 2. Model version tracking
CREATE TABLE IF NOT EXISTS model_versions (
    version_id      TEXT PRIMARY KEY,
    embedding_model  TEXT NOT NULL,
    clustering_config JSONB NOT NULL DEFAULT '{}',
    config_snapshot  JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description      TEXT
);

-- 3. Backtest run audit log
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id            TEXT PRIMARY KEY,
    model_version_id  TEXT NOT NULL REFERENCES model_versions(version_id),
    date_range_start  DATE NOT NULL,
    date_range_end    DATE NOT NULL,
    parameters        JSONB NOT NULL DEFAULT '{}',
    results           JSONB,
    status            TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at      TIMESTAMPTZ,
    error_message     TEXT
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_status
    ON backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_created_at
    ON backtest_runs(created_at DESC);

-- 4. Price data cache (OHLCV)
CREATE TABLE IF NOT EXISTS price_cache (
    ticker     TEXT NOT NULL,
    date       DATE NOT NULL,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     BIGINT NOT NULL DEFAULT 0,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_price_cache_ticker
    ON price_cache(ticker);

-- 5. Index for point-in-time document queries on fetched_at
CREATE INDEX IF NOT EXISTS idx_documents_fetched_at
    ON documents(fetched_at DESC);
