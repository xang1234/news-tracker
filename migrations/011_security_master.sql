-- Migration 008: Security master table
--
-- Adds:
--   1. pg_trgm extension for fuzzy name search
--   2. securities table with composite PK (ticker, exchange)
--   3. Indexes: ticker B-tree, name GIN trigram, figi unique partial, active partial, sector

-- 1. Enable trigram extension for fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. Securities table
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

-- 3. Indexes
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
