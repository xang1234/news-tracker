-- Sources table for database-backed ingestion source management.
-- Replaces hardcoded adapter lists (twitter_accounts.py, FINANCIAL_SUBREDDITS, TRACKED_PUBLICATIONS).

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
