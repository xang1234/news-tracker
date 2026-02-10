-- Migration: Add authority_profiles table for Bayesian authority scoring.
-- Tracks per-author accuracy, tier classification, topic-specific expertise,
-- and metadata for computing authority scores.

CREATE TABLE IF NOT EXISTS authority_profiles (
    author_id        TEXT NOT NULL,
    platform         TEXT NOT NULL,
    tier             TEXT NOT NULL DEFAULT 'anonymous'
                     CHECK (tier IN ('anonymous', 'verified', 'research')),
    base_weight      REAL NOT NULL DEFAULT 1.0,
    total_calls      INTEGER NOT NULL DEFAULT 0,
    correct_calls    INTEGER NOT NULL DEFAULT 0,
    first_seen       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_good_call   TIMESTAMPTZ,
    topic_scores     JSONB NOT NULL DEFAULT '{}',
    centrality_score REAL NOT NULL DEFAULT 0.0,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (author_id, platform),
    CONSTRAINT chk_calls CHECK (correct_calls <= total_calls)
);

-- Index for platform-scoped queries (e.g., list all Twitter authors)
CREATE INDEX IF NOT EXISTS idx_authority_platform
    ON authority_profiles (platform);

-- Index for finding stale profiles that need recomputation
CREATE INDEX IF NOT EXISTS idx_authority_updated_at
    ON authority_profiles (updated_at);

-- Index for leaderboard queries (highest base_weight first)
CREATE INDEX IF NOT EXISTS idx_authority_base_weight
    ON authority_profiles (base_weight DESC);
