-- Migration 025: Claim dead-letter table for unrecoverable extraction failures
--
-- Dead-letter records capture extraction failures with full context
-- for replay and diagnosis. Claims that fail structural quality checks
-- or encounter extraction errors land here instead of being silently
-- dropped.
--
-- Failure reasons:
--   quality_check_failed  — claim failed structural/semantic checks
--   extraction_error      — extractor threw an unhandled exception
--   parse_error           — source text could not be parsed
--   validation_error      — claim data violated schema constraints
--   timeout               — extraction exceeded time budget

-- ============================================================
-- 1. claim_dead_letters — failed extraction records
-- ============================================================

CREATE TABLE IF NOT EXISTS news_intel.claim_dead_letters (
    record_id           TEXT PRIMARY KEY,
    lane                TEXT NOT NULL
        CHECK (lane IN ('narrative', 'filing', 'structural', 'backtest')),
    run_id              TEXT NOT NULL,
    source_id           TEXT NOT NULL,
    reason              TEXT NOT NULL
        CHECK (reason IN ('quality_check_failed', 'extraction_error', 'parse_error', 'validation_error', 'timeout')),
    error_message       TEXT NOT NULL,
    error_detail        JSONB NOT NULL DEFAULT '{}',
    -- Source context for replay
    source_text         TEXT,
    claim_snapshot      JSONB,
    -- Extensible
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Find failures by run (for post-run analysis)
CREATE INDEX IF NOT EXISTS idx_claim_dl_run
    ON news_intel.claim_dead_letters (run_id, created_at DESC);

-- Find failures by lane and reason (for monitoring dashboards)
CREATE INDEX IF NOT EXISTS idx_claim_dl_lane_reason
    ON news_intel.claim_dead_letters (lane, reason, created_at DESC);

-- Find failures by source (for source-level replay)
CREATE INDEX IF NOT EXISTS idx_claim_dl_source
    ON news_intel.claim_dead_letters (source_id);
