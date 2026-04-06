-- Migration 023: Evidence claims schema
--
-- The atomic intelligence unit. Every assertion, score, graph edge,
-- and published artifact traces back to evidence claims.
--
-- Claim key is deterministic for idempotent writes:
--   claim_key = sha256(lane + source_id + subject + predicate + object)
-- Retries and replays produce the same key, so ON CONFLICT deduplicates.

-- ============================================================
-- 1. evidence_claims — atomic claim records
-- ============================================================

CREATE TABLE IF NOT EXISTS news_intel.evidence_claims (
    claim_id            TEXT PRIMARY KEY,
    claim_key           TEXT NOT NULL UNIQUE,
    -- Source lineage
    lane                TEXT NOT NULL
        CHECK (lane IN ('narrative', 'filing', 'structural', 'backtest')),
    run_id              TEXT,
    source_id           TEXT NOT NULL,
    source_type         TEXT NOT NULL DEFAULT 'document'
        CHECK (source_type IN ('document', 'filing_section', 'graph_edge', 'manual')),
    source_span_start   INTEGER,
    source_span_end     INTEGER,
    source_text         TEXT,
    -- Claim structure (subject-predicate-object)
    subject_text        TEXT NOT NULL,
    subject_concept_id  TEXT REFERENCES concepts(concept_id) ON DELETE SET NULL,
    predicate           TEXT NOT NULL,
    object_text         TEXT,
    object_concept_id   TEXT REFERENCES concepts(concept_id) ON DELETE SET NULL,
    -- Confidence and quality
    confidence          REAL NOT NULL DEFAULT 0.5
        CHECK (confidence >= 0 AND confidence <= 1),
    extraction_method   TEXT NOT NULL DEFAULT 'rule'
        CHECK (extraction_method IN ('rule', 'llm', 'hybrid', 'manual')),
    -- Temporal fields
    claim_valid_from    TIMESTAMPTZ,
    claim_valid_to      TIMESTAMPTZ,
    source_published_at TIMESTAMPTZ,
    -- Contract
    contract_version    TEXT NOT NULL,
    -- State
    status              TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'retracted', 'disputed')),
    -- Extensible
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_claims_lane_status
    ON news_intel.evidence_claims (lane, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_claims_source
    ON news_intel.evidence_claims (source_id, source_type);

CREATE INDEX IF NOT EXISTS idx_claims_subject
    ON news_intel.evidence_claims (subject_concept_id)
    WHERE subject_concept_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_claims_object
    ON news_intel.evidence_claims (object_concept_id)
    WHERE object_concept_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_claims_predicate
    ON news_intel.evidence_claims (predicate, status);

CREATE INDEX IF NOT EXISTS idx_claims_run
    ON news_intel.evidence_claims (run_id)
    WHERE run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_claims_valid_from
    ON news_intel.evidence_claims (claim_valid_from DESC)
    WHERE claim_valid_from IS NOT NULL;

DROP TRIGGER IF EXISTS update_evidence_claims_updated_at ON news_intel.evidence_claims;
CREATE TRIGGER update_evidence_claims_updated_at
    BEFORE UPDATE ON news_intel.evidence_claims
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
