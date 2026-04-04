-- Migration 026: Resolved assertions and claim links
--
-- Assertions are the aggregation layer between raw evidence claims
-- and downstream consumers (graph, scoring, publishing). Each
-- assertion represents a stable current-belief about a subject-
-- predicate-object triple, backed by explicit support and
-- contradiction links.
--
-- Assertion ID is deterministic from the triple, so multiple claims
-- about the same relationship feed into one assertion.

-- ============================================================
-- 1. resolved_assertions — stable current-belief objects
-- ============================================================

CREATE TABLE IF NOT EXISTS news_intel.resolved_assertions (
    assertion_id        TEXT PRIMARY KEY,
    -- The triple
    subject_concept_id  TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    predicate           TEXT NOT NULL,
    object_concept_id   TEXT REFERENCES concepts(concept_id) ON DELETE SET NULL,
    -- Aggregate confidence
    confidence          REAL NOT NULL DEFAULT 0.0
        CHECK (confidence >= 0 AND confidence <= 1),
    status              TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'disputed', 'retracted', 'superseded')),
    -- Validity window (bitemporal)
    valid_from          TIMESTAMPTZ,
    valid_to            TIMESTAMPTZ,
    -- Evidence counts
    support_count       INTEGER NOT NULL DEFAULT 0,
    contradiction_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at       TIMESTAMPTZ,
    last_evidence_at    TIMESTAMPTZ,
    source_diversity    INTEGER NOT NULL DEFAULT 0,
    -- Extensible
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Find assertions by subject concept
CREATE INDEX IF NOT EXISTS idx_assertions_subject
    ON news_intel.resolved_assertions (subject_concept_id, status);

-- Find assertions by object concept
CREATE INDEX IF NOT EXISTS idx_assertions_object
    ON news_intel.resolved_assertions (object_concept_id)
    WHERE object_concept_id IS NOT NULL;

-- Find assertions by predicate
CREATE INDEX IF NOT EXISTS idx_assertions_predicate
    ON news_intel.resolved_assertions (predicate, status);

-- High-confidence active assertions (for graph/scoring consumers)
CREATE INDEX IF NOT EXISTS idx_assertions_active_confidence
    ON news_intel.resolved_assertions (confidence DESC, last_evidence_at DESC)
    WHERE status = 'active';

-- Disputed assertions needing attention
CREATE INDEX IF NOT EXISTS idx_assertions_disputed
    ON news_intel.resolved_assertions (contradiction_count DESC)
    WHERE status = 'disputed';

-- Auto-update updated_at
DROP TRIGGER IF EXISTS update_resolved_assertions_updated_at ON news_intel.resolved_assertions;
CREATE TRIGGER update_resolved_assertions_updated_at
    BEFORE UPDATE ON news_intel.resolved_assertions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 2. assertion_claim_links — evidence backing each assertion
-- ============================================================

CREATE TABLE IF NOT EXISTS news_intel.assertion_claim_links (
    assertion_id        TEXT NOT NULL
        REFERENCES news_intel.resolved_assertions(assertion_id) ON DELETE CASCADE,
    claim_id            TEXT NOT NULL
        REFERENCES news_intel.evidence_claims(claim_id) ON DELETE CASCADE,
    link_type           TEXT NOT NULL
        CHECK (link_type IN ('support', 'contradiction')),
    contribution_weight REAL NOT NULL DEFAULT 1.0
        CHECK (contribution_weight >= 0 AND contribution_weight <= 1),
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Composite PK: one link per (assertion, claim) pair
    PRIMARY KEY (assertion_id, claim_id)
);

-- Find all claims backing an assertion
CREATE INDEX IF NOT EXISTS idx_acl_assertion_type
    ON news_intel.assertion_claim_links (assertion_id, link_type);

-- Find all assertions a claim contributes to
CREATE INDEX IF NOT EXISTS idx_acl_claim
    ON news_intel.assertion_claim_links (claim_id);
