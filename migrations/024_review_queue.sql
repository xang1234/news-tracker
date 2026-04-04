-- Migration 024: Review queue for claim ambiguity and high-impact outcomes
--
-- Review tasks capture ambiguous or risky claim outcomes that need
-- human or automated review before becoming authoritative.
--
-- Task types:
--   entity_review   — ambiguous entity resolution
--   claim_review    — low-confidence or contradictory claim
--   merge_proposal  — two concepts may be the same entity
--   split_proposal  — one concept may be multiple entities

-- ============================================================
-- 1. review_tasks — the review work queue
-- ============================================================

CREATE TABLE IF NOT EXISTS news_intel.review_tasks (
    task_id             TEXT PRIMARY KEY,
    task_type           TEXT NOT NULL
        CHECK (task_type IN ('entity_review', 'claim_review', 'merge_proposal', 'split_proposal')),
    trigger_reason      TEXT NOT NULL
        CHECK (trigger_reason IN ('low_confidence', 'close_alternatives', 'llm_proposed', 'contradiction', 'high_impact_predicate', 'manual')),
    status              TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'assigned', 'resolved', 'dismissed')),
    -- Related entities
    claim_ids           TEXT[] NOT NULL DEFAULT '{}',
    concept_ids         TEXT[] NOT NULL DEFAULT '{}',
    -- Assignment and resolution
    priority            INTEGER NOT NULL DEFAULT 2
        CHECK (priority >= 0 AND priority <= 4),
    assigned_to         TEXT,
    resolution          TEXT
        CHECK (resolution IS NULL OR resolution IN ('approved', 'rejected', 'merged', 'split', 'deferred')),
    resolution_notes    TEXT,
    -- Structured data
    payload             JSONB NOT NULL DEFAULT '{}',
    lineage             JSONB NOT NULL DEFAULT '{}',
    metadata            JSONB NOT NULL DEFAULT '{}',
    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Queue ordering: highest priority first, oldest first within same priority
CREATE INDEX IF NOT EXISTS idx_review_tasks_queue
    ON news_intel.review_tasks (priority ASC, created_at ASC)
    WHERE status IN ('pending', 'assigned');

-- Filter by task type
CREATE INDEX IF NOT EXISTS idx_review_tasks_type_status
    ON news_intel.review_tasks (task_type, status);

-- Find tasks by related claim (GIN for array containment)
CREATE INDEX IF NOT EXISTS idx_review_tasks_claim_ids
    ON news_intel.review_tasks USING GIN (claim_ids);

-- Find tasks by related concept (GIN for array containment)
CREATE INDEX IF NOT EXISTS idx_review_tasks_concept_ids
    ON news_intel.review_tasks USING GIN (concept_ids);

-- Filter by assignee
CREATE INDEX IF NOT EXISTS idx_review_tasks_assigned
    ON news_intel.review_tasks (assigned_to)
    WHERE assigned_to IS NOT NULL;

-- Trigger reason lookup
CREATE INDEX IF NOT EXISTS idx_review_tasks_trigger
    ON news_intel.review_tasks (trigger_reason, status);

-- Auto-update updated_at
DROP TRIGGER IF EXISTS update_review_tasks_updated_at ON news_intel.review_tasks;
CREATE TRIGGER update_review_tasks_updated_at
    BEFORE UPDATE ON news_intel.review_tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
