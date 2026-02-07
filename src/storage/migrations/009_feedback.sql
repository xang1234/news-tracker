-- Feedback table for user quality ratings on themes, alerts, and documents.
-- Enables calibration of compellingness scoring and authority weights.

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id   TEXT PRIMARY KEY,
    entity_type   TEXT NOT NULL CHECK (entity_type IN ('theme', 'alert', 'document')),
    entity_id     TEXT NOT NULL,
    rating        SMALLINT NOT NULL CHECK (rating >= 1 AND rating <= 5),
    quality_label TEXT CHECK (quality_label IN ('useful', 'noise', 'too_late', 'wrong_direction')),
    comment       TEXT,
    user_id       TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fast lookups for feedback on a specific entity
CREATE INDEX IF NOT EXISTS idx_feedback_entity
    ON feedback (entity_type, entity_id);

-- Chronological listing (most recent first)
CREATE INDEX IF NOT EXISTS idx_feedback_created_at
    ON feedback (created_at DESC);

-- Partial index for filtering by user when present
CREATE INDEX IF NOT EXISTS idx_feedback_user_id
    ON feedback (user_id) WHERE user_id IS NOT NULL;
