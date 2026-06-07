-- Migration 046: alert supporting evidence (epic o59.2, explainability)
--
-- Each alert carries the docs/claims/events that caused it (the "receipt"),
-- populated at trigger evaluation from the doc→metric attribution service.
-- JSONB so the payload shape can vary by trigger type and grow over time.

ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS supporting_evidence JSONB NOT NULL DEFAULT '{}'::jsonb;
