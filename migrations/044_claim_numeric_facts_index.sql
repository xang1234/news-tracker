-- Migration 044: Comparable-numeric-fact lookup index
--
-- Supports the numeric contradiction query (issue news-tracker-al3):
--   ClaimRepository.list_comparable_numeric_claims() finds active,
--   value-bearing claims sharing a (subject_concept_id, metric, period).
--
-- Deferred here from migration 043 (which added the typed columns) because
-- the index only earns its keep once a query reads these columns by group.
-- Partial on metric IS NOT NULL so it indexes only numeric-fact rows.

CREATE INDEX IF NOT EXISTS idx_claims_numeric_subject_metric_period
    ON news_intel.evidence_claims (subject_concept_id, metric, period)
    WHERE metric IS NOT NULL AND numeric_value IS NOT NULL;
