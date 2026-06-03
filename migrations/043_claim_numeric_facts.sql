-- Migration 043: Typed numeric facts on evidence claims
--
-- Promotes quantities from free-text metadata to first-class, comparable
-- columns so numeric facts can be deduplicated and contradiction-checked:
--   metric        — the value-type measured (capex, capacity, price, guidance, ...)
--   numeric_value — normalized magnitude in base units (e.g. 42e9 for "$42 billion")
--   unit          — canonical unit (USD, %, nm, count, weeks, months, days)
--   period        — normalized time period the value applies to (e.g. "2026-Q3")
--   modality      — epistemic status (confirmed, guided, rumored, estimate)
--
-- All columns are nullable: non-numeric (relationship) claims leave them NULL.

ALTER TABLE news_intel.evidence_claims
    ADD COLUMN IF NOT EXISTS metric        TEXT,
    ADD COLUMN IF NOT EXISTS numeric_value DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS unit          TEXT,
    ADD COLUMN IF NOT EXISTS period        TEXT,
    ADD COLUMN IF NOT EXISTS modality      TEXT
        CHECK (modality IS NULL OR modality IN ('confirmed', 'guided', 'rumored', 'estimate'));

-- NOTE: the (subject_concept_id, metric, period) lookup index lands with the
-- contradiction query that needs it (see issue news-tracker-al3), not here —
-- no query reads these columns by group yet.
