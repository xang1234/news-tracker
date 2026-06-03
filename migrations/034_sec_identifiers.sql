-- Migration 034: SEC issuer identifiers on security master
--
-- Adds SEC CIK, issuer names, former names, external identifiers, and
-- JSONB lineage so downstream SEC Company Facts/submissions ingestion can
-- explain how a security was mapped to an issuer.

ALTER TABLE securities ADD COLUMN IF NOT EXISTS sec_cik TEXT;

ALTER TABLE securities ADD COLUMN IF NOT EXISTS issuer_name TEXT NOT NULL DEFAULT '';

ALTER TABLE securities ADD COLUMN IF NOT EXISTS former_names TEXT[] NOT NULL DEFAULT '{}';

ALTER TABLE securities ADD COLUMN IF NOT EXISTS external_identifiers JSONB NOT NULL DEFAULT '{}';

ALTER TABLE securities ADD COLUMN IF NOT EXISTS identifier_lineage JSONB NOT NULL DEFAULT '[]';

CREATE INDEX IF NOT EXISTS idx_securities_sec_cik
    ON securities(sec_cik) WHERE sec_cik IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_securities_issuer_name_trgm
    ON securities USING GIN (issuer_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_securities_former_names_gin
    ON securities USING GIN (former_names);
