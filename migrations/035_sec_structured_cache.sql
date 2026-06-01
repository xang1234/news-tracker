-- Migration 035: SEC structured payload cache
--
-- Stores official SEC Submissions and Company Facts JSON snapshots with
-- issuer CIK, payload hash, and accession lineage for point-in-time reuse.

CREATE TABLE IF NOT EXISTS sec_structured_payloads (
    id                  BIGSERIAL PRIMARY KEY,
    cik                 TEXT NOT NULL
        CHECK (cik ~ '^[0-9]{10}$'),
    payload_type        TEXT NOT NULL
        CHECK (payload_type IN ('companyfacts', 'submissions')),
    source_url          TEXT NOT NULL,
    payload_hash        TEXT NOT NULL,
    payload             JSONB NOT NULL,
    accession_numbers   TEXT[] NOT NULL DEFAULT '{}',
    fetched_at          TIMESTAMPTZ NOT NULL,
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (cik, payload_type, payload_hash)
);

CREATE INDEX IF NOT EXISTS idx_sec_structured_payloads_cik_type_seen
    ON sec_structured_payloads (cik, payload_type, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_sec_structured_payloads_accessions
    ON sec_structured_payloads USING GIN (accession_numbers);

DROP TRIGGER IF EXISTS update_sec_structured_payloads_updated_at
    ON sec_structured_payloads;
CREATE TRIGGER update_sec_structured_payloads_updated_at
    BEFORE UPDATE ON sec_structured_payloads
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
