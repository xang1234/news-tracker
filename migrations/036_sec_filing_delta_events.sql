-- Migration 036: SEC filing-delta events
--
-- Stores point-in-time fact deltas computed from SEC Company Facts payloads.
-- Events keep accession, fact, unit, period, filed-date, fetched-at, and
-- payload lineage so downstream manifests and backtests can replay safely.

CREATE TABLE IF NOT EXISTS sec_filing_delta_events (
    event_id                    TEXT PRIMARY KEY,
    cik                         TEXT NOT NULL,
    event_type                  TEXT NOT NULL
        CHECK (event_type IN (
            'revenue_growth',
            'inventory_change',
            'capex_change',
            'rnd_change',
            'margin_compression',
            'restatement'
        )),
    accession_number            TEXT NOT NULL,
    previous_accession_number   TEXT,
    taxonomy                    TEXT NOT NULL,
    fact_name                   TEXT NOT NULL,
    unit                        TEXT NOT NULL,
    period_start                DATE,
    period_end                  DATE NOT NULL,
    previous_period_start       DATE,
    previous_period_end         DATE,
    filed_date                  DATE NOT NULL,
    previous_filed_date         DATE,
    form                        TEXT NOT NULL DEFAULT '',
    previous_form               TEXT,
    available_at                TIMESTAMPTZ NOT NULL,
    fetched_at                  TIMESTAMPTZ NOT NULL,
    current_value               NUMERIC,
    previous_value              NUMERIC,
    absolute_delta              NUMERIC,
    relative_delta              DOUBLE PRECISION,
    source_payload_hash         TEXT NOT NULL,
    source_url                  TEXT NOT NULL DEFAULT '',
    metadata                    JSONB NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sec_filing_delta_events_as_of
    ON sec_filing_delta_events (cik, available_at DESC, filed_date DESC);

CREATE INDEX IF NOT EXISTS idx_sec_filing_delta_events_fact
    ON sec_filing_delta_events (fact_name, unit, period_end DESC);

CREATE INDEX IF NOT EXISTS idx_sec_filing_delta_events_accession
    ON sec_filing_delta_events (accession_number);

DROP TRIGGER IF EXISTS update_sec_filing_delta_events_updated_at
    ON sec_filing_delta_events;
CREATE TRIGGER update_sec_filing_delta_events_updated_at
    BEFORE UPDATE ON sec_filing_delta_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
