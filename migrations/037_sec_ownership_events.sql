-- Migration 037: SEC ownership events
--
-- Stores structured ownership events parsed from Form 4, Schedule 13D/G,
-- and 13F-HR filings. Events preserve issuer/filer mapping status, accession
-- lineage, point-in-time availability, and raw filing source URL.

CREATE TABLE IF NOT EXISTS sec_ownership_events (
    event_id                            TEXT PRIMARY KEY,
    event_type                          TEXT NOT NULL
        CHECK (event_type IN (
            'form4_non_derivative_transaction',
            'form4_derivative_transaction',
            'schedule_13d_ownership',
            'schedule_13g_ownership',
            '13f_position'
        )),
    accession_number                    TEXT NOT NULL,
    filing_type                         TEXT NOT NULL,
    filed_date                          DATE NOT NULL,
    issuer_cik                          TEXT NOT NULL DEFAULT '',
    issuer_name                         TEXT NOT NULL DEFAULT '',
    issuer_ticker                       TEXT,
    filer_cik                           TEXT NOT NULL DEFAULT '',
    filer_name                          TEXT NOT NULL DEFAULT '',
    security_title                      TEXT NOT NULL DEFAULT '',
    transaction_code                    TEXT,
    transaction_date                    DATE,
    transaction_shares                  NUMERIC,
    transaction_price_per_share         NUMERIC,
    transaction_acquired_disposed_code  TEXT,
    shares_owned_following              NUMERIC,
    derivative_underlying_shares        NUMERIC,
    ownership_percent                   NUMERIC,
    position_cusip                      TEXT,
    position_shares                     NUMERIC,
    position_value_usd                  NUMERIC,
    previous_position_shares            NUMERIC,
    position_delta_shares               NUMERIC,
    is_amendment                        BOOLEAN NOT NULL DEFAULT FALSE,
    available_at                        TIMESTAMPTZ NOT NULL,
    fetched_at                          TIMESTAMPTZ,
    source_url                          TEXT NOT NULL DEFAULT '',
    metadata                            JSONB NOT NULL DEFAULT '{}',
    created_at                          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sec_ownership_events_issuer_as_of
    ON sec_ownership_events (issuer_cik, available_at DESC, filed_date DESC);

CREATE INDEX IF NOT EXISTS idx_sec_ownership_events_filer
    ON sec_ownership_events (filer_cik, available_at DESC);

CREATE INDEX IF NOT EXISTS idx_sec_ownership_events_accession
    ON sec_ownership_events (accession_number);

CREATE INDEX IF NOT EXISTS idx_sec_ownership_events_position_cusip
    ON sec_ownership_events (position_cusip, available_at DESC)
    WHERE position_cusip IS NOT NULL;

DROP TRIGGER IF EXISTS update_sec_ownership_events_updated_at
    ON sec_ownership_events;
CREATE TRIGGER update_sec_ownership_events_updated_at
    BEFORE UPDATE ON sec_ownership_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
