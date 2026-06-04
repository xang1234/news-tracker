-- Migration 038: Market-structure events
--
-- Stores point-in-time FINRA daily short-volume rows and SEC
-- fails-to-deliver rows with explicit ratio/notional/persistence signals.
-- These records are trading/settlement activity signals and are not
-- short-interest position records.

CREATE TABLE IF NOT EXISTS market_structure_events (
    event_id             TEXT PRIMARY KEY,
    event_type           TEXT NOT NULL
        CHECK (event_type IN ('finra_short_volume', 'sec_fail_to_deliver')),
    source_name          TEXT NOT NULL,
    source_url           TEXT NOT NULL DEFAULT '',
    source_date          DATE NOT NULL,
    trade_date           DATE,
    settlement_date      DATE,
    symbol               TEXT NOT NULL DEFAULT '',
    security_ticker      TEXT NOT NULL DEFAULT '',
    security_exchange    TEXT NOT NULL DEFAULT 'US',
    issuer_cik           TEXT NOT NULL DEFAULT '',
    issuer_name          TEXT NOT NULL DEFAULT '',
    cusip                TEXT NOT NULL DEFAULT '',
    market_code          TEXT NOT NULL DEFAULT '',
    market_name          TEXT NOT NULL DEFAULT '',
    short_volume         BIGINT,
    short_exempt_volume  BIGINT,
    total_volume         BIGINT,
    short_volume_ratio   NUMERIC,
    short_exempt_ratio   NUMERIC,
    fail_quantity        BIGINT,
    fail_price           NUMERIC,
    fail_notional        NUMERIC,
    signal_type          TEXT
        CHECK (
            signal_type IS NULL
            OR signal_type IN ('short_volume_ratio', 'fails_to_deliver_notional')
        ),
    anomaly_level        TEXT NOT NULL DEFAULT 'none'
        CHECK (anomaly_level IN ('none', 'watch', 'elevated', 'extreme')),
    persistence_count    INTEGER NOT NULL DEFAULT 0,
    available_at         TIMESTAMPTZ NOT NULL,
    fetched_at           TIMESTAMPTZ,
    metadata             JSONB NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_structure_events_symbol_as_of
    ON market_structure_events (symbol, available_at DESC, source_date DESC);

CREATE INDEX IF NOT EXISTS idx_market_structure_events_cusip_as_of
    ON market_structure_events (cusip, available_at DESC, source_date DESC)
    WHERE cusip <> '';

CREATE INDEX IF NOT EXISTS idx_market_structure_events_event_type_as_of
    ON market_structure_events (event_type, available_at DESC, source_date DESC);

DROP TRIGGER IF EXISTS update_market_structure_events_updated_at
    ON market_structure_events;
CREATE TRIGGER update_market_structure_events_updated_at
    BEFORE UPDATE ON market_structure_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
