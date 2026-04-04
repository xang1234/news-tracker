-- Migration 022: Filing artifact persistence tables
--
-- Stores raw filings, normalized sections, attachments, and XBRL facts
-- with full lineage and bitemporal timestamps for replay.
--
-- Tables:
--   filings            — one row per SEC filing (keyed by accession number)
--   filing_sections    — parsed sections within a filing
--   filing_attachments — exhibits and other filing attachments
--   filing_xbrl_facts  — structured XBRL data points

-- ============================================================
-- 1. filings — SEC filing metadata and content hash
-- ============================================================

CREATE TABLE IF NOT EXISTS filings (
    accession_number    TEXT PRIMARY KEY,
    cik                 TEXT NOT NULL,
    filing_type         TEXT NOT NULL,
    filed_date          DATE NOT NULL,
    period_of_report    DATE,
    company_name        TEXT NOT NULL DEFAULT '',
    ticker              TEXT,
    concept_id          TEXT REFERENCES concepts(concept_id) ON DELETE SET NULL,
    -- Content
    raw_url             TEXT NOT NULL DEFAULT '',
    content_hash        TEXT,
    total_word_count    INTEGER NOT NULL DEFAULT 0,
    section_count       INTEGER NOT NULL DEFAULT 0,
    -- Provider and lineage
    provider            TEXT NOT NULL DEFAULT '',
    run_id              TEXT,
    status              TEXT NOT NULL DEFAULT 'fetched'
        CHECK (status IN ('pending', 'fetched', 'parsed', 'failed', 'skipped')),
    error_message       TEXT,
    -- Bitemporal timestamps
    source_published_at TIMESTAMPTZ,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Extensible
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_filings_cik
    ON filings (cik);

CREATE INDEX IF NOT EXISTS idx_filings_ticker
    ON filings (ticker) WHERE ticker IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_filings_type_date
    ON filings (filing_type, filed_date DESC);

CREATE INDEX IF NOT EXISTS idx_filings_concept
    ON filings (concept_id) WHERE concept_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_filings_status
    ON filings (status, ingested_at DESC);

DROP TRIGGER IF EXISTS update_filings_updated_at ON filings;
CREATE TRIGGER update_filings_updated_at
    BEFORE UPDATE ON filings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 2. filing_sections — parsed sections within a filing
-- ============================================================

CREATE TABLE IF NOT EXISTS filing_sections (
    section_id          TEXT PRIMARY KEY,
    accession_number    TEXT NOT NULL REFERENCES filings(accession_number) ON DELETE CASCADE,
    section_index       INTEGER NOT NULL,
    section_name        TEXT NOT NULL,
    section_type        TEXT NOT NULL DEFAULT 'narrative',
    content             TEXT NOT NULL DEFAULT '',
    word_count          INTEGER NOT NULL DEFAULT 0,
    content_hash        TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_filing_sections_accession
    ON filing_sections (accession_number, section_index);

-- ============================================================
-- 3. filing_attachments — exhibits and other attachments
-- ============================================================

CREATE TABLE IF NOT EXISTS filing_attachments (
    attachment_id       TEXT PRIMARY KEY,
    accession_number    TEXT NOT NULL REFERENCES filings(accession_number) ON DELETE CASCADE,
    filename            TEXT NOT NULL,
    content_type        TEXT NOT NULL DEFAULT '',
    description         TEXT NOT NULL DEFAULT '',
    url                 TEXT NOT NULL DEFAULT '',
    size_bytes          BIGINT,
    content_hash        TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_filing_attachments_accession
    ON filing_attachments (accession_number);

-- ============================================================
-- 4. filing_xbrl_facts — structured XBRL data points
-- ============================================================

CREATE TABLE IF NOT EXISTS filing_xbrl_facts (
    id                  SERIAL PRIMARY KEY,
    accession_number    TEXT NOT NULL REFERENCES filings(accession_number) ON DELETE CASCADE,
    taxonomy            TEXT NOT NULL DEFAULT 'us-gaap',
    concept_name        TEXT NOT NULL,
    value               TEXT NOT NULL,
    unit                TEXT,
    period_start        DATE,
    period_end          DATE,
    instant_date        DATE,
    decimals            INTEGER,
    segment             TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_xbrl_facts_accession
    ON filing_xbrl_facts (accession_number);

CREATE INDEX IF NOT EXISTS idx_xbrl_facts_concept
    ON filing_xbrl_facts (concept_name, accession_number);
