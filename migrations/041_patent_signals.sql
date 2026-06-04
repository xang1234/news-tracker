-- Patent/application innovation evidence linked to issuers, securities, and themes.
-- Source records come from USPTO ODP search during the PatentsView transition
-- or PatentsView/ODP bulk snapshots with explicit staleness metadata.

CREATE TABLE IF NOT EXISTS innovation_patent_signals (
    patent_id            TEXT NOT NULL,
    patent_family_id     TEXT NOT NULL DEFAULT '',
    event_type           TEXT NOT NULL,
    event_date           DATE NOT NULL,
    title                TEXT NOT NULL DEFAULT '',
    issuer_concept_id    TEXT NOT NULL REFERENCES concepts(concept_id),
    security_concept_id  TEXT NOT NULL REFERENCES concepts(concept_id),
    theme_id             TEXT NOT NULL,
    confidence           DOUBLE PRECISION NOT NULL,
    confidence_reasons   JSONB NOT NULL DEFAULT '[]',
    source_lineage       JSONB NOT NULL DEFAULT '{}',
    metadata             JSONB NOT NULL DEFAULT '{}',
    source_url           TEXT NOT NULL DEFAULT '',
    fetched_at           TIMESTAMPTZ NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT innovation_patent_signals_event_type_check
        CHECK (event_type IN ('application', 'grant')),
    CONSTRAINT innovation_patent_signals_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    PRIMARY KEY (
        patent_id,
        event_type,
        issuer_concept_id,
        security_concept_id,
        theme_id
    )
);

CREATE INDEX IF NOT EXISTS idx_innovation_patent_signals_event_date
    ON innovation_patent_signals(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_innovation_patent_signals_theme_date
    ON innovation_patent_signals(theme_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_innovation_patent_signals_issuer_date
    ON innovation_patent_signals(issuer_concept_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_innovation_patent_signals_security_date
    ON innovation_patent_signals(security_concept_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_innovation_patent_signals_family
    ON innovation_patent_signals(patent_family_id);
