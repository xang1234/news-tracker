-- Research/preprint innovation evidence linked to issuers, securities, and themes.
-- Source records come from OpenAlex Works metadata and arXiv Atom search feeds.

CREATE TABLE IF NOT EXISTS innovation_research_signals (
    source               TEXT NOT NULL,
    record_id            TEXT NOT NULL,
    published_date       DATE NOT NULL,
    title                TEXT NOT NULL DEFAULT '',
    issuer_concept_id    TEXT NOT NULL REFERENCES concepts(concept_id),
    security_concept_id  TEXT NOT NULL REFERENCES concepts(concept_id),
    theme_id             TEXT NOT NULL,
    confidence           DOUBLE PRECISION NOT NULL,
    confidence_reasons   JSONB NOT NULL DEFAULT '[]',
    source_lineage       JSONB NOT NULL DEFAULT '{}',
    metadata             JSONB NOT NULL DEFAULT '{}',
    url                  TEXT NOT NULL DEFAULT '',
    fetched_at           TIMESTAMPTZ NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT innovation_research_signals_source_check
        CHECK (source IN ('openalex', 'arxiv')),
    CONSTRAINT innovation_research_signals_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    PRIMARY KEY (
        source,
        record_id,
        issuer_concept_id,
        security_concept_id,
        theme_id
    )
);

CREATE INDEX IF NOT EXISTS idx_innovation_research_signals_published_date
    ON innovation_research_signals(published_date DESC);
CREATE INDEX IF NOT EXISTS idx_innovation_research_signals_theme_date
    ON innovation_research_signals(theme_id, published_date DESC);
CREATE INDEX IF NOT EXISTS idx_innovation_research_signals_issuer_date
    ON innovation_research_signals(issuer_concept_id, published_date DESC);
CREATE INDEX IF NOT EXISTS idx_innovation_research_signals_security_date
    ON innovation_research_signals(security_concept_id, published_date DESC);
CREATE INDEX IF NOT EXISTS idx_innovation_research_signals_source_date
    ON innovation_research_signals(source, published_date DESC);
