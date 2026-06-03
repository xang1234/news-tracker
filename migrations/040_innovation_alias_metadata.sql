-- Migration 040: Innovation alias confidence and review metadata
--
-- Patent assignees and research affiliations often use subsidiaries, lab
-- names, acquired entities, and ambiguous abbreviations. Keep those aliases in
-- the canonical concept registry, but carry confidence and review metadata so
-- downstream datasource joins can retain ambiguity instead of overclaiming.

ALTER TABLE concept_aliases
    ADD COLUMN IF NOT EXISTS confidence REAL NOT NULL DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS source_attribution TEXT,
    ADD COLUMN IF NOT EXISTS review_status TEXT NOT NULL DEFAULT 'accepted',
    ADD COLUMN IF NOT EXISTS review_note TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';

ALTER TABLE concept_aliases
    DROP CONSTRAINT IF EXISTS concept_aliases_alias_type_check;

ALTER TABLE concept_aliases
    ADD CONSTRAINT concept_aliases_alias_type_check
    CHECK (alias_type IN (
        'name', 'ticker', 'abbreviation', 'former_name', 'local_name',
        'subsidiary', 'acquired_entity', 'lab', 'research_institution'
    ));

ALTER TABLE concept_aliases
    DROP CONSTRAINT IF EXISTS concept_aliases_confidence_check;

ALTER TABLE concept_aliases
    ADD CONSTRAINT concept_aliases_confidence_check
    CHECK (confidence >= 0.0 AND confidence <= 1.0);

ALTER TABLE concept_aliases
    DROP CONSTRAINT IF EXISTS concept_aliases_review_status_check;

ALTER TABLE concept_aliases
    ADD CONSTRAINT concept_aliases_review_status_check
    CHECK (review_status IN ('accepted', 'needs_review', 'rejected'));

CREATE INDEX IF NOT EXISTS idx_concept_aliases_review_status
    ON concept_aliases (review_status);

CREATE INDEX IF NOT EXISTS idx_concept_aliases_alias_confidence
    ON concept_aliases (lower(alias), confidence DESC);
