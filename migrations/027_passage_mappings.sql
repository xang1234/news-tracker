-- Migration 027: Passage-to-concept mappings for narrative frames and themes
--
-- Maps source passages to two distinct concept layers:
--   narrative_frame: the specific angle expressed (e.g., "TSMC HBM Bottleneck")
--   theme_concept: the broader thematic category (e.g., "High Bandwidth Memory")
--
-- This separates the single overloaded theme_id used by narrative_runs
-- into explicit concept mappings through the canonical identity layer.

CREATE TABLE IF NOT EXISTS news_intel.passage_mappings (
    mapping_id          TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL,
    -- Concept identities
    narrative_frame_id  TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    theme_concept_id    TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    narrative_frame_name TEXT NOT NULL DEFAULT '',
    theme_concept_name  TEXT NOT NULL DEFAULT '',
    -- Source span (optional, for passage-level tracing)
    source_span_start   INTEGER,
    source_span_end     INTEGER,
    passage_text        TEXT,
    -- Lineage
    narrative_run_id    TEXT,
    confidence          REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0 AND confidence <= 1),
    -- Extensible
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Find mappings by source document
CREATE INDEX IF NOT EXISTS idx_passage_map_source
    ON news_intel.passage_mappings (source_id);

-- Find mappings by narrative frame concept
CREATE INDEX IF NOT EXISTS idx_passage_map_frame
    ON news_intel.passage_mappings (narrative_frame_id);

-- Find mappings by theme concept
CREATE INDEX IF NOT EXISTS idx_passage_map_theme
    ON news_intel.passage_mappings (theme_concept_id);

-- Find mappings by narrative run
CREATE INDEX IF NOT EXISTS idx_passage_map_run
    ON news_intel.passage_mappings (narrative_run_id)
    WHERE narrative_run_id IS NOT NULL;
