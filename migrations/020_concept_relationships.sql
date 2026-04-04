-- Migration 020: Concept relationship tables and theme/narrative links
--
-- Adds:
--   1. concept_relationships — typed directed edges between concepts
--      (subsidiary, supplier, customer, competitor, technology_user, etc.)
--   2. concept_theme_links — explicit links from theme/narrative_frame
--      concepts to the entities they cover
--
-- These tables model structural intelligence at the identity level.
-- They are distinct from causal_edges (which model graph-level analysis
-- with weighted edges for sentiment propagation).

-- ============================================================
-- 1. concept_relationships — structural edges between concepts
-- ============================================================

CREATE TABLE IF NOT EXISTS concept_relationships (
    source_concept_id   TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    target_concept_id   TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    relationship_type   TEXT NOT NULL
        CHECK (relationship_type IN (
            'subsidiary_of', 'parent_of',
            'supplies_to', 'customer_of',
            'competes_with',
            'produces', 'consumes',
            'operates_facility', 'located_at',
            'uses_technology', 'develops_technology',
            'component_of', 'contains_component'
        )),
    confidence          REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    source_attribution  TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}',
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_concept_id, target_concept_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_concept_rels_target
    ON concept_relationships (target_concept_id, relationship_type);

CREATE INDEX IF NOT EXISTS idx_concept_rels_type
    ON concept_relationships (relationship_type, is_active);

DROP TRIGGER IF EXISTS update_concept_relationships_updated_at ON concept_relationships;
CREATE TRIGGER update_concept_relationships_updated_at
    BEFORE UPDATE ON concept_relationships
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 2. concept_theme_links — theme/narrative coverage links
-- ============================================================
-- Explicitly connects theme and narrative_frame concepts to the
-- entities (issuers, technologies, products, etc.) they cover.
-- This keeps themes and narrative frames as distinct concept types
-- while making their coverage queryable.

CREATE TABLE IF NOT EXISTS concept_theme_links (
    theme_concept_id    TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    linked_concept_id   TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    link_type           TEXT NOT NULL DEFAULT 'covers'
        CHECK (link_type IN (
            'covers', 'driven_by', 'impacts', 'monitors'
        )),
    relevance_score     REAL NOT NULL DEFAULT 1.0 CHECK (relevance_score >= 0 AND relevance_score <= 1),
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (theme_concept_id, linked_concept_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_theme_links_linked
    ON concept_theme_links (linked_concept_id, link_type);

CREATE INDEX IF NOT EXISTS idx_theme_links_type
    ON concept_theme_links (link_type);
