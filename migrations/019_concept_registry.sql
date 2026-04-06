-- Migration 019: Concept registry, alias model, and issuer/security crosswalk
--
-- Extends the security master into a broader identity layer.
-- Every entity in the intelligence layer (issuer, security, technology,
-- theme, etc.) gets a canonical concept record with aliases.
--
-- Coexistence: the existing securities table is untouched except for
-- an optional concept_id column linking securities to concepts.

-- ============================================================
-- 1. concepts — canonical concept registry
-- ============================================================

CREATE TABLE IF NOT EXISTS concepts (
    concept_id      TEXT PRIMARY KEY,
    concept_type    TEXT NOT NULL
        CHECK (concept_type IN (
            'issuer', 'security', 'technology', 'product',
            'theme', 'narrative_frame', 'facility', 'index'
        )),
    canonical_name  TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    metadata        JSONB NOT NULL DEFAULT '{}',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_concepts_type
    ON concepts (concept_type, is_active);

CREATE INDEX IF NOT EXISTS idx_concepts_name_trgm
    ON concepts USING GIN (canonical_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_concepts_active
    ON concepts (is_active) WHERE is_active = TRUE;

DROP TRIGGER IF EXISTS update_concepts_updated_at ON concepts;
CREATE TRIGGER update_concepts_updated_at
    BEFORE UPDATE ON concepts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 2. concept_aliases — multiple names resolving to one concept
-- ============================================================

CREATE TABLE IF NOT EXISTS concept_aliases (
    alias           TEXT NOT NULL,
    concept_id      TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    alias_type      TEXT NOT NULL DEFAULT 'name'
        CHECK (alias_type IN ('name', 'ticker', 'abbreviation', 'former_name', 'local_name')),
    is_primary      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (alias, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_concept_aliases_concept
    ON concept_aliases (concept_id);

CREATE INDEX IF NOT EXISTS idx_concept_aliases_alias_lower
    ON concept_aliases (lower(alias));

-- ============================================================
-- 3. issuer_security_map — crosswalk between issuers and securities
-- ============================================================
-- An issuer (company) concept maps to one or more security concepts.
-- This bridges the existing securities table to the concept layer.

CREATE TABLE IF NOT EXISTS issuer_security_map (
    issuer_concept_id   TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    security_concept_id TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    relationship_type   TEXT NOT NULL DEFAULT 'primary'
        CHECK (relationship_type IN ('primary', 'adr', 'subsidiary_listing', 'preferred', 'warrant')),
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (issuer_concept_id, security_concept_id)
);

CREATE INDEX IF NOT EXISTS idx_issuer_security_map_security
    ON issuer_security_map (security_concept_id);

-- ============================================================
-- 4. Link securities to concepts (optional, non-breaking)
-- ============================================================
-- Adds concept_id to the existing securities table so ticker-centric
-- flows can resolve to canonical concepts when available.

ALTER TABLE securities ADD COLUMN IF NOT EXISTS concept_id TEXT
    REFERENCES concepts(concept_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_securities_concept_id
    ON securities (concept_id) WHERE concept_id IS NOT NULL;
