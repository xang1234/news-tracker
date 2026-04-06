-- Migration 021: Coverage profiles, tier history, and domain pack membership
--
-- Makes coverage maturity explicit by concept, theme, and domain pack.
-- History is append-only so point-in-time queries work without rewriting.
--
-- Tables:
--   coverage_profiles    — current coverage state per concept
--   coverage_tier_history — bitemporal tier change log
--   domain_packs         — named coverage domains (e.g., "Semiconductors Pack 1")
--   domain_pack_members  — concept membership in packs

-- ============================================================
-- 1. coverage_profiles — current coverage state
-- ============================================================

CREATE TABLE IF NOT EXISTS coverage_profiles (
    concept_id      TEXT PRIMARY KEY REFERENCES concepts(concept_id) ON DELETE CASCADE,
    coverage_tier   TEXT NOT NULL DEFAULT 'stub'
        CHECK (coverage_tier IN ('full', 'partial', 'stub', 'unsupported')),
    coverage_notes  TEXT NOT NULL DEFAULT '',
    structural_completeness REAL NOT NULL DEFAULT 0.0
        CHECK (structural_completeness >= 0 AND structural_completeness <= 1),
    filing_coverage     BOOLEAN NOT NULL DEFAULT FALSE,
    narrative_coverage  BOOLEAN NOT NULL DEFAULT FALSE,
    graph_coverage      BOOLEAN NOT NULL DEFAULT FALSE,
    last_assessed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_coverage_profiles_tier
    ON coverage_profiles (coverage_tier);

DROP TRIGGER IF EXISTS update_coverage_profiles_updated_at ON coverage_profiles;
CREATE TRIGGER update_coverage_profiles_updated_at
    BEFORE UPDATE ON coverage_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 2. coverage_tier_history — bitemporal tier change log
-- ============================================================
-- Append-only: new rows are added when tier changes, old rows
-- get valid_to set. Enables point-in-time coverage queries.

CREATE TABLE IF NOT EXISTS coverage_tier_history (
    id              SERIAL PRIMARY KEY,
    concept_id      TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    coverage_tier   TEXT NOT NULL
        CHECK (coverage_tier IN ('full', 'partial', 'stub', 'unsupported')),
    valid_from      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to        TIMESTAMPTZ,
    changed_by      TEXT,
    change_reason   TEXT NOT NULL DEFAULT '',
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_coverage_history_concept
    ON coverage_tier_history (concept_id, valid_from DESC);

CREATE INDEX IF NOT EXISTS idx_coverage_history_current
    ON coverage_tier_history (concept_id)
    WHERE valid_to IS NULL;

-- ============================================================
-- 3. domain_packs — named coverage domains
-- ============================================================

CREATE TABLE IF NOT EXISTS domain_packs (
    pack_id         TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    version         TEXT NOT NULL DEFAULT '1.0',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DROP TRIGGER IF EXISTS update_domain_packs_updated_at ON domain_packs;
CREATE TRIGGER update_domain_packs_updated_at
    BEFORE UPDATE ON domain_packs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 4. domain_pack_members — concept membership in packs
-- ============================================================

CREATE TABLE IF NOT EXISTS domain_pack_members (
    pack_id         TEXT NOT NULL REFERENCES domain_packs(pack_id) ON DELETE CASCADE,
    concept_id      TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    role            TEXT NOT NULL DEFAULT 'member'
        CHECK (role IN ('anchor', 'member', 'peripheral')),
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (pack_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_pack_members_concept
    ON domain_pack_members (concept_id);
