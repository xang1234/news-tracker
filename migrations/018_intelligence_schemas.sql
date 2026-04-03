-- Migration 018: Intelligence layer schema namespaces and lifecycle tables
--
-- Creates three PostgreSQL schemas for the intelligence layer:
--   news_intel  — work-in-progress tables (claims, runs, intermediate state)
--   intel_pub   — published/resolved rows (assertions, manifests, pointers)
--   intel_export — bundle export artifacts (snapshots for offline consumers)
--
-- Also creates shared lifecycle primitives:
--   news_intel.lane_runs    — identity and lifecycle for lane executions
--   intel_pub.manifests     — versioned manifest headers (published bundles)
--   intel_pub.manifest_pointers — current serving pointer per lane
--   intel_pub.published_objects — individual published items within manifests
--
-- Coexistence: All new tables live in dedicated schemas.
-- Existing public.* tables (documents, themes, narrative_runs, etc.) are untouched.

-- ============================================================
-- 1. Create schema namespaces
-- ============================================================

CREATE SCHEMA IF NOT EXISTS news_intel;
CREATE SCHEMA IF NOT EXISTS intel_pub;
CREATE SCHEMA IF NOT EXISTS intel_export;

-- ============================================================
-- 2. news_intel.lane_runs — lane execution lifecycle
-- ============================================================
-- Every lane execution (narrative batch, filing parse, structural refresh,
-- backtest evaluation) gets a lane_run record. This is the anchor for
-- all work-in-progress artifacts produced during that run.

CREATE TABLE IF NOT EXISTS news_intel.lane_runs (
    run_id          TEXT PRIMARY KEY,
    lane            TEXT NOT NULL
        CHECK (lane IN ('narrative', 'filing', 'structural', 'backtest')),
    status          TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    contract_version TEXT NOT NULL,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    config_snapshot JSONB NOT NULL DEFAULT '{}',
    metrics         JSONB NOT NULL DEFAULT '{}',
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lane_runs_lane_status
    ON news_intel.lane_runs (lane, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_lane_runs_status
    ON news_intel.lane_runs (status, updated_at DESC);

DROP TRIGGER IF EXISTS update_lane_runs_updated_at ON news_intel.lane_runs;
CREATE TRIGGER update_lane_runs_updated_at
    BEFORE UPDATE ON news_intel.lane_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 3. intel_pub.manifests — versioned published manifests
-- ============================================================
-- A manifest groups published objects from a single lane-run into
-- an addressable, versioned unit. Downstream consumers read manifests
-- via the manifest_pointers table, never news_intel tables directly.

CREATE TABLE IF NOT EXISTS intel_pub.manifests (
    manifest_id     TEXT PRIMARY KEY,
    lane            TEXT NOT NULL
        CHECK (lane IN ('narrative', 'filing', 'structural', 'backtest')),
    run_id          TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    object_count    INTEGER NOT NULL DEFAULT 0 CHECK (object_count >= 0),
    checksum        TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_manifests_lane_published
    ON intel_pub.manifests (lane, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_manifests_run_id
    ON intel_pub.manifests (run_id);

-- ============================================================
-- 4. intel_pub.manifest_pointers — current serving pointer per lane
-- ============================================================
-- Each lane has exactly one active manifest pointer. Advancing the
-- pointer is an atomic operation that switches downstream consumers
-- to a new manifest without downtime.

CREATE TABLE IF NOT EXISTS intel_pub.manifest_pointers (
    lane            TEXT PRIMARY KEY
        CHECK (lane IN ('narrative', 'filing', 'structural', 'backtest')),
    manifest_id     TEXT NOT NULL REFERENCES intel_pub.manifests(manifest_id),
    activated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    previous_manifest_id TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'
);

-- ============================================================
-- 5. intel_pub.published_objects — individual items in manifests
-- ============================================================
-- Each published item (claim, assertion, signal, etc.) gets a row.
-- The manifest_id groups them; object_type + lane identify the kind.

CREATE TABLE IF NOT EXISTS intel_pub.published_objects (
    object_id       TEXT PRIMARY KEY,
    object_type     TEXT NOT NULL,
    manifest_id     TEXT NOT NULL REFERENCES intel_pub.manifests(manifest_id),
    lane            TEXT NOT NULL
        CHECK (lane IN ('narrative', 'filing', 'structural', 'backtest')),
    publish_state   TEXT NOT NULL DEFAULT 'draft'
        CHECK (publish_state IN ('draft', 'review', 'published', 'retracted')),
    contract_version TEXT NOT NULL,
    -- Lineage
    source_ids      TEXT[] NOT NULL DEFAULT '{}',
    run_id          TEXT NOT NULL,
    -- Bitemporal fields
    valid_from      TIMESTAMPTZ,
    valid_to        TIMESTAMPTZ,
    -- Payload
    payload         JSONB NOT NULL DEFAULT '{}',
    lineage         JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_published_objects_manifest
    ON intel_pub.published_objects (manifest_id, object_type);

CREATE INDEX IF NOT EXISTS idx_published_objects_lane_state
    ON intel_pub.published_objects (lane, publish_state, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_published_objects_type_state
    ON intel_pub.published_objects (object_type, publish_state);

DROP TRIGGER IF EXISTS update_published_objects_updated_at ON intel_pub.published_objects;
CREATE TRIGGER update_published_objects_updated_at
    BEFORE UPDATE ON intel_pub.published_objects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 6. intel_export.export_bundles — bundle export tracking
-- ============================================================
-- Tracks exported bundle artifacts for offline consumers.
-- The actual bundle content lives in the filesystem or object store;
-- this table records the export metadata and checksums.

CREATE TABLE IF NOT EXISTS intel_export.export_bundles (
    bundle_id       TEXT PRIMARY KEY,
    manifest_id     TEXT NOT NULL,
    lane            TEXT NOT NULL
        CHECK (lane IN ('narrative', 'filing', 'structural', 'backtest')),
    contract_version TEXT NOT NULL,
    format          TEXT NOT NULL DEFAULT 'jsonl'
        CHECK (format IN ('jsonl', 'parquet', 'csv')),
    object_count    INTEGER NOT NULL DEFAULT 0 CHECK (object_count >= 0),
    size_bytes      BIGINT,
    checksum        TEXT,
    exported_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    exported_by     TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_export_bundles_manifest
    ON intel_export.export_bundles (manifest_id);

CREATE INDEX IF NOT EXISTS idx_export_bundles_lane_exported
    ON intel_export.export_bundles (lane, exported_at DESC);
