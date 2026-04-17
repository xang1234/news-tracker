-- Migration 028: Read-model table for published intelligence outputs
--
-- Materializes published objects into a stable consumer-facing read
-- surface. Downstream consumers query this table instead of joining
-- across news_intel/intel_pub working tables.
--
-- Records are immutable once materialized. New manifest publications
-- add records; old manifests remain queryable for historical analysis.

CREATE TABLE IF NOT EXISTS intel_pub.read_model (
    record_id           TEXT PRIMARY KEY,
    manifest_id         TEXT NOT NULL,
    object_id           TEXT NOT NULL,
    object_type         TEXT NOT NULL,
    lane                TEXT NOT NULL
        CHECK (lane IN ('narrative', 'filing', 'structural', 'backtest')),
    contract_version    TEXT NOT NULL,
    publish_state       TEXT NOT NULL DEFAULT 'published',
    -- Lineage
    source_ids          TEXT[] NOT NULL DEFAULT '{}',
    run_id              TEXT NOT NULL DEFAULT '',
    -- Bitemporal
    valid_from          TIMESTAMPTZ,
    valid_to            TIMESTAMPTZ,
    -- Content
    payload             JSONB NOT NULL DEFAULT '{}',
    lineage             JSONB NOT NULL DEFAULT '{}',
    published_at        TIMESTAMPTZ,
    -- Extensible
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Consumer queries by manifest (get all objects in a publication)
CREATE INDEX IF NOT EXISTS idx_read_model_manifest
    ON intel_pub.read_model (manifest_id, object_type);

-- Consumer queries by lane (latest objects for a lane)
CREATE INDEX IF NOT EXISTS idx_read_model_lane
    ON intel_pub.read_model (lane, object_type, published_at DESC NULLS LAST);

-- Consumer queries by object type (e.g., all assertions)
CREATE INDEX IF NOT EXISTS idx_read_model_type
    ON intel_pub.read_model (object_type, published_at DESC NULLS LAST);

-- Dedup: one record per (manifest, object) pair
CREATE UNIQUE INDEX IF NOT EXISTS idx_read_model_manifest_object
    ON intel_pub.read_model (manifest_id, object_id);
