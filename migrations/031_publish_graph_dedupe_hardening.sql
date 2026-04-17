-- Migration 031: Harden consumer publish reads, graph evidence storage,
-- and shared duplicate suppression.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- Table: causal_edge_supports
-- One row per supporting evidence source for an aggregated edge.
-- ============================================================
CREATE TABLE IF NOT EXISTS causal_edge_supports (
    source          TEXT NOT NULL REFERENCES causal_nodes(node_id) ON DELETE CASCADE,
    target          TEXT NOT NULL REFERENCES causal_nodes(node_id) ON DELETE CASCADE,
    relation        TEXT NOT NULL
        CHECK (relation IN ('depends_on', 'supplies_to', 'competes_with', 'drives', 'blocks')),
    support_key     TEXT NOT NULL,
    origin_kind     TEXT NOT NULL
        CHECK (origin_kind IN ('assertion', 'legacy')),
    confidence      REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    source_doc_ids  TEXT[] NOT NULL DEFAULT '{}',
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (source, target, relation, support_key)
);

CREATE INDEX IF NOT EXISTS idx_causal_edge_supports_lookup
    ON causal_edge_supports (source, target, relation, active);

CREATE INDEX IF NOT EXISTS idx_causal_edge_supports_origin
    ON causal_edge_supports (origin_kind, active);

DROP TRIGGER IF EXISTS update_causal_edge_supports_updated_at ON causal_edge_supports;
CREATE TRIGGER update_causal_edge_supports_updated_at
    BEFORE UPDATE ON causal_edge_supports
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

INSERT INTO causal_edge_supports (
    source, target, relation, support_key, origin_kind,
    confidence, source_doc_ids, active, metadata
)
SELECT
    ce.source,
    ce.target,
    ce.relation,
    format('legacy::%s::%s::%s', ce.source, ce.target, ce.relation) AS support_key,
    'legacy' AS origin_kind,
    ce.confidence,
    ce.source_doc_ids,
    TRUE AS active,
    COALESCE(ce.metadata, '{}'::jsonb) || jsonb_build_object('backfilled_from', 'causal_edges')
FROM causal_edges ce
ON CONFLICT (source, target, relation, support_key) DO NOTHING;

-- ============================================================
-- Table: document_dedup_signatures
-- Canonical exact + near-duplicate state persisted for shared workers.
-- ============================================================
CREATE TABLE IF NOT EXISTS document_dedup_signatures (
    document_id              TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    canonical_document_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    exact_fingerprint        TEXT NOT NULL UNIQUE,
    minhash_signature        JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_dedup_signatures_canonical
    ON document_dedup_signatures (canonical_document_id);

DROP TRIGGER IF EXISTS update_document_dedup_signatures_updated_at
    ON document_dedup_signatures;
CREATE TRIGGER update_document_dedup_signatures_updated_at
    BEFORE UPDATE ON document_dedup_signatures
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Backfill the published read model for all manifests currently pointed
-- to by active manifest pointers before consumer routes switch over.
-- ============================================================
INSERT INTO intel_pub.read_model (
    record_id,
    manifest_id,
    object_id,
    object_type,
    lane,
    contract_version,
    publish_state,
    source_ids,
    run_id,
    valid_from,
    valid_to,
    payload,
    lineage,
    published_at,
    metadata,
    created_at
)
SELECT
    'rm_' || substr(
        encode(digest(po.manifest_id || chr(0) || po.object_id, 'sha256'), 'hex'),
        1,
        16
    ) AS record_id,
    po.manifest_id,
    po.object_id,
    po.object_type,
    po.lane,
    po.contract_version,
    po.publish_state,
    po.source_ids,
    po.run_id,
    po.valid_from,
    po.valid_to,
    COALESCE(po.payload, '{}'::jsonb),
    COALESCE(po.lineage, '{}'::jsonb),
    m.published_at,
    jsonb_build_object(
        'manifest_checksum', m.checksum,
        'manifest_object_count', m.object_count,
        'backfilled_from', 'published_objects'
    ),
    po.created_at
FROM intel_pub.published_objects po
JOIN intel_pub.manifests m
  ON m.manifest_id = po.manifest_id
JOIN intel_pub.manifest_pointers mp
  ON mp.lane = po.lane
 AND mp.manifest_id = po.manifest_id
WHERE po.publish_state = 'published'
ON CONFLICT (manifest_id, object_id) DO NOTHING;
