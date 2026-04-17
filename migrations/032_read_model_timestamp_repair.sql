-- Migration 032: Repair read-model timestamps and add updated_at.

ALTER TABLE intel_pub.read_model
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

UPDATE intel_pub.read_model rm
SET created_at = po.created_at,
    updated_at = po.updated_at
FROM intel_pub.published_objects po
WHERE po.manifest_id = rm.manifest_id
  AND po.object_id = rm.object_id
  AND (
        rm.created_at IS DISTINCT FROM po.created_at
        OR rm.updated_at IS DISTINCT FROM po.updated_at
      );

UPDATE intel_pub.read_model
SET updated_at = created_at
WHERE updated_at IS NULL;
