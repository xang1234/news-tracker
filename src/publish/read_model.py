"""Read-model builder for published intelligence outputs.

Materializes published objects into stable read-model records that
downstream consumers query. Read models are explicit products of the
producer, not aliases over mutable working tables.

Consumer contract:
    - Read models are keyed by (manifest_id, object_id)
    - Each record is immutable once materialized
    - Consumers read by lane, object_type, or manifest_id
    - The read surface is stable across manifest pointer advances

Builder flow:
    1. Receive a sealed manifest with published objects
    2. For each published object, build a ReadModelRecord
    3. Return records for the caller to persist
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.contracts.intelligence.db_schemas import Manifest, PublishedObject
from src.contracts.intelligence.ownership import OwnershipPolicy

# -- ReadModelRecord -------------------------------------------------------


@dataclass(frozen=True)
class ReadModelRecord:
    """A materialized read-model entry for a published object.

    Denormalized for downstream consumer reads. Carries enough
    context from the manifest and object for consumers to query
    without joining back to producer tables.

    Attributes:
        record_id: Deterministic ID (manifest_id + object_id).
        manifest_id: Which manifest this belongs to.
        object_id: The published object.
        object_type: Kind of object (claim, assertion, signal, etc.).
        lane: Which lane produced this object.
        contract_version: Contract version at publication time.
        publish_state: Object lifecycle state.
        source_ids: Lineage source IDs.
        run_id: Lane run that produced this.
        valid_from: Bitemporal start.
        valid_to: Bitemporal end.
        payload: Object-specific content.
        lineage: Full lineage metadata.
        published_at: When the manifest was sealed/published.
        metadata: Extensible metadata.
    """

    record_id: str
    manifest_id: str
    object_id: str
    object_type: str
    lane: str
    contract_version: str
    publish_state: str = "published"
    source_ids: list[str] = field(default_factory=list)
    run_id: str = ""
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    published_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


def make_record_id(manifest_id: str, object_id: str) -> str:
    """Generate a deterministic read-model record ID.

    Same manifest + object always produces the same record,
    enabling idempotent materialization.
    """
    key = f"{manifest_id}\x00{object_id}"
    return f"rm_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


# -- Builder ---------------------------------------------------------------


class ReadModelBuilder:
    """Builds read-model records from published manifest objects.

    Stateless builder — takes manifest + objects, returns records.
    The caller handles persistence.

    Usage:
        builder = ReadModelBuilder()
        records = builder.build(manifest, published_objects)
    """

    def build_record(
        self,
        manifest: Manifest,
        obj: PublishedObject,
    ) -> ReadModelRecord:
        """Build a single read-model record from a published object.

        Validates that the object type is publishable, then
        denormalizes manifest and object data into a flat record.

        Raises:
            ValueError: If the object type is not publishable.
        """
        OwnershipPolicy.validate_publishable_type(obj.object_type)
        return self._build_record_unchecked(manifest, obj)

    def _build_record_unchecked(
        self,
        manifest: Manifest,
        obj: PublishedObject,
    ) -> ReadModelRecord:
        """Build a record without type validation (caller pre-validated)."""
        return ReadModelRecord(
            record_id=make_record_id(manifest.manifest_id, obj.object_id),
            manifest_id=manifest.manifest_id,
            object_id=obj.object_id,
            object_type=obj.object_type,
            lane=obj.lane,
            contract_version=obj.contract_version,
            publish_state=obj.publish_state,
            source_ids=list(obj.source_ids),
            run_id=obj.run_id,
            valid_from=obj.valid_from,
            valid_to=obj.valid_to,
            payload=dict(obj.payload),
            lineage=dict(obj.lineage),
            published_at=manifest.published_at,
            metadata={
                "manifest_checksum": manifest.checksum,
                "manifest_object_count": manifest.object_count,
            },
            created_at=obj.created_at,
            updated_at=obj.updated_at,
        )

    def build(
        self,
        manifest: Manifest,
        objects: list[PublishedObject],
        *,
        published_only: bool = True,
    ) -> list[ReadModelRecord]:
        """Build read-model records for all objects in a manifest.

        Args:
            manifest: The sealed manifest.
            objects: Published objects within the manifest.
            published_only: If True (default), skip non-published objects.

        Returns:
            List of ReadModelRecord for materialization.
        """
        # Validate unique object types once upfront
        unique_types = {obj.object_type for obj in objects}
        for ot in unique_types:
            OwnershipPolicy.validate_publishable_type(ot)

        records: list[ReadModelRecord] = []
        for obj in objects:
            if published_only and obj.publish_state != "published":
                continue
            records.append(self._build_record_unchecked(manifest, obj))
        return records

    def build_summary(
        self,
        manifest: Manifest,
        records: list[ReadModelRecord],
    ) -> dict[str, Any]:
        """Build a summary of what was materialized.

        Useful for audit logging and operational dashboards.
        """
        type_counts: dict[str, int] = {}
        for r in records:
            type_counts[r.object_type] = type_counts.get(r.object_type, 0) + 1

        return {
            "manifest_id": manifest.manifest_id,
            "lane": manifest.lane,
            "contract_version": manifest.contract_version,
            "total_records": len(records),
            "by_object_type": type_counts,
            "published_at": (manifest.published_at.isoformat() if manifest.published_at else None),
        }
