"""Bundle export service for the intelligence layer.

Exports a sealed manifest's published objects into an immutable JSONL
bundle with integrity checksums. The bundle format preserves contract
semantics so that offline consumers see the same data as direct
intel_pub.published_objects readers.

Bundle format (JSONL):
    Line 1: manifest header (JSON object with manifest metadata)
    Lines 2..N: published objects (one JSON object per line, sorted by object_id)

Checksum:
    SHA-256 over the raw JSONL bytes. Stored in the ExportBundle record
    and verifiable independently by any consumer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from src.contracts.intelligence.db_schemas import (
    ExportBundle,
    Manifest,
    PublishedObject,
)
from src.contracts.intelligence.version import ContractRegistry
from src.publish.repository import PublishRepository

logger = logging.getLogger(__name__)


def _serialize_object(obj: PublishedObject) -> dict[str, Any]:
    """Serialize a PublishedObject to a JSON-safe dict.

    Converts datetime fields to ISO strings for JSONL output.
    """
    return {
        "object_id": obj.object_id,
        "object_type": obj.object_type,
        "manifest_id": obj.manifest_id,
        "lane": obj.lane,
        "publish_state": obj.publish_state,
        "contract_version": obj.contract_version,
        "source_ids": obj.source_ids,
        "run_id": obj.run_id,
        "valid_from": obj.valid_from.isoformat() if obj.valid_from else None,
        "valid_to": obj.valid_to.isoformat() if obj.valid_to else None,
        "payload": obj.payload,
        "lineage": obj.lineage,
        "created_at": obj.created_at.isoformat(),
        "updated_at": obj.updated_at.isoformat(),
    }


def _serialize_manifest_header(manifest: Manifest) -> dict[str, Any]:
    """Serialize a Manifest to a JSON-safe header dict for JSONL output."""
    return {
        "_type": "manifest_header",
        "manifest_id": manifest.manifest_id,
        "lane": manifest.lane,
        "run_id": manifest.run_id,
        "contract_version": manifest.contract_version,
        "published_at": manifest.published_at.isoformat()
        if manifest.published_at
        else None,
        "object_count": manifest.object_count,
        "checksum": manifest.checksum,
        "metadata": manifest.metadata,
        "created_at": manifest.created_at.isoformat(),
    }


def build_bundle_lines(
    manifest: Manifest, objects: list[PublishedObject]
) -> list[str]:
    """Build JSONL lines for a bundle export.

    Args:
        manifest: The sealed manifest to export.
        objects: Published objects to include (should be sorted).

    Returns:
        List of JSON strings (one per line), header first.
    """
    lines = [json.dumps(_serialize_manifest_header(manifest), sort_keys=True)]
    for obj in sorted(objects, key=lambda o: o.object_id):
        lines.append(json.dumps(_serialize_object(obj), sort_keys=True))
    return lines


def compute_bundle_checksum(lines: list[str]) -> str:
    """Compute SHA-256 checksum over raw JSONL content.

    Args:
        lines: JSONL lines (as returned by build_bundle_lines).

    Returns:
        Checksum string in format 'sha256:<hex>'.
    """
    content = "\n".join(lines).encode("utf-8")
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def parse_bundle_lines(
    lines: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse JSONL bundle lines back into header + objects.

    This is the consumer-side deserialization function. Downstream
    consumers use this to read bundle files.

    Args:
        lines: Raw JSONL lines from a bundle file.

    Returns:
        Tuple of (manifest_header_dict, list_of_object_dicts).

    Raises:
        ValueError: If the bundle is empty or the header is missing.
    """
    if not lines:
        raise ValueError("Bundle is empty")
    header = json.loads(lines[0])
    if header.get("_type") != "manifest_header":
        raise ValueError(
            "First line of bundle must be a manifest header "
            f"(got _type={header.get('_type')!r})"
        )
    objects = [json.loads(line) for line in lines[1:]]
    return header, objects


def verify_bundle_checksum(lines: list[str], expected: str) -> bool:
    """Verify a bundle's integrity checksum.

    Args:
        lines: Raw JSONL lines from a bundle file.
        expected: Expected checksum string.

    Returns:
        True if the checksum matches, False otherwise.
    """
    return compute_bundle_checksum(lines) == expected


class BundleExporter:
    """Exports sealed manifests as immutable JSONL bundles.

    Usage:
        exporter = BundleExporter(repo)
        bundle, lines = await exporter.export_manifest(manifest_id)
        # Write lines to file, S3, etc.
    """

    def __init__(self, repository: PublishRepository) -> None:
        self._repo = repository

    async def export_manifest(
        self,
        manifest_id: str,
        *,
        exported_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ExportBundle, list[str]]:
        """Export a sealed manifest as a JSONL bundle.

        Fetches the manifest and its published objects, serializes
        them to JSONL, computes a checksum, and records the export.

        Args:
            manifest_id: Manifest to export (must be sealed).
            exported_by: Who/what triggered the export.
            metadata: Extensible metadata for the export record.

        Returns:
            Tuple of (ExportBundle record, JSONL lines).

        Raises:
            ValueError: If the manifest doesn't exist or isn't sealed.
        """
        manifest = await self._repo.get_manifest(manifest_id)
        if manifest is None:
            raise ValueError(f"Manifest not found: {manifest_id}")
        if manifest.published_at is None:
            raise ValueError(
                f"Manifest {manifest_id} is not sealed. "
                "Call seal_manifest() before exporting."
            )

        # Fetch only published objects
        objects = await self._repo.list_objects_by_manifest(
            manifest_id, publish_state="published"
        )

        # Build JSONL lines and compute checksum
        lines = build_bundle_lines(manifest, objects)
        checksum = compute_bundle_checksum(lines)
        content_bytes = "\n".join(lines).encode("utf-8")

        # Record the export
        bundle = ExportBundle(
            bundle_id=f"bundle_{hashlib.sha256(manifest_id.encode()).hexdigest()[:12]}",
            manifest_id=manifest_id,
            lane=manifest.lane,
            contract_version=str(ContractRegistry.CURRENT),
            format="jsonl",
            object_count=len(objects),
            size_bytes=len(content_bytes),
            checksum=checksum,
            exported_by=exported_by,
            metadata=metadata or {},
        )
        saved = await self._repo.create_export_bundle(bundle)

        logger.info(
            "Bundle exported: %s (manifest=%s, objects=%d, bytes=%d)",
            saved.bundle_id,
            manifest_id,
            len(objects),
            len(content_bytes),
        )
        return saved, lines
