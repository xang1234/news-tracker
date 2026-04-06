"""Compatibility fixtures for contract tests.

Provides factory functions that produce valid, minimal instances of all
intelligence contract types. Downstream consumers (e.g., stock-screener)
import these fixtures to verify their deserialization code handles the
canonical shapes correctly.

Usage in consumer tests:
    from src.contracts.intelligence.fixtures import (
        make_lineage,
        make_manifest_header,
        make_published_object_ref,
        make_lane_run,
        make_manifest,
        make_manifest_pointer,
        make_published_object,
        make_export_bundle,
    )
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.contracts.intelligence.db_schemas import (
    ExportBundle,
    LaneRun,
    Manifest,
    ManifestPointer,
    PublishedObject,
)
from src.contracts.intelligence.lanes import LANE_NARRATIVE
from src.contracts.intelligence.schemas import (
    Lineage,
    ManifestHeader,
    PublishedObjectRef,
    PublishState,
)
from src.contracts.intelligence.version import ContractRegistry


def _now() -> datetime:
    return datetime.now(UTC)


# -- Pydantic contract model fixtures --------------------------------------


def make_lineage(**overrides: Any) -> Lineage:
    """Create a minimal valid Lineage instance."""
    defaults = {
        "source_ids": ["doc_fixture_1"],
        "lane": LANE_NARRATIVE,
        "run_id": "run_fixture_001",
    }
    defaults.update(overrides)
    return Lineage(**defaults)


def make_manifest_header(**overrides: Any) -> ManifestHeader:
    """Create a minimal valid ManifestHeader instance."""
    defaults = {
        "manifest_id": "manifest_fixture_001",
        "lane": LANE_NARRATIVE,
        "run_id": "run_fixture_001",
        "object_count": 5,
        "checksum": "sha256:fixture",
    }
    defaults.update(overrides)
    return ManifestHeader(**defaults)


def make_published_object_ref(**overrides: Any) -> PublishedObjectRef:
    """Create a minimal valid PublishedObjectRef instance."""
    defaults = {
        "object_id": "obj_fixture_001",
        "object_type": "claim",
        "manifest_id": "manifest_fixture_001",
        "lane": LANE_NARRATIVE,
        "publish_state": PublishState.PUBLISHED,
    }
    defaults.update(overrides)
    return PublishedObjectRef(**defaults)


# -- DB schema fixtures ----------------------------------------------------


def make_lane_run(**overrides: Any) -> LaneRun:
    """Create a minimal valid LaneRun instance."""
    defaults = {
        "run_id": "run_fixture_001",
        "lane": LANE_NARRATIVE,
        "status": "completed",
        "contract_version": str(ContractRegistry.CURRENT),
    }
    defaults.update(overrides)
    return LaneRun(**defaults)


def make_manifest(**overrides: Any) -> Manifest:
    """Create a minimal valid Manifest instance."""
    defaults = {
        "manifest_id": "manifest_fixture_001",
        "lane": LANE_NARRATIVE,
        "run_id": "run_fixture_001",
        "contract_version": str(ContractRegistry.CURRENT),
        "object_count": 5,
    }
    defaults.update(overrides)
    return Manifest(**defaults)


def make_manifest_pointer(**overrides: Any) -> ManifestPointer:
    """Create a minimal valid ManifestPointer instance."""
    defaults = {
        "lane": LANE_NARRATIVE,
        "manifest_id": "manifest_fixture_001",
    }
    defaults.update(overrides)
    return ManifestPointer(**defaults)


def make_published_object(**overrides: Any) -> PublishedObject:
    """Create a minimal valid PublishedObject instance."""
    defaults = {
        "object_id": "obj_fixture_001",
        "object_type": "claim",
        "manifest_id": "manifest_fixture_001",
        "lane": LANE_NARRATIVE,
        "publish_state": "published",
        "contract_version": str(ContractRegistry.CURRENT),
        "run_id": "run_fixture_001",
        "payload": {"text": "Fixture claim content"},
        "source_ids": ["doc_fixture_1"],
    }
    defaults.update(overrides)
    return PublishedObject(**defaults)


def make_export_bundle(**overrides: Any) -> ExportBundle:
    """Create a minimal valid ExportBundle instance."""
    defaults = {
        "bundle_id": "bundle_fixture_001",
        "manifest_id": "manifest_fixture_001",
        "lane": LANE_NARRATIVE,
        "contract_version": str(ContractRegistry.CURRENT),
        "format": "jsonl",
        "object_count": 5,
    }
    defaults.update(overrides)
    return ExportBundle(**defaults)
