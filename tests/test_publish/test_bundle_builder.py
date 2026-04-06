"""Tests for composite bundle builder.

Verifies that composite bundles are built with correct artifacts,
checksums, integrity verification, and parity checking.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.publish.bundle_builder import (
    BundleArtifact,
    CompositeBundle,
    build_composite_bundle,
    build_manifest_artifact,
    build_objects_artifact,
    build_rollups_artifact,
    check_bundle_parity,
    verify_bundle_integrity,
)
from src.publish.exporter import compute_bundle_checksum
from src.publish.lane_health import PublishReadiness
from src.publish.manifest_assembly import CompositeManifest, LaneContribution

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _composite(
    composite_id: str = "composite_abc123",
    lanes: list[str] | None = None,
) -> CompositeManifest:
    if lanes is None:
        lanes = ["narrative", "filing"]
    contributions = [
        LaneContribution(
            lane=lane,
            manifest_id=f"manifest_{lane}",
            object_count=5,
            readiness=PublishReadiness.READY,
        )
        for lane in lanes
    ]
    return CompositeManifest(
        composite_id=composite_id,
        contributions=contributions,
        total_object_count=sum(c.object_count for c in contributions),
        contract_version="0.1.0",
        assembled_at=NOW,
    )


def _objects(lanes: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    if lanes is None:
        lanes = ["narrative", "filing"]
    return {
        lane: [
            {"object_id": f"{lane}_obj_{i}", "type": "test", "lane": lane}
            for i in range(3)
        ]
        for lane in lanes
    }


def _rollups(count: int = 2) -> list[dict[str, Any]]:
    return [{"rollup_id": f"roll_{i}", "type": "summary"} for i in range(count)]


# -- Checksum tests --------------------------------------------------------


class TestChecksum:
    """SHA-256 checksum utility."""

    def test_deterministic(self) -> None:
        c1 = compute_bundle_checksum(["line1", "line2"])
        c2 = compute_bundle_checksum(["line1", "line2"])
        assert c1 == c2

    def test_prefix(self) -> None:
        assert compute_bundle_checksum(["test"]).startswith("sha256:")

    def test_different_content(self) -> None:
        c1 = compute_bundle_checksum(["a"])
        c2 = compute_bundle_checksum(["b"])
        assert c1 != c2

    def test_empty(self) -> None:
        c = compute_bundle_checksum([])
        assert c.startswith("sha256:")


# -- Objects artifact tests ------------------------------------------------


class TestObjectsArtifact:
    """Build objects.jsonl from per-lane objects."""

    def test_basic(self) -> None:
        art = build_objects_artifact(_objects(["narrative"]))
        assert art.name == "objects.jsonl"
        assert art.record_count == 3
        assert len(art.lines) == 3

    def test_multi_lane_combined(self) -> None:
        art = build_objects_artifact(_objects(["narrative", "filing"]))
        assert art.record_count == 6

    def test_sorted_by_object_id(self) -> None:
        import json
        art = build_objects_artifact(_objects(["narrative", "filing"]))
        ids = [json.loads(line)["object_id"] for line in art.lines]
        assert ids == sorted(ids)

    def test_empty_lanes(self) -> None:
        art = build_objects_artifact({})
        assert art.record_count == 0
        assert art.lines == []

    def test_checksum_present(self) -> None:
        art = build_objects_artifact(_objects())
        assert art.checksum.startswith("sha256:")


# -- Rollups artifact tests ------------------------------------------------


class TestRollupsArtifact:
    """Build rollups.jsonl from summary records."""

    def test_basic(self) -> None:
        art = build_rollups_artifact(_rollups(3))
        assert art.name == "rollups.jsonl"
        assert art.record_count == 3

    def test_empty(self) -> None:
        art = build_rollups_artifact([])
        assert art.record_count == 0

    def test_checksum_present(self) -> None:
        art = build_rollups_artifact(_rollups())
        assert art.checksum.startswith("sha256:")


# -- Manifest artifact tests -----------------------------------------------


class TestManifestArtifact:
    """Build manifest.json with metadata and checksums."""

    def test_basic(self) -> None:
        checksums = {"objects.jsonl": "sha256:abc", "rollups.jsonl": "sha256:def"}
        art = build_manifest_artifact(_composite(), checksums, NOW)
        assert art.name == "manifest.json"
        assert art.record_count == 1
        assert art.checksum.startswith("sha256:")

    def test_contains_composite_metadata(self) -> None:
        import json
        checksums = {"objects.jsonl": "sha256:abc"}
        art = build_manifest_artifact(_composite(), checksums, NOW)
        data = json.loads(art.lines[0])
        assert data["_type"] == "composite_bundle_manifest"
        assert data["composite_id"] == "composite_abc123"
        assert data["contract_version"] == "0.1.0"
        assert "narrative" in data["lanes_included"]

    def test_includes_artifact_checksums(self) -> None:
        import json
        checksums = {"objects.jsonl": "sha256:abc", "rollups.jsonl": "sha256:def"}
        art = build_manifest_artifact(_composite(), checksums, NOW)
        data = json.loads(art.lines[0])
        assert data["artifact_checksums"] == checksums

    def test_includes_contributions(self) -> None:
        import json
        art = build_manifest_artifact(_composite(), {}, NOW)
        data = json.loads(art.lines[0])
        assert len(data["contributions"]) == 2
        assert data["contributions"][0]["lane"] == "narrative"


# -- Composite bundle tests ------------------------------------------------


class TestBuildCompositeBundle:
    """Full bundle building pipeline."""

    def test_basic_bundle(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        assert bundle.composite_id == "composite_abc123"
        assert bundle.contract_version == "0.1.0"
        assert "manifest.json" in bundle.artifacts
        assert "objects.jsonl" in bundle.artifacts
        assert "rollups.jsonl" in bundle.artifacts

    def test_three_artifacts(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        assert len(bundle.artifacts) == 3

    def test_overall_checksum_is_manifestcompute_bundle_checksum(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        assert bundle.overall_checksum == bundle.artifacts["manifest.json"].checksum

    def test_total_records(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(3), now=NOW,
        )
        # 6 objects + 3 rollups + 1 manifest = 10
        assert bundle.total_records == 10

    def test_empty_inputs(self) -> None:
        bundle = build_composite_bundle(
            _composite(lanes=[]), {}, [], now=NOW,
        )
        assert bundle.artifacts["objects.jsonl"].record_count == 0
        assert bundle.artifacts["rollups.jsonl"].record_count == 0
        assert bundle.total_records == 1  # just the manifest

    def test_deterministic(self) -> None:
        b1 = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        b2 = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        assert b1.overall_checksum == b2.overall_checksum

    def test_created_at(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        assert bundle.created_at == NOW

    def test_to_dict(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        d = bundle.to_dict()
        assert d["composite_id"] == "composite_abc123"
        assert "artifact_names" in d
        assert "overall_checksum" in d


# -- Integrity verification tests -----------------------------------------


class TestVerifyIntegrity:
    """Checksum-based integrity verification."""

    def test_valid_bundle(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        assert verify_bundle_integrity(bundle) is True

    def test_tampered_bundle(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        # Tamper by replacing an artifact with wrong checksum
        tampered = BundleArtifact(
            name="objects.jsonl",
            lines=["tampered content"],
            checksum="sha256:wrong",
            record_count=1,
        )
        tampered_artifacts = dict(bundle.artifacts)
        tampered_artifacts["objects.jsonl"] = tampered
        tampered_bundle = CompositeBundle(
            composite_id=bundle.composite_id,
            contract_version=bundle.contract_version,
            artifacts=tampered_artifacts,
            overall_checksum=bundle.overall_checksum,
            created_at=NOW,
        )
        assert verify_bundle_integrity(tampered_bundle) is False


# -- Parity checking tests -------------------------------------------------


class TestBundleParity:
    """Bundle vs expected state parity."""

    def test_parity_holds(self) -> None:
        objects = _objects()
        rollups = _rollups()
        bundle = build_composite_bundle(
            _composite(), objects, rollups, now=NOW,
        )
        mismatches = check_bundle_parity(bundle, objects, rollups)
        assert mismatches == []

    def test_object_mismatch(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        different_objects = {"narrative": [{"object_id": "different", "type": "x"}]}
        mismatches = check_bundle_parity(bundle, different_objects, _rollups())
        assert len(mismatches) == 1
        assert "objects.jsonl" in mismatches[0]

    def test_rollup_mismatch(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(2), now=NOW,
        )
        mismatches = check_bundle_parity(bundle, _objects(), _rollups(3))
        assert len(mismatches) == 1
        assert "rollups.jsonl" in mismatches[0]

    def test_both_mismatches(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        mismatches = check_bundle_parity(
            bundle,
            {"narrative": [{"object_id": "x"}]},
            [{"rollup_id": "x"}],
        )
        assert len(mismatches) == 2


# -- Dataclass tests -------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_artifact_frozen(self) -> None:
        art = build_objects_artifact(_objects())
        try:
            art.checksum = "tampered"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_bundle_frozen(self) -> None:
        bundle = build_composite_bundle(
            _composite(), _objects(), _rollups(), now=NOW,
        )
        try:
            bundle.overall_checksum = "tampered"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass
