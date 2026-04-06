"""Tests for composite manifest assembly and pointer advancement.

Verifies that lane outputs are evaluated for inclusion, composites
are built from approved lanes, pointer advancements are planned,
and diagnostics capture exclusion reasons.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.publish.lane_health import (
    FreshnessLevel,
    LaneHealthStatus,
    PublishReadiness,
    QualityLevel,
    QuarantineState,
)
from src.publish.manifest_assembly import (
    LaneOutput,
    assemble_composite_manifest,
    make_composite_id,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _healthy() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="narrative",
        freshness=FreshnessLevel.FRESH,
        quality=QualityLevel.HEALTHY,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.READY,
    )


def _warn() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="filing",
        freshness=FreshnessLevel.AGING,
        quality=QualityLevel.HEALTHY,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.WARN,
    )


def _blocked() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="structural",
        freshness=FreshnessLevel.STALE,
        quality=QualityLevel.CRITICAL,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.BLOCKED,
    )


def _output(
    lane: str = "narrative",
    published: bool = True,
    object_count: int = 10,
    manifest_id: str = "manifest_001",
    health: LaneHealthStatus | None = None,
    block_reason: str | None = None,
) -> LaneOutput:
    if health is None:
        health = LaneHealthStatus(
            lane=lane,
            freshness=FreshnessLevel.FRESH,
            quality=QualityLevel.HEALTHY,
            quarantine=QuarantineState.CLEAR,
            readiness=PublishReadiness.READY,
        )
    return LaneOutput(
        lane=lane,
        published=published,
        object_count=object_count,
        manifest_id=manifest_id,
        health=health,
        block_reason=block_reason,
    )


# -- Composite ID tests ----------------------------------------------------


class TestCompositeId:
    """Deterministic composite ID generation."""

    def test_deterministic(self) -> None:
        id1 = make_composite_id(["m_a", "m_b"])
        id2 = make_composite_id(["m_a", "m_b"])
        assert id1 == id2

    def test_order_independent(self) -> None:
        """Same manifest IDs in different order produce same ID."""
        id1 = make_composite_id(["m_a", "m_b"])
        id2 = make_composite_id(["m_b", "m_a"])
        assert id1 == id2

    def test_different_inputs_different_ids(self) -> None:
        id1 = make_composite_id(["m_a", "m_b"])
        id2 = make_composite_id(["m_a", "m_c"])
        assert id1 != id2

    def test_prefix(self) -> None:
        cid = make_composite_id(["m_a"])
        assert cid.startswith("composite_")

    def test_empty_list(self) -> None:
        """Empty manifest list still produces a valid ID."""
        cid = make_composite_id([])
        assert cid.startswith("composite_")


# -- Lane inclusion/exclusion tests ----------------------------------------


class TestLaneInclusion:
    """Which lanes are included in the composite."""

    def test_all_healthy_included(self) -> None:
        outputs = [
            _output(lane="narrative", manifest_id="m_n", object_count=5),
            _output(lane="filing", manifest_id="m_f", object_count=3),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert result.ready is True
        assert set(result.composite.lanes_included) == {"narrative", "filing"}

    def test_blocked_excluded(self) -> None:
        outputs = [
            _output(lane="narrative", manifest_id="m_n"),
            _output(
                lane="filing",
                published=False,
                block_reason="Lane blocked",
                health=_blocked(),
            ),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert "narrative" in result.composite.lanes_included
        assert "filing" in result.composite.lanes_excluded
        assert "filing" in result.composite.exclusion_reasons

    def test_empty_objects_excluded(self) -> None:
        outputs = [
            _output(lane="narrative", object_count=0, manifest_id="m_n"),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert "narrative" in result.composite.lanes_excluded
        assert "No objects" in result.composite.exclusion_reasons["narrative"]

    def test_unpublished_excluded(self) -> None:
        outputs = [
            _output(lane="narrative", published=False, block_reason="Health blocked"),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert result.ready is False
        assert "narrative" in result.composite.lanes_excluded

    def test_warn_included(self) -> None:
        """WARN readiness still publishes (not blocked)."""
        outputs = [
            _output(lane="filing", manifest_id="m_f", health=_warn()),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert "filing" in result.composite.lanes_included

    def test_blocked_health_excluded_even_if_published_true(self) -> None:
        """Edge case: published=True but health is BLOCKED → exclude."""
        blocked_health = LaneHealthStatus(
            lane="structural",
            freshness=FreshnessLevel.STALE,
            quality=QualityLevel.CRITICAL,
            quarantine=QuarantineState.CLEAR,
            readiness=PublishReadiness.BLOCKED,
        )
        outputs = [
            _output(lane="structural", published=True, health=blocked_health),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert "structural" in result.composite.lanes_excluded

    def test_no_outputs(self) -> None:
        result = assemble_composite_manifest([], {}, now=NOW)
        assert result.ready is False
        assert result.composite.total_object_count == 0
        assert result.composite.composite_id == "composite_empty"


# -- Pointer advancement tests ---------------------------------------------


class TestPointerAdvancement:
    """Planned pointer movements for atomic execution."""

    def test_advancement_per_included_lane(self) -> None:
        outputs = [
            _output(lane="narrative", manifest_id="m_n"),
            _output(lane="filing", manifest_id="m_f"),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        lanes = {a.lane for a in result.advancements}
        assert lanes == {"narrative", "filing"}

    def test_previous_manifest_from_pointers(self) -> None:
        outputs = [_output(lane="narrative", manifest_id="m_new")]
        pointers = {"narrative": "m_old"}
        result = assemble_composite_manifest(outputs, pointers, now=NOW)
        adv = result.advancements[0]
        assert adv.manifest_id == "m_new"
        assert adv.previous_manifest_id == "m_old"

    def test_no_previous_pointer(self) -> None:
        outputs = [_output(lane="narrative", manifest_id="m_first")]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert result.advancements[0].previous_manifest_id is None

    def test_excluded_lanes_no_advancement(self) -> None:
        outputs = [
            _output(lane="narrative", published=False, block_reason="blocked"),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert result.advancements == []


# -- Composite metadata tests ----------------------------------------------


class TestCompositeMetadata:
    """Composite manifest metadata and diagnostics."""

    def test_total_object_count(self) -> None:
        outputs = [
            _output(lane="narrative", object_count=10, manifest_id="m_n"),
            _output(lane="filing", object_count=5, manifest_id="m_f"),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert result.composite.total_object_count == 15

    def test_contract_version_default(self) -> None:
        outputs = [_output(manifest_id="m_n")]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        from src.contracts.intelligence.version import ContractRegistry
        assert result.composite.contract_version == str(ContractRegistry.CURRENT)

    def test_contract_version_override(self) -> None:
        outputs = [_output(manifest_id="m_n")]
        result = assemble_composite_manifest(
            outputs, {}, contract_version="1.2.3", now=NOW,
        )
        assert result.composite.contract_version == "1.2.3"

    def test_assembled_at(self) -> None:
        result = assemble_composite_manifest([], {}, now=NOW)
        assert result.composite.assembled_at == NOW

    def test_block_reasons_collected(self) -> None:
        outputs = [
            _output(lane="narrative", published=False, block_reason="Lane stale"),
            _output(lane="filing", object_count=0, manifest_id="m_f"),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        assert len(result.block_reasons) == 2
        assert any("narrative" in r for r in result.block_reasons)
        assert any("filing" in r for r in result.block_reasons)

    def test_composite_id_deterministic(self) -> None:
        outputs = [
            _output(lane="narrative", manifest_id="m_n"),
            _output(lane="filing", manifest_id="m_f"),
        ]
        r1 = assemble_composite_manifest(outputs, {}, now=NOW)
        r2 = assemble_composite_manifest(outputs, {}, now=NOW)
        assert r1.composite.composite_id == r2.composite.composite_id

    def test_to_dict(self) -> None:
        outputs = [_output(manifest_id="m_n")]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        d = result.composite.to_dict()
        assert "composite_id" in d
        assert "lanes_included" in d
        assert "exclusion_reasons" in d
        assert isinstance(d["assembled_at"], str)

    def test_contributions_track_readiness(self) -> None:
        outputs = [
            _output(lane="narrative", manifest_id="m_n"),
            _output(lane="filing", manifest_id="m_f", health=_warn()),
        ]
        result = assemble_composite_manifest(outputs, {}, now=NOW)
        readiness = {
            c.lane: c.readiness for c in result.composite.contributions
        }
        assert readiness["narrative"] == PublishReadiness.READY
        assert readiness["filing"] == PublishReadiness.WARN


# -- Integration: realistic scenario ----------------------------------------


class TestRealisticScenario:
    """Full assembly with mixed lane states."""

    def test_mixed_lanes(self) -> None:
        """Narrative ready, filing warn, structural blocked."""
        outputs = [
            _output(lane="narrative", manifest_id="m_n", object_count=15),
            _output(lane="filing", manifest_id="m_f", object_count=8,
                    health=_warn()),
            _output(lane="structural", published=False,
                    block_reason="Lane structural is blocked",
                    health=_blocked()),
        ]
        pointers = {"narrative": "m_n_old", "filing": "m_f_old"}
        result = assemble_composite_manifest(outputs, pointers, now=NOW)

        assert result.ready is True
        assert set(result.composite.lanes_included) == {"narrative", "filing"}
        assert "structural" in result.composite.lanes_excluded
        assert result.composite.total_object_count == 23

        adv_lanes = {a.lane for a in result.advancements}
        assert adv_lanes == {"narrative", "filing"}

        narr_adv = next(a for a in result.advancements if a.lane == "narrative")
        assert narr_adv.manifest_id == "m_n"
        assert narr_adv.previous_manifest_id == "m_n_old"


# -- Dataclass tests -------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_lane_output_frozen(self) -> None:
        o = _output()
        try:
            o.published = False  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_composite_frozen(self) -> None:
        result = assemble_composite_manifest([], {}, now=NOW)
        try:
            result.composite.total_object_count = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_assembly_result_frozen(self) -> None:
        result = assemble_composite_manifest([], {}, now=NOW)
        try:
            result.ready = True  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass
