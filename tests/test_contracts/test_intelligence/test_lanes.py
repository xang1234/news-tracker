"""Tests for intelligence lane definitions."""

import pytest

from src.contracts.intelligence.lanes import (
    ALL_LANES,
    LANE_BACKTEST,
    LANE_FILING,
    LANE_NARRATIVE,
    LANE_STRUCTURAL,
    VALID_LANES,
    LaneDescriptor,
    LaneRegistry,
    validate_lane,
)


class TestLaneConstants:
    """Lane name constants and validation."""

    def test_all_lanes_in_valid_set(self) -> None:
        for lane in ALL_LANES:
            assert lane in VALID_LANES

    def test_valid_lanes_matches_all_lanes(self) -> None:
        assert VALID_LANES == frozenset(ALL_LANES)

    def test_named_constants_in_all_lanes(self) -> None:
        expected = {LANE_NARRATIVE, LANE_FILING, LANE_STRUCTURAL, LANE_BACKTEST}
        assert expected == set(ALL_LANES)

    def test_lane_names_are_lowercase_identifiers(self) -> None:
        for lane in ALL_LANES:
            assert lane == lane.lower()
            assert lane.isidentifier()


class TestValidateLane:
    """validate_lane() function."""

    @pytest.mark.parametrize("lane", ALL_LANES)
    def test_valid_lanes_pass(self, lane: str) -> None:
        assert validate_lane(lane) == lane

    def test_unknown_lane_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown lane"):
            validate_lane("nonexistent")

    def test_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="Unknown lane"):
            validate_lane("NARRATIVE")


class TestLaneDescriptor:
    """LaneDescriptor validation."""

    def test_valid_descriptor(self) -> None:
        desc = LaneDescriptor(
            name=LANE_NARRATIVE,
            description="Test",
            source_types=("news",),
            produces=("claims",),
        )
        assert desc.name == LANE_NARRATIVE

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown lane"):
            LaneDescriptor(
                name="bad_lane",
                description="Test",
                source_types=(),
                produces=(),
            )

    def test_immutable(self) -> None:
        desc = LaneDescriptor(
            name=LANE_NARRATIVE,
            description="Test",
            source_types=(),
            produces=(),
        )
        with pytest.raises(AttributeError):
            desc.name = "other"  # type: ignore[misc]


class TestLaneRegistry:
    """LaneRegistry coverage and consistency."""

    def test_every_lane_has_descriptor(self) -> None:
        for lane in ALL_LANES:
            desc = LaneRegistry.get(lane)
            assert desc.name == lane

    def test_descriptors_have_descriptions(self) -> None:
        for lane in ALL_LANES:
            desc = LaneRegistry.get(lane)
            assert len(desc.description) > 10

    def test_descriptors_have_source_types(self) -> None:
        for lane in ALL_LANES:
            desc = LaneRegistry.get(lane)
            assert len(desc.source_types) > 0

    def test_descriptors_have_produces(self) -> None:
        for lane in ALL_LANES:
            desc = LaneRegistry.get(lane)
            assert len(desc.produces) > 0

    def test_get_unknown_lane_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown lane"):
            LaneRegistry.get("nonexistent")
