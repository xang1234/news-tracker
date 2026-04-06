"""Tests for intelligence layer point-in-time reconstruction.

Verifies no-lookahead filtering for claims, assertions, filings,
lane runs, and manifests, plus snapshot building and validation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.assertions.schemas import ResolvedAssertion
from src.backtest.intelligence_pit import (
    IntelligenceSnapshot,
    build_intelligence_snapshot,
    filter_assertions_pit,
    filter_claims_pit,
    filter_filings_pit,
    filter_lane_runs_pit,
    filter_manifests_pit,
    get_active_manifest_pit,
    validate_no_lookahead,
)
from src.contracts.intelligence.db_schemas import LaneRun, Manifest

NOW = datetime(2026, 4, 1, tzinfo=UTC)
PAST = NOW - timedelta(days=7)
FUTURE = NOW + timedelta(days=7)


# -- Helpers ---------------------------------------------------------------


def _claim(
    claim_id: str = "clk_001",
    created_at: datetime = PAST,
) -> dict[str, Any]:
    return {"claim_id": claim_id, "created_at": created_at, "status": "active"}


def _assertion(
    assertion_id: str = "asrt_001",
    created_at: datetime = PAST,
    status: str = "active",
) -> ResolvedAssertion:
    return ResolvedAssertion(
        assertion_id=assertion_id,
        subject_concept_id="concept_a",
        predicate="supplies_to",
        confidence=0.8,
        status=status,
        created_at=created_at,
    )


def _filing(
    accession: str = "acc-001",
    ingested_at: datetime = PAST,
) -> dict[str, Any]:
    return {"accession_number": accession, "ingested_at": ingested_at}


def _run(
    run_id: str = "run_001",
    lane: str = "narrative",
    status: str = "completed",
    completed_at: datetime | None = None,
) -> LaneRun:
    return LaneRun(
        run_id=run_id,
        lane=lane,
        status=status,
        contract_version="0.1.0",
        completed_at=completed_at or PAST,
    )


_UNSET = object()


def _manifest(
    manifest_id: str = "manifest_001",
    lane: str = "narrative",
    published_at: datetime | None | object = _UNSET,
) -> Manifest:
    return Manifest(
        manifest_id=manifest_id,
        lane=lane,
        run_id="run_001",
        contract_version="0.1.0",
        published_at=PAST if published_at is _UNSET else published_at,  # type: ignore[arg-type]
    )


# -- Claim PIT tests -------------------------------------------------------


class TestClaimsPIT:
    """Claims filtered by created_at."""

    def test_past_claim_visible(self) -> None:
        result = filter_claims_pit([_claim(created_at=PAST)], NOW)
        assert len(result) == 1

    def test_future_claim_hidden(self) -> None:
        result = filter_claims_pit([_claim(created_at=FUTURE)], NOW)
        assert len(result) == 0

    def test_exact_boundary(self) -> None:
        result = filter_claims_pit([_claim(created_at=NOW)], NOW)
        assert len(result) == 1

    def test_missing_created_at(self) -> None:
        result = filter_claims_pit([{"claim_id": "x"}], NOW)
        assert len(result) == 0

    def test_empty_list(self) -> None:
        assert filter_claims_pit([], NOW) == []


# -- Assertion PIT tests ---------------------------------------------------


class TestAssertionsPIT:
    """Assertions filtered by created_at."""

    def test_past_visible(self) -> None:
        result = filter_assertions_pit([_assertion(created_at=PAST)], NOW)
        assert len(result) == 1

    def test_future_hidden(self) -> None:
        result = filter_assertions_pit([_assertion(created_at=FUTURE)], NOW)
        assert len(result) == 0

    def test_retracted_still_visible(self) -> None:
        """Retracted assertions are visible — their status tells the backtest."""
        a = _assertion(created_at=PAST, status="retracted")
        result = filter_assertions_pit([a], NOW)
        assert len(result) == 1
        assert result[0].status == "retracted"

    def test_exact_boundary(self) -> None:
        result = filter_assertions_pit([_assertion(created_at=NOW)], NOW)
        assert len(result) == 1


# -- Filing PIT tests ------------------------------------------------------


class TestFilingsPIT:
    """Filings filtered by ingested_at."""

    def test_past_visible(self) -> None:
        result = filter_filings_pit([_filing(ingested_at=PAST)], NOW)
        assert len(result) == 1

    def test_future_hidden(self) -> None:
        result = filter_filings_pit([_filing(ingested_at=FUTURE)], NOW)
        assert len(result) == 0

    def test_missing_ingested_at(self) -> None:
        result = filter_filings_pit([{"accession_number": "x"}], NOW)
        assert len(result) == 0

    def test_uses_ingested_not_published(self) -> None:
        """Filing published before as_of but ingested after → hidden."""
        filing = {
            "accession_number": "acc-late",
            "source_published_at": PAST,
            "ingested_at": FUTURE,
        }
        result = filter_filings_pit([filing], NOW)
        assert len(result) == 0


# -- Lane run PIT tests ----------------------------------------------------


class TestLaneRunsPIT:
    """Lane runs filtered by completed_at."""

    def test_completed_visible(self) -> None:
        result = filter_lane_runs_pit([_run(completed_at=PAST)], NOW)
        assert len(result) == 1

    def test_future_hidden(self) -> None:
        result = filter_lane_runs_pit([_run(completed_at=FUTURE)], NOW)
        assert len(result) == 0

    def test_running_hidden(self) -> None:
        result = filter_lane_runs_pit([_run(status="running")], NOW)
        assert len(result) == 0

    def test_failed_hidden(self) -> None:
        result = filter_lane_runs_pit(
            [_run(status="failed", completed_at=PAST)],
            NOW,
        )
        assert len(result) == 0

    def test_lane_filter(self) -> None:
        runs = [_run(lane="narrative"), _run(run_id="r2", lane="filing")]
        result = filter_lane_runs_pit(runs, NOW, lane="narrative")
        assert len(result) == 1
        assert result[0].lane == "narrative"

    def test_no_completed_at(self) -> None:
        run = LaneRun(
            run_id="r",
            lane="narrative",
            status="completed",
            contract_version="0.1.0",
            completed_at=None,
        )
        result = filter_lane_runs_pit([run], NOW)
        assert len(result) == 0


# -- Manifest PIT tests ----------------------------------------------------


class TestManifestsPIT:
    """Manifests filtered by published_at."""

    def test_sealed_visible(self) -> None:
        result = filter_manifests_pit([_manifest(published_at=PAST)], NOW)
        assert len(result) == 1

    def test_future_hidden(self) -> None:
        result = filter_manifests_pit([_manifest(published_at=FUTURE)], NOW)
        assert len(result) == 0

    def test_unsealed_hidden(self) -> None:
        result = filter_manifests_pit([_manifest(published_at=None)], NOW)
        assert len(result) == 0

    def test_lane_filter(self) -> None:
        manifests = [
            _manifest("m_n", lane="narrative"),
            _manifest("m_f", lane="filing"),
        ]
        result = filter_manifests_pit(manifests, NOW, lane="narrative")
        assert len(result) == 1


# -- Active manifest PIT tests ---------------------------------------------


class TestActiveManifestPIT:
    """Most recent sealed manifest per lane at as_of."""

    def test_latest_selected(self) -> None:
        manifests = [
            _manifest("m_old", published_at=PAST - timedelta(days=1)),
            _manifest("m_new", published_at=PAST),
        ]
        active = get_active_manifest_pit(manifests, NOW, "narrative")
        assert active is not None
        assert active.manifest_id == "m_new"

    def test_none_when_no_manifests(self) -> None:
        assert get_active_manifest_pit([], NOW, "narrative") is None

    def test_future_excluded(self) -> None:
        manifests = [_manifest(published_at=FUTURE)]
        assert get_active_manifest_pit(manifests, NOW, "narrative") is None

    def test_lane_specific(self) -> None:
        manifests = [
            _manifest("m_n", lane="narrative", published_at=PAST),
            _manifest("m_f", lane="filing", published_at=PAST),
        ]
        active = get_active_manifest_pit(manifests, NOW, "filing")
        assert active is not None
        assert active.manifest_id == "m_f"


# -- Snapshot builder tests ------------------------------------------------


class TestBuildSnapshot:
    """Build complete PIT snapshot from unfiltered data."""

    def test_basic_snapshot(self) -> None:
        snap = build_intelligence_snapshot(
            NOW,
            claims=[_claim(created_at=PAST), _claim("c2", created_at=FUTURE)],
            assertions=[_assertion(created_at=PAST)],
            filings=[_filing(ingested_at=PAST)],
            lane_runs=[_run(completed_at=PAST)],
            manifests=[_manifest(published_at=PAST)],
        )
        assert snap.as_of == NOW
        assert len(snap.claims) == 1
        assert len(snap.assertions) == 1
        assert len(snap.filings) == 1
        assert len(snap.lane_runs) == 1
        assert len(snap.manifests) == 1

    def test_active_manifests_populated(self) -> None:
        snap = build_intelligence_snapshot(
            NOW,
            claims=[],
            assertions=[],
            filings=[],
            lane_runs=[],
            manifests=[
                _manifest("m_n", lane="narrative", published_at=PAST),
                _manifest("m_f", lane="filing", published_at=PAST),
            ],
        )
        assert "narrative" in snap.active_manifests
        assert "filing" in snap.active_manifests

    def test_empty_inputs(self) -> None:
        snap = build_intelligence_snapshot(
            NOW,
            claims=[],
            assertions=[],
            filings=[],
            lane_runs=[],
            manifests=[],
        )
        assert snap.claims == []
        assert snap.active_manifests == {}

    def test_to_dict(self) -> None:
        snap = build_intelligence_snapshot(
            NOW,
            claims=[_claim()],
            assertions=[],
            filings=[],
            lane_runs=[],
            manifests=[],
        )
        d = snap.to_dict()
        assert d["claim_count"] == 1
        assert isinstance(d["as_of"], str)
        assert "active_manifest_lanes" in d


# -- No-lookahead validation tests -----------------------------------------


class TestNoLookahead:
    """Validate no data leaks past as_of."""

    def test_clean_snapshot(self) -> None:
        snap = build_intelligence_snapshot(
            NOW,
            claims=[_claim(created_at=PAST)],
            assertions=[_assertion(created_at=PAST)],
            filings=[_filing(ingested_at=PAST)],
            lane_runs=[_run(completed_at=PAST)],
            manifests=[_manifest(published_at=PAST)],
        )
        assert validate_no_lookahead(snap) == []

    def test_future_claim_violation(self) -> None:
        snap = IntelligenceSnapshot(
            as_of=NOW,
            claims=[_claim(created_at=FUTURE)],
        )
        violations = validate_no_lookahead(snap)
        assert len(violations) == 1
        assert "Claim" in violations[0]

    def test_future_assertion_violation(self) -> None:
        snap = IntelligenceSnapshot(
            as_of=NOW,
            assertions=[_assertion(created_at=FUTURE)],
        )
        violations = validate_no_lookahead(snap)
        assert len(violations) == 1
        assert "Assertion" in violations[0]

    def test_future_filing_violation(self) -> None:
        snap = IntelligenceSnapshot(
            as_of=NOW,
            filings=[_filing(ingested_at=FUTURE)],
        )
        violations = validate_no_lookahead(snap)
        assert len(violations) == 1
        assert "Filing" in violations[0]

    def test_future_run_violation(self) -> None:
        snap = IntelligenceSnapshot(
            as_of=NOW,
            lane_runs=[_run(completed_at=FUTURE)],
        )
        violations = validate_no_lookahead(snap)
        assert len(violations) == 1
        assert "LaneRun" in violations[0]

    def test_future_manifest_violation(self) -> None:
        snap = IntelligenceSnapshot(
            as_of=NOW,
            manifests=[_manifest(published_at=FUTURE)],
        )
        violations = validate_no_lookahead(snap)
        assert len(violations) == 1
        assert "Manifest" in violations[0]

    def test_multiple_violations(self) -> None:
        snap = IntelligenceSnapshot(
            as_of=NOW,
            claims=[_claim(created_at=FUTURE)],
            assertions=[_assertion(created_at=FUTURE)],
        )
        violations = validate_no_lookahead(snap)
        assert len(violations) == 2

    def test_empty_snapshot_clean(self) -> None:
        snap = IntelligenceSnapshot(as_of=NOW)
        assert validate_no_lookahead(snap) == []


# -- Dataclass tests -------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_snapshot_frozen(self) -> None:
        snap = build_intelligence_snapshot(
            NOW,
            claims=[],
            assertions=[],
            filings=[],
            lane_runs=[],
            manifests=[],
        )
        try:
            snap.as_of = FUTURE  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass
