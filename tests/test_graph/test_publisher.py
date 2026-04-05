"""Tests for structural lane publication.

Verifies that scored paths and baskets are prepared for manifest
publication with explanation-ready path summaries and lane health
gating.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.graph.baskets import BasketMember, ThematicBasket
from src.graph.path_scoring import PathScoreBreakdown, ScoredEdge, ScoredPath
from src.graph.publisher import (
    DEFAULT_TOP_N,
    BasketPayload,
    PathExplanation,
    StructuralPublicationResult,
    build_basket_payload,
    build_path_explanation,
    prepare_structural_publication,
)
from src.graph.structural import StructuralRelation
from src.publish.lane_health import (
    FreshnessLevel,
    LaneHealthStatus,
    PublishReadiness,
    QualityLevel,
    QuarantineState,
)

NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)
THEME = "concept_theme_hbm"


# -- Helpers ---------------------------------------------------------------


def _rel(
    source: str = THEME,
    target: str = "concept_issuer_aaa",
    predicate: str = "supplies_to",
    assertion_id: str = "asrt_001",
) -> StructuralRelation:
    return StructuralRelation(
        source_concept_id=source,
        target_concept_id=target,
        predicate=predicate,
        confidence=0.8,
        sign=1,
        assertion_id=assertion_id,
        is_current=True,
        assertion_status="active",
    )


def _scored_edge(
    source: str = THEME,
    target: str = "concept_issuer_aaa",
    predicate: str = "supplies_to",
    assertion_id: str = "asrt_001",
) -> ScoredEdge:
    return ScoredEdge(
        relation=_rel(source, target, predicate, assertion_id),
        freshness_factor=0.9,
        corroboration_factor=0.8,
        edge_score=0.576,
    )


def _path(
    target: str = "concept_issuer_aaa",
    hops: int = 1,
    path_score: float = 0.5,
    path_sign: int = 1,
    intermediate: str | None = None,
    edges: list[ScoredEdge] | None = None,
) -> ScoredPath:
    if edges is None:
        edges = [_scored_edge(target=target)]
    return ScoredPath(
        source_concept_id=THEME,
        target_concept_id=target,
        hops=hops,
        path_score=path_score,
        path_sign=path_sign,
        breakdown=PathScoreBreakdown(
            confidence_product=0.8,
            freshness_product=0.9,
            corroboration_product=0.8,
            hop_decay=0.7 if hops == 2 else 1.0,
        ),
        edges=edges,
        intermediate_concept_id=intermediate,
    )


def _basket(
    beneficiaries: int = 3,
    at_risk: int = 1,
    first_order: int = 2,
    second_order: int = 2,
) -> ThematicBasket:
    bens = [
        BasketMember(
            concept_id=f"ben_{i}",
            role="beneficiary",
            best_score=0.8 - i * 0.1,
            best_sign=1,
            min_hops=1 if i < first_order else 2,
            positive_paths=1,
            negative_paths=0,
        )
        for i in range(beneficiaries)
    ]
    risks = [
        BasketMember(
            concept_id=f"risk_{i}",
            role="at_risk",
            best_score=0.5 - i * 0.1,
            best_sign=-1,
            min_hops=1,
            positive_paths=0,
            negative_paths=1,
        )
        for i in range(at_risk)
    ]
    return ThematicBasket(
        source_concept_id=THEME,
        beneficiaries=bens,
        at_risk=risks,
        first_order_count=first_order,
        second_order_count=second_order,
        computed_at=NOW,
    )


def _healthy_status() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="structural",
        freshness=FreshnessLevel.FRESH,
        quality=QualityLevel.HEALTHY,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.READY,
    )


def _blocked_status() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="structural",
        freshness=FreshnessLevel.STALE,
        quality=QualityLevel.CRITICAL,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.BLOCKED,
    )


# -- Path explanation tests -------------------------------------------------


class TestPathExplanation:
    """Build publishable path explanations from scored paths."""

    def test_basic_conversion(self) -> None:
        p = _path(target="A")
        expl = build_path_explanation(p)
        assert expl.source_concept_id == THEME
        assert expl.target_concept_id == "A"
        assert expl.hops == 1
        assert expl.path_score == 0.5
        assert expl.path_sign == 1

    def test_preserves_breakdown(self) -> None:
        p = _path()
        expl = build_path_explanation(p)
        assert expl.confidence_product == 0.8
        assert expl.freshness_product == 0.9
        assert expl.corroboration_product == 0.8
        assert expl.hop_decay == 1.0

    def test_extracts_assertion_ids(self) -> None:
        e1 = _scored_edge(assertion_id="asrt_001")
        e2 = _scored_edge(target="B", assertion_id="asrt_002")
        p = _path(target="B", hops=2, edges=[e1, e2], intermediate="A")
        expl = build_path_explanation(p)
        assert expl.assertion_ids == ["asrt_001", "asrt_002"]

    def test_extracts_predicates(self) -> None:
        e1 = _scored_edge(predicate="supplies_to")
        e2 = _scored_edge(target="B", predicate="competes_with")
        p = _path(target="B", hops=2, edges=[e1, e2], intermediate="A")
        expl = build_path_explanation(p)
        assert expl.edge_predicates == ["supplies_to", "competes_with"]

    def test_intermediate_preserved(self) -> None:
        p = _path(target="C", hops=2, intermediate="B")
        expl = build_path_explanation(p)
        assert expl.intermediate_concept_id == "B"

    def test_1hop_no_intermediate(self) -> None:
        p = _path(hops=1)
        expl = build_path_explanation(p)
        assert expl.intermediate_concept_id is None

    def test_to_dict(self) -> None:
        expl = build_path_explanation(_path())
        d = expl.to_dict()
        assert d["source_concept_id"] == THEME
        assert "assertion_ids" in d
        assert "edge_predicates" in d
        assert "confidence_product" in d

    def test_frozen(self) -> None:
        expl = build_path_explanation(_path())
        try:
            expl.path_score = 0.0  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


# -- Basket payload tests ---------------------------------------------------


class TestBasketPayload:
    """Build publishable basket summaries."""

    def test_basic_conversion(self) -> None:
        basket = _basket(beneficiaries=3, at_risk=1)
        payload = build_basket_payload(basket)
        assert payload.source_concept_id == THEME
        assert payload.beneficiary_count == 3
        assert payload.at_risk_count == 1

    def test_top_n_limits(self) -> None:
        basket = _basket(beneficiaries=15)
        payload = build_basket_payload(basket, top_n=5)
        assert len(payload.top_beneficiaries) == 5

    def test_default_top_n(self) -> None:
        basket = _basket(beneficiaries=3)
        payload = build_basket_payload(basket)
        assert len(payload.top_beneficiaries) == 3  # fewer than DEFAULT_TOP_N

    def test_preserves_order_counts(self) -> None:
        basket = _basket(first_order=2, second_order=3)
        payload = build_basket_payload(basket)
        assert payload.first_order_count == 2
        assert payload.second_order_count == 3

    def test_top_members_serialized(self) -> None:
        basket = _basket(beneficiaries=2)
        payload = build_basket_payload(basket)
        assert isinstance(payload.top_beneficiaries[0], dict)
        assert "concept_id" in payload.top_beneficiaries[0]
        assert "role" in payload.top_beneficiaries[0]

    def test_to_dict(self) -> None:
        payload = build_basket_payload(_basket())
        d = payload.to_dict()
        assert d["source_concept_id"] == THEME
        assert "beneficiary_count" in d
        assert "top_beneficiaries" in d

    def test_frozen(self) -> None:
        payload = build_basket_payload(_basket())
        try:
            payload.beneficiary_count = 0  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


# -- prepare_structural_publication tests -----------------------------------


class TestPrepareStructuralPublication:
    """Full structural publication pipeline."""

    def test_healthy_publication(self) -> None:
        paths = [_path(target="A"), _path(target="B")]
        baskets = [_basket()]
        result = prepare_structural_publication(
            paths, baskets, _healthy_status(),
        )
        assert result.published is True
        assert len(result.path_explanations) == 2
        assert len(result.basket_payloads) == 1
        assert result.object_count == 3
        assert result.block_reason is None

    def test_blocked_publication(self) -> None:
        result = prepare_structural_publication(
            [_path()], [_basket()], _blocked_status(),
        )
        assert result.published is False
        assert result.block_reason is not None
        assert result.path_explanations == []
        assert result.basket_payloads == []
        assert result.object_count == 0

    def test_empty_inputs(self) -> None:
        result = prepare_structural_publication(
            [], [], _healthy_status(),
        )
        assert result.published is True
        assert result.object_count == 0

    def test_object_count(self) -> None:
        paths = [_path(target="A"), _path(target="B"), _path(target="C")]
        baskets = [_basket(), _basket()]
        result = prepare_structural_publication(
            paths, baskets, _healthy_status(),
        )
        assert result.object_count == 5  # 3 paths + 2 baskets

    def test_warn_status_still_publishes(self) -> None:
        warn_health = LaneHealthStatus(
            lane="structural",
            freshness=FreshnessLevel.AGING,
            quality=QualityLevel.HEALTHY,
            quarantine=QuarantineState.CLEAR,
            readiness=PublishReadiness.WARN,
        )
        result = prepare_structural_publication(
            [_path()], [], warn_health,
        )
        assert result.published is True

    def test_block_reason_includes_details(self) -> None:
        result = prepare_structural_publication(
            [], [], _blocked_status(),
        )
        assert result.block_reason is not None
        assert "stale" in result.block_reason.lower() or "blocked" in result.block_reason.lower()

    def test_path_explanations_have_assertion_lineage(self) -> None:
        paths = [_path()]
        result = prepare_structural_publication(
            paths, [], _healthy_status(),
        )
        assert len(result.path_explanations[0].assertion_ids) > 0

    def test_custom_top_n(self) -> None:
        baskets = [_basket(beneficiaries=10)]
        result = prepare_structural_publication(
            [], baskets, _healthy_status(), top_n=3,
        )
        assert len(result.basket_payloads[0].top_beneficiaries) == 3
