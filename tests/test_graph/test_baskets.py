"""Tests for thematic baskets and second-order beneficiary outputs.

Verifies that scored paths are assembled into decision-oriented
baskets with correct role classification, first/second-order counts,
mixed-signal detection, and path provenance.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.graph.baskets import (
    ROLE_AT_RISK,
    ROLE_BENEFICIARY,
    BasketMember,
    ThematicBasket,
    build_thematic_basket,
)
from src.graph.path_scoring import PathScoreBreakdown, ScoredPath

NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)
THEME = "concept_theme_hbm"


# -- Helpers ---------------------------------------------------------------


def _path(
    source: str = THEME,
    target: str = "concept_issuer_aaa",
    hops: int = 1,
    path_score: float = 0.5,
    path_sign: int = 1,
    intermediate: str | None = None,
) -> ScoredPath:
    return ScoredPath(
        source_concept_id=source,
        target_concept_id=target,
        hops=hops,
        path_score=path_score,
        path_sign=path_sign,
        breakdown=PathScoreBreakdown(
            confidence_product=0.8,
            freshness_product=0.9,
            corroboration_product=0.7,
            hop_decay=0.7 if hops == 2 else 1.0,
        ),
        intermediate_concept_id=intermediate,
    )


# -- BasketMember tests ----------------------------------------------------


class TestBasketMember:
    """Basket member construction and properties."""

    def test_beneficiary_role(self) -> None:
        basket = build_thematic_basket(THEME, [_path(path_sign=1)], now=NOW)
        assert len(basket.beneficiaries) == 1
        assert basket.beneficiaries[0].role == ROLE_BENEFICIARY

    def test_at_risk_role(self) -> None:
        basket = build_thematic_basket(THEME, [_path(path_sign=-1)], now=NOW)
        assert len(basket.at_risk) == 1
        assert basket.at_risk[0].role == ROLE_AT_RISK

    def test_is_second_order(self) -> None:
        basket = build_thematic_basket(THEME, [
            _path(target="A", hops=2, intermediate="B"),
        ], now=NOW)
        assert basket.beneficiaries[0].is_second_order is True

    def test_is_first_order(self) -> None:
        basket = build_thematic_basket(THEME, [_path(hops=1)], now=NOW)
        assert basket.beneficiaries[0].is_second_order is False

    def test_mixed_signals(self) -> None:
        """Same target via positive and negative paths."""
        paths = [
            _path(target="A", path_score=0.8, path_sign=1),
            _path(target="A", path_score=0.3, path_sign=-1),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        member = basket.beneficiaries[0]  # best path is positive
        assert member.has_mixed_signals is True
        assert member.positive_paths == 1
        assert member.negative_paths == 1

    def test_no_mixed_signals(self) -> None:
        basket = build_thematic_basket(THEME, [_path(target="A")], now=NOW)
        assert basket.beneficiaries[0].has_mixed_signals is False

    def test_best_score_from_highest_path(self) -> None:
        paths = [
            _path(target="A", path_score=0.3),
            _path(target="A", path_score=0.8),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert basket.beneficiaries[0].best_score == 0.8

    def test_min_hops_from_closest_path(self) -> None:
        paths = [
            _path(target="A", hops=2, path_score=0.8, intermediate="B"),
            _path(target="A", hops=1, path_score=0.3),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert basket.beneficiaries[0].min_hops == 1

    def test_path_count(self) -> None:
        paths = [
            _path(target="A", path_score=0.8),
            _path(target="A", path_score=0.5),
            _path(target="A", path_score=0.3),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert basket.beneficiaries[0].path_count == 3

    def test_paths_sorted_by_score(self) -> None:
        paths = [
            _path(target="A", path_score=0.3),
            _path(target="A", path_score=0.8),
            _path(target="A", path_score=0.5),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        scores = [p.path_score for p in basket.beneficiaries[0].paths]
        assert scores == sorted(scores, reverse=True)

    def test_to_dict(self) -> None:
        basket = build_thematic_basket(THEME, [_path()], now=NOW)
        d = basket.beneficiaries[0].to_dict()
        assert d["concept_id"] == "concept_issuer_aaa"
        assert d["role"] == ROLE_BENEFICIARY
        assert "best_score" in d
        assert "is_second_order" in d
        assert "has_mixed_signals" in d


# -- ThematicBasket tests --------------------------------------------------


class TestThematicBasket:
    """Basket assembly from scored paths."""

    def test_basic_basket(self) -> None:
        paths = [
            _path(target="A", path_score=0.8, path_sign=1),
            _path(target="B", path_score=0.3, path_sign=-1),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert basket.source_concept_id == THEME
        assert len(basket.beneficiaries) == 1
        assert len(basket.at_risk) == 1
        assert basket.member_count == 2

    def test_empty_paths(self) -> None:
        basket = build_thematic_basket(THEME, [], now=NOW)
        assert basket.beneficiaries == []
        assert basket.at_risk == []
        assert basket.member_count == 0
        assert basket.first_order_count == 0
        assert basket.second_order_count == 0

    def test_all_beneficiaries(self) -> None:
        paths = [
            _path(target="A", path_score=0.8),
            _path(target="B", path_score=0.5),
            _path(target="C", path_score=0.3),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert len(basket.beneficiaries) == 3
        assert len(basket.at_risk) == 0

    def test_all_at_risk(self) -> None:
        paths = [
            _path(target="A", path_sign=-1),
            _path(target="B", path_sign=-1),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert len(basket.beneficiaries) == 0
        assert len(basket.at_risk) == 2

    def test_first_second_order_counts(self) -> None:
        paths = [
            _path(target="A", hops=1),
            _path(target="B", hops=1),
            _path(target="C", hops=2, intermediate="A"),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert basket.first_order_count == 2
        assert basket.second_order_count == 1

    def test_mixed_order_same_target(self) -> None:
        """Target reachable via both 1-hop and 2-hop → first order."""
        paths = [
            _path(target="A", hops=1, path_score=0.8),
            _path(target="A", hops=2, path_score=0.3, intermediate="B"),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert basket.first_order_count == 1
        assert basket.second_order_count == 0
        assert basket.beneficiaries[0].min_hops == 1

    def test_beneficiaries_sorted_by_score(self) -> None:
        paths = [
            _path(target="A", path_score=0.3),
            _path(target="B", path_score=0.8),
            _path(target="C", path_score=0.5),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        scores = [m.best_score for m in basket.beneficiaries]
        assert scores == sorted(scores, reverse=True)

    def test_at_risk_sorted_by_score(self) -> None:
        paths = [
            _path(target="A", path_score=0.2, path_sign=-1),
            _path(target="B", path_score=0.6, path_sign=-1),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert basket.at_risk[0].best_score > basket.at_risk[1].best_score

    def test_mixed_sign_classified_by_best(self) -> None:
        """If best path is negative, member goes to at_risk despite positive paths."""
        paths = [
            _path(target="A", path_score=0.8, path_sign=-1),
            _path(target="A", path_score=0.3, path_sign=1),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        assert len(basket.at_risk) == 1
        assert len(basket.beneficiaries) == 0
        assert basket.at_risk[0].has_mixed_signals is True

    def test_computed_at(self) -> None:
        basket = build_thematic_basket(THEME, [], now=NOW)
        assert basket.computed_at == NOW

    def test_to_dict(self) -> None:
        paths = [_path(target="A"), _path(target="B", path_sign=-1)]
        basket = build_thematic_basket(THEME, paths, now=NOW)
        d = basket.to_dict()
        assert d["source_concept_id"] == THEME
        assert d["beneficiary_count"] == 1
        assert d["at_risk_count"] == 1
        assert d["member_count"] == 2
        assert isinstance(d["computed_at"], str)


# -- Integration: realistic scenario ----------------------------------------


class TestRealisticScenario:
    """HBM theme basket with supply chain and competition."""

    def test_hbm_supply_chain(self) -> None:
        """
        HBM Surge theme:
        - 1-hop beneficiaries: SK Hynix (supplier), Samsung (supplier)
        - 2-hop beneficiary: ASML (equipment → SK Hynix)
        - 1-hop at-risk: DRAM (competitor)
        """
        paths = [
            _path(target="sk_hynix", hops=1, path_score=0.72, path_sign=1),
            _path(target="samsung", hops=1, path_score=0.65, path_sign=1),
            _path(target="asml", hops=2, path_score=0.35, path_sign=1,
                  intermediate="sk_hynix"),
            _path(target="dram_vendors", hops=1, path_score=0.45, path_sign=-1),
        ]
        basket = build_thematic_basket(THEME, paths, now=NOW)

        assert basket.member_count == 4
        assert basket.first_order_count == 3
        assert basket.second_order_count == 1

        assert len(basket.beneficiaries) == 3
        assert basket.beneficiaries[0].concept_id == "sk_hynix"
        assert basket.beneficiaries[2].concept_id == "asml"
        assert basket.beneficiaries[2].is_second_order is True

        assert len(basket.at_risk) == 1
        assert basket.at_risk[0].concept_id == "dram_vendors"


# -- Dataclass tests -------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_basket_member_frozen(self) -> None:
        basket = build_thematic_basket(THEME, [_path()], now=NOW)
        try:
            basket.beneficiaries[0].best_score = 0.0  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_basket_frozen(self) -> None:
        basket = build_thematic_basket(THEME, [], now=NOW)
        try:
            basket.member_count = 99  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass
