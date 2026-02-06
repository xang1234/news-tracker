"""Tests for event-theme integration (linker and summary)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.event_extraction.theme_integration import EventThemeLinker, ThemeWithEvents


def _make_event(
    event_id: str = "evt-1",
    doc_id: str = "doc-1",
    event_type: str = "capacity_expansion",
    actor: str = "TSMC",
    action: str = "is expanding",
    object: str = "fab capacity",
    time_ref: str = "Q3 2026",
    tickers: list[str] | None = None,
    confidence: float = 0.7,
    created_at: datetime | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "doc_id": doc_id,
        "event_type": event_type,
        "actor": actor,
        "action": action,
        "object": object,
        "time_ref": time_ref,
        "tickers": tickers or ["TSM"],
        "confidence": confidence,
        "created_at": created_at or datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc),
    }


def _make_theme(
    theme_id: str = "theme_abc123",
    top_tickers: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        theme_id=theme_id,
        top_tickers=top_tickers if top_tickers is not None else ["TSM", "NVDA"],
    )


class TestLinkEventsToTheme:
    """Tests for EventThemeLinker.link_events_to_theme()."""

    def test_ticker_overlap_match(self):
        events = [_make_event(tickers=["TSM", "INTC"])]
        theme = _make_theme(top_tickers=["TSM", "NVDA"])

        result = EventThemeLinker.link_events_to_theme(events, theme)

        assert len(result) == 1
        assert result[0]["theme_id"] == "theme_abc123"

    def test_no_ticker_overlap(self):
        events = [_make_event(tickers=["INTC"])]
        theme = _make_theme(top_tickers=["TSM", "NVDA"])

        result = EventThemeLinker.link_events_to_theme(events, theme)

        assert len(result) == 0

    def test_theme_id_added_to_event(self):
        events = [_make_event(tickers=["NVDA"])]
        theme = _make_theme(theme_id="theme_xyz789", top_tickers=["NVDA"])

        result = EventThemeLinker.link_events_to_theme(events, theme)

        assert result[0]["theme_id"] == "theme_xyz789"

    def test_empty_events(self):
        theme = _make_theme()
        result = EventThemeLinker.link_events_to_theme([], theme)
        assert result == []

    def test_empty_theme_tickers(self):
        events = [_make_event()]
        theme = _make_theme(top_tickers=[])

        result = EventThemeLinker.link_events_to_theme(events, theme)
        assert result == []

    def test_none_theme_tickers(self):
        events = [_make_event()]
        theme = SimpleNamespace(theme_id="t1", top_tickers=None)

        result = EventThemeLinker.link_events_to_theme(events, theme)
        assert result == []

    def test_multiple_events_partial_match(self):
        events = [
            _make_event(event_id="e1", tickers=["TSM"]),
            _make_event(event_id="e2", tickers=["INTC"]),
            _make_event(event_id="e3", tickers=["NVDA", "AMD"]),
        ]
        theme = _make_theme(top_tickers=["TSM", "NVDA"])

        result = EventThemeLinker.link_events_to_theme(events, theme)

        assert len(result) == 2
        result_ids = {e["event_id"] for e in result}
        assert result_ids == {"e1", "e3"}

    def test_original_event_not_mutated(self):
        original = _make_event(tickers=["TSM"])
        events = [original]
        theme = _make_theme(top_tickers=["TSM"])

        EventThemeLinker.link_events_to_theme(events, theme)

        assert "theme_id" not in original


class TestDeduplicateEvents:
    """Tests for EventThemeLinker.deduplicate_events()."""

    def test_composite_key_dedup(self):
        events = [
            _make_event(
                event_id="e1", doc_id="d1", actor="TSMC", action="is expanding",
                object="fab capacity", time_ref="Q3 2026",
                created_at=datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc),
            ),
            _make_event(
                event_id="e2", doc_id="d2", actor="TSMC", action="is expanding",
                object="fab capacity", time_ref="Q3 2026",
                created_at=datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        result = EventThemeLinker.deduplicate_events(events)

        assert len(result) == 1

    def test_earliest_kept(self):
        early = datetime(2026, 2, 3, 8, 0, 0, tzinfo=timezone.utc)
        late = datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            _make_event(event_id="e-late", doc_id="d2", created_at=late),
            _make_event(event_id="e-early", doc_id="d1", created_at=early),
        ]

        result = EventThemeLinker.deduplicate_events(events)

        assert len(result) == 1
        assert result[0]["event_id"] == "e-early"

    def test_source_doc_ids_collected(self):
        events = [
            _make_event(event_id="e1", doc_id="d1"),
            _make_event(event_id="e2", doc_id="d2"),
            _make_event(event_id="e3", doc_id="d3"),
        ]

        result = EventThemeLinker.deduplicate_events(events)

        assert len(result) == 1
        assert set(result[0]["source_doc_ids"]) == {"d1", "d2", "d3"}

    def test_confidence_boost(self):
        events = [
            _make_event(event_id="e1", doc_id="d1", confidence=0.7),
            _make_event(event_id="e2", doc_id="d2", confidence=0.7),
        ]

        result = EventThemeLinker.deduplicate_events(events)

        # Base 0.7 + 0.05 for 1 extra source = 0.75
        assert result[0]["confidence"] == pytest.approx(0.75)

    def test_confidence_cap_at_1(self):
        events = [
            _make_event(event_id=f"e{i}", doc_id=f"d{i}", confidence=0.9)
            for i in range(10)
        ]

        result = EventThemeLinker.deduplicate_events(events)

        assert result[0]["confidence"] <= 1.0

    def test_none_fields_in_key(self):
        events = [
            _make_event(event_id="e1", doc_id="d1", actor=None, object=None, time_ref=None),
            _make_event(event_id="e2", doc_id="d2", actor=None, object=None, time_ref=None),
        ]

        result = EventThemeLinker.deduplicate_events(events)

        assert len(result) == 1

    def test_different_keys_not_deduped(self):
        events = [
            _make_event(event_id="e1", actor="TSMC", action="is expanding"),
            _make_event(event_id="e2", actor="Intel", action="is expanding"),
        ]

        result = EventThemeLinker.deduplicate_events(events)

        assert len(result) == 2

    def test_empty_input(self):
        result = EventThemeLinker.deduplicate_events([])
        assert result == []

    def test_case_insensitive_dedup(self):
        events = [
            _make_event(event_id="e1", doc_id="d1", actor="TSMC", action="Is Expanding"),
            _make_event(event_id="e2", doc_id="d2", actor="tsmc", action="is expanding"),
        ]

        result = EventThemeLinker.deduplicate_events(events)

        assert len(result) == 1

    def test_result_sorted_by_created_at_desc(self):
        events = [
            _make_event(
                event_id="e1", doc_id="d1", actor="TSMC",
                created_at=datetime(2026, 2, 3, tzinfo=timezone.utc),
            ),
            _make_event(
                event_id="e2", doc_id="d2", actor="Intel",
                created_at=datetime(2026, 2, 5, tzinfo=timezone.utc),
            ),
        ]

        result = EventThemeLinker.deduplicate_events(events)

        assert result[0]["actor"] == "Intel"
        assert result[1]["actor"] == "TSMC"


class TestThemeWithEvents:
    """Tests for ThemeWithEvents dataclass."""

    def test_event_counts(self):
        events = [
            _make_event(event_type="capacity_expansion"),
            _make_event(event_type="capacity_expansion"),
            _make_event(event_type="product_launch"),
        ]

        summary = ThemeWithEvents.from_events("theme_1", events)

        assert summary.event_counts == {"capacity_expansion": 2, "product_launch": 1}

    def test_investment_signal_supply_increasing(self):
        events = [
            _make_event(event_type="capacity_expansion"),
            _make_event(event_type="capacity_expansion"),
            _make_event(event_type="capacity_constraint"),
        ]

        summary = ThemeWithEvents.from_events("theme_1", events)

        assert summary.investment_signal() == "supply_increasing"

    def test_investment_signal_supply_decreasing(self):
        events = [
            _make_event(event_type="capacity_constraint"),
            _make_event(event_type="capacity_constraint"),
        ]

        summary = ThemeWithEvents.from_events("theme_1", events)

        assert summary.investment_signal() == "supply_decreasing"

    def test_investment_signal_product_momentum(self):
        events = [
            _make_event(event_type="product_launch"),
            _make_event(event_type="product_launch"),
        ]

        summary = ThemeWithEvents.from_events("theme_1", events)

        assert summary.investment_signal() == "product_momentum"

    def test_investment_signal_product_risk(self):
        events = [
            _make_event(event_type="product_delay"),
            _make_event(event_type="product_delay"),
            _make_event(event_type="product_launch"),
        ]

        summary = ThemeWithEvents.from_events("theme_1", events)

        assert summary.investment_signal() == "product_risk"

    def test_investment_signal_balanced(self):
        events = [
            _make_event(event_type="capacity_expansion"),
            _make_event(event_type="capacity_constraint"),
        ]

        summary = ThemeWithEvents.from_events("theme_1", events)

        assert summary.investment_signal() is None

    def test_investment_signal_empty(self):
        summary = ThemeWithEvents.from_events("theme_1", [])

        assert summary.investment_signal() is None

    def test_investment_signal_non_directional_types(self):
        """Events like price_change and guidance_change don't contribute to signals."""
        events = [
            _make_event(event_type="price_change"),
            _make_event(event_type="guidance_change"),
        ]

        summary = ThemeWithEvents.from_events("theme_1", events)

        assert summary.investment_signal() is None

    def test_from_events_preserves_events(self):
        events = [_make_event(), _make_event(event_id="e2")]

        summary = ThemeWithEvents.from_events("theme_1", events)

        assert len(summary.recent_events) == 2
        assert summary.theme_id == "theme_1"
