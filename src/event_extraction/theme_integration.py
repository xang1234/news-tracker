"""Event-theme integration for linking extracted events to themes.

Provides stateless linker (pure functions) and summary dataclass for
theme-level event aggregation. Links events to themes via ticker overlap
and deduplicates cross-document events using composite keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class EventThemeLinker:
    """Stateless linker for associating events with themes via ticker overlap.

    All methods are static — no instance state, no DB writes. Designed for
    easy testing and composability in API endpoints.
    """

    @staticmethod
    def link_events_to_theme(
        events: list[dict[str, Any]],
        theme: Any,
    ) -> list[dict[str, Any]]:
        """Filter events that share tickers with a theme and annotate them.

        Args:
            events: Raw event dicts from the repository.
            theme: Theme object with ``top_tickers`` attribute.

        Returns:
            Events whose tickers overlap with the theme's top_tickers,
            each augmented with a ``theme_id`` key.
        """
        theme_tickers = set(getattr(theme, "top_tickers", []) or [])
        if not theme_tickers:
            return []

        linked: list[dict[str, Any]] = []
        for event in events:
            event_tickers = set(event.get("tickers", []) or [])
            if event_tickers & theme_tickers:
                enriched = dict(event)
                enriched["theme_id"] = theme.theme_id
                linked.append(enriched)

        return linked

    @staticmethod
    def deduplicate_events(
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deduplicate events using a composite key, keeping the earliest.

        Composite key: ``(actor, action, object, time_ref)``
        — lowercased and stripped for robustness.

        For each group of duplicates:
        - The earliest ``created_at`` event is kept.
        - ``source_doc_ids`` collects all contributing document IDs.
        - Confidence is boosted +0.05 per additional source, capped at 1.0.

        Args:
            events: Event dicts (may contain duplicates).

        Returns:
            Deduplicated events sorted by created_at descending.
        """
        if not events:
            return []

        groups: dict[tuple, dict[str, Any]] = {}

        for event in events:
            key = (
                (event.get("actor") or "").strip().lower(),
                (event.get("action") or "").strip().lower(),
                (event.get("object") or "").strip().lower(),
                (event.get("time_ref") or "").strip().lower(),
            )

            if key not in groups:
                deduped = dict(event)
                deduped["source_doc_ids"] = [event["doc_id"]]
                groups[key] = deduped
            else:
                existing = groups[key]
                # Keep earliest created_at
                if event.get("created_at") and existing.get("created_at"):
                    if event["created_at"] < existing["created_at"]:
                        doc_ids = existing["source_doc_ids"]
                        new_event = dict(event)
                        new_event["source_doc_ids"] = doc_ids
                        groups[key] = new_event

                # Track source document
                doc_id = event["doc_id"]
                if doc_id not in groups[key]["source_doc_ids"]:
                    groups[key]["source_doc_ids"].append(doc_id)

                # Boost confidence per additional source
                extra_sources = len(groups[key]["source_doc_ids"]) - 1
                base_confidence = event.get("confidence", 0.7)
                groups[key]["confidence"] = min(
                    1.0, base_confidence + 0.05 * extra_sources
                )

        result = list(groups.values())
        result.sort(
            key=lambda e: e.get("created_at") or "",
            reverse=True,
        )
        return result


# Investment signal mapping: event_type → directional signal
_SUPPLY_INCREASING = {"capacity_expansion"}
_SUPPLY_DECREASING = {"capacity_constraint"}
_PRODUCT_MOMENTUM = {"product_launch"}
_PRODUCT_RISK = {"product_delay"}


@dataclass
class ThemeWithEvents:
    """Summary of events linked to a theme.

    Aggregates event counts by type and derives an investment signal
    from the dominant event category.
    """

    theme_id: str
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    event_counts: dict[str, int] = field(default_factory=dict)

    def investment_signal(self) -> str | None:
        """Derive a directional investment signal from event distribution.

        Compares supply-side and product-side event counts to determine
        the dominant narrative:
        - ``supply_increasing``: More capacity expansions than constraints.
        - ``supply_decreasing``: More capacity constraints than expansions.
        - ``product_momentum``: More product launches than delays.
        - ``product_risk``: More product delays than launches.
        - ``None``: Balanced or no meaningful signal.

        Returns:
            Signal string or None.
        """
        supply_up = sum(
            self.event_counts.get(t, 0) for t in _SUPPLY_INCREASING
        )
        supply_down = sum(
            self.event_counts.get(t, 0) for t in _SUPPLY_DECREASING
        )
        product_up = sum(
            self.event_counts.get(t, 0) for t in _PRODUCT_MOMENTUM
        )
        product_down = sum(
            self.event_counts.get(t, 0) for t in _PRODUCT_RISK
        )

        # Determine dominant axis
        supply_delta = supply_up - supply_down
        product_delta = product_up - product_down

        if supply_delta == 0 and product_delta == 0:
            return None

        if abs(supply_delta) >= abs(product_delta):
            if supply_delta > 0:
                return "supply_increasing"
            elif supply_delta < 0:
                return "supply_decreasing"
        else:
            if product_delta > 0:
                return "product_momentum"
            elif product_delta < 0:
                return "product_risk"

        return None

    @classmethod
    def from_events(
        cls,
        theme_id: str,
        events: list[dict[str, Any]],
    ) -> "ThemeWithEvents":
        """Build a ThemeWithEvents from a list of event dicts.

        Args:
            theme_id: Theme identifier.
            events: Linked and deduplicated event dicts.

        Returns:
            ThemeWithEvents with computed event_counts.
        """
        counts: dict[str, int] = {}
        for event in events:
            et = event.get("event_type", "unknown")
            counts[et] = counts.get(et, 0) + 1

        return cls(
            theme_id=theme_id,
            recent_events=events,
            event_counts=counts,
        )
