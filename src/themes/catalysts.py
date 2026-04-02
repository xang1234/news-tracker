"""Helpers for converting live narratives into tradable market catalysts."""

from __future__ import annotations

from typing import Iterable

_SIGNAL_TO_BIAS = {
    "supply_increasing": "bullish",
    "product_momentum": "bullish",
    "supply_decreasing": "bearish",
    "product_risk": "bearish",
}

_SIGNAL_LABELS = {
    "supply_increasing": "expanding supply",
    "supply_decreasing": "tightening supply",
    "product_momentum": "product momentum",
    "product_risk": "product risk",
}

_EVENT_LABELS = {
    "capacity_constraint": "capacity constraint",
    "capacity_expansion": "capacity expansion",
    "guidance_change": "guidance change",
    "price_change": "price change",
    "product_delay": "product delay",
    "product_launch": "product launch",
}


def humanize_identifier(value: str) -> str:
    """Convert underscored identifiers into title-case labels."""
    cleaned = (value or "").strip()
    if not cleaned:
        return "Unknown"
    if "_" not in cleaned:
        return cleaned
    return cleaned.replace("_", " ").strip().title()


def dominant_event_types(event_counts: dict[str, int], limit: int = 3) -> list[str]:
    """Return the most corroborated event types for a catalyst."""
    ordered = sorted(
        event_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [event_type for event_type, _ in ordered[:limit]]


def infer_market_bias(
    avg_sentiment: float | None,
    investment_signal: str | None,
) -> str:
    """Combine sentiment and event signals into a directional market bias."""
    signal_bias = _SIGNAL_TO_BIAS.get(investment_signal or "")

    sentiment_bias: str | None = None
    sentiment = avg_sentiment or 0.0
    if sentiment >= 0.08:
        sentiment_bias = "bullish"
    elif sentiment <= -0.08:
        sentiment_bias = "bearish"

    if signal_bias and sentiment_bias and signal_bias != sentiment_bias:
        return "mixed"
    if signal_bias:
        return signal_bias
    if sentiment_bias:
        return sentiment_bias
    return "mixed"


def propagation_delta(avg_sentiment: float | None, bias: str) -> float:
    """Derive a bounded propagation delta from narrative sentiment and bias."""
    magnitude = max(abs(avg_sentiment or 0.0), 0.15)
    magnitude = min(magnitude, 0.35)

    if bias == "bullish":
        return magnitude
    if bias == "bearish":
        return -magnitude
    if (avg_sentiment or 0.0) > 0:
        return magnitude
    if (avg_sentiment or 0.0) < 0:
        return -magnitude
    return 0.0


def compute_market_impact_score(
    conviction_score: float,
    volume_zscore: float | None,
    acceleration: float,
    platform_count: int,
    avg_authority: float | None,
    event_count: int,
) -> float:
    """Blend corroboration and urgency into a 0-100 impact score."""
    positive_volume = max(volume_zscore or 0.0, 0.0)
    positive_acceleration = max(acceleration, 0.0)

    score = 0.0
    score += min(max(conviction_score, 0.0) * 0.55, 55.0)
    score += min(positive_volume * 12.0, 18.0)
    score += min(positive_acceleration * 4.0, 12.0)
    score += min(platform_count * 3.0, 9.0)
    score += min(max(avg_authority or 0.0, 0.0) * 15.0, 12.0)
    score += min(max(event_count, 0) * 2.0, 8.0)
    return round(min(score, 100.0), 2)


def event_label(event_type: str) -> str:
    """Convert an event type into a compact human label."""
    return _EVENT_LABELS.get(event_type, humanize_identifier(event_type).lower())


def signal_label(signal: str | None) -> str | None:
    """Convert an investment signal into plain language."""
    if signal is None:
        return None
    return _SIGNAL_LABELS.get(signal, humanize_identifier(signal).lower())


def summarize_market_catalyst(
    *,
    theme_name: str,
    bias: str,
    primary_tickers: Iterable[str],
    investment_signal: str | None,
    dominant_events: list[str],
    platform_count: int,
    volume_zscore: float | None,
    conviction_score: float,
    related_tickers: Iterable[str] = (),
) -> str:
    """Render a concise stock-market-oriented catalyst summary."""
    lead = {
        "bullish": "Bullish setup",
        "bearish": "Bearish setup",
        "mixed": "Mixed setup",
    }.get(bias, "Live setup")

    ticker_list = list(primary_tickers)
    ticker_text = ", ".join(f"${ticker}" for ticker in ticker_list[:3]) or "linked equities"
    signal_text = signal_label(investment_signal)
    event_text = ", ".join(event_label(evt) for evt in dominant_events[:2]) or "cross-platform narrative momentum"

    parts = [
        f"{lead} for {ticker_text}.",
        f"{theme_name} is spreading across {platform_count} platforms",
    ]
    if signal_text:
        parts[-1] += f" with {signal_text}"
    else:
        parts[-1] += f" around {event_text}"

    if volume_zscore is not None:
        parts[-1] += f", backed by volume z-score {volume_zscore:.1f}"

    parts[-1] += f" and conviction {round(conviction_score)}."

    related = list(related_tickers)
    if related:
        parts.append(
            "Watch follow-through in "
            + ", ".join(f"${ticker}" for ticker in related[:2])
            + "."
        )

    return " ".join(parts)
