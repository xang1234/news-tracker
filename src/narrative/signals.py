"""Pure narrative signal evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any

from src.narrative.config import NarrativeConfig
from src.narrative.schemas import NarrativeRun, NarrativeRunBucket


@dataclass
class SignalEvaluation:
    """Evaluation result for a single narrative trigger."""

    trigger_type: str
    metric_value: float
    deactivate_below: float
    triggered: bool
    severity: str | None = None
    conviction_score: float | None = None
    title: str | None = None
    message: str | None = None
    trigger_data: dict[str, Any] = field(default_factory=dict)


def _bucket_rate_per_hour(bucket: NarrativeRunBucket, bucket_minutes: int) -> float:
    return float(bucket.doc_count) * (60.0 / float(bucket_minutes))


def _weighted_average(total: float, weight: float) -> float:
    if weight <= 0:
        return 0.0
    return total / weight


def _avg_confidence_support(run: NarrativeRun) -> float:
    support = min(1.0, run.doc_count / 20.0)
    platform = min(1.0, run.platform_count / 4.0)
    authority = min(1.0, max(run.avg_authority, 0.0))
    return (support * 0.5) + (platform * 0.25) + (authority * 0.25)


def _conviction_from_components(
    support_component: float,
    magnitude_component: float,
    breadth_component: float,
    authority_component: float,
) -> float:
    score = (
        max(0.0, min(support_component, 1.0)) * 30.0
        + max(0.0, min(magnitude_component, 1.0)) * 35.0
        + max(0.0, min(breadth_component, 1.0)) * 20.0
        + max(0.0, min(authority_component, 1.0)) * 15.0
    )
    return round(min(score, 100.0), 2)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def evaluate_narrative_surge(
    run: NarrativeRun,
    buckets: list[NarrativeRunBucket],
    config: NarrativeConfig,
) -> SignalEvaluation:
    recent_cutoff = _now() - timedelta(minutes=config.surge_window_minutes)
    baseline_cutoff = _now() - timedelta(hours=config.surge_baseline_hours)

    recent_buckets = [b for b in buckets if b.bucket_start >= recent_cutoff]
    baseline_buckets = [
        b for b in buckets
        if baseline_cutoff <= b.bucket_start < recent_cutoff
    ]
    recent_docs = sum(b.doc_count for b in recent_buckets)
    recent_rate = (
        (recent_docs / max(config.surge_window_minutes, 1)) * 60.0
        if recent_buckets
        else 0.0
    )

    baseline_rates = [
        _bucket_rate_per_hour(b, config.bucket_minutes) for b in baseline_buckets
    ]
    baseline_mean = mean(baseline_rates) if baseline_rates else 0.0
    baseline_std = pstdev(baseline_rates) if len(baseline_rates) > 1 else 0.0
    uplift = recent_rate / baseline_mean if baseline_mean > 0 else 0.0
    z_score = (
        (recent_rate - baseline_mean) / baseline_std
        if baseline_std > 0
        else 0.0
    )

    triggered = (
        run.doc_count >= config.surge_min_total_docs
        and recent_rate >= config.surge_min_rate_per_hour
        and (z_score >= config.surge_trigger_zscore or uplift >= config.surge_trigger_uplift)
    )

    if not triggered:
        return SignalEvaluation(
            trigger_type="narrative_surge",
            metric_value=z_score,
            deactivate_below=config.surge_reset_zscore,
            triggered=False,
            trigger_data={
                "recent_rate_per_hour": round(recent_rate, 3),
                "baseline_rate_per_hour": round(baseline_mean, 3),
                "z_score": round(z_score, 3),
                "uplift": round(uplift, 3),
            },
        )

    magnitude = max(z_score / 5.0, uplift / 4.0)
    conviction = _conviction_from_components(
        support_component=min(run.doc_count / 20.0, 1.0),
        magnitude_component=min(magnitude, 1.0),
        breadth_component=min(run.platform_count / 4.0, 1.0),
        authority_component=_avg_confidence_support(run),
    )
    severity = "critical" if z_score >= 5.0 or uplift >= 4.0 else "warning"
    tickers = list(run.ticker_counts)[:3]
    title = f"Narrative surge: {run.label}"
    message = (
        f"{run.label} is running at {recent_rate:.1f}/hr "
        f"({z_score:.1f}σ, {uplift:.1f}x baseline)"
    )
    return SignalEvaluation(
        trigger_type="narrative_surge",
        metric_value=max(z_score, uplift),
        deactivate_below=config.surge_reset_zscore,
        triggered=True,
        severity=severity,
        conviction_score=conviction,
        title=title,
        message=message,
        trigger_data={
            "recent_rate_per_hour": round(recent_rate, 3),
            "baseline_rate_per_hour": round(baseline_mean, 3),
            "z_score": round(z_score, 3),
            "uplift": round(uplift, 3),
            "platform_count": run.platform_count,
            "top_tickers": tickers,
        },
    )


def evaluate_cross_platform_breakout(
    run: NarrativeRun,
    buckets: list[NarrativeRunBucket],
    config: NarrativeConfig,
) -> SignalEvaluation:
    first_seen_values = sorted(run.platform_first_seen.values())
    if len(first_seen_values) < 3 or run.doc_count < config.cross_platform_min_docs:
        return SignalEvaluation(
            trigger_type="cross_platform_breakout",
            metric_value=float(run.platform_count),
            deactivate_below=2.0,
            triggered=False,
            trigger_data={"platform_count": run.platform_count},
        )

    timestamps = [datetime.fromisoformat(v) for v in first_seen_values[:3]]
    spread_hours = (timestamps[2] - timestamps[0]).total_seconds() / 3600.0
    triggered = spread_hours <= config.cross_platform_window_hours

    if not triggered:
        return SignalEvaluation(
            trigger_type="cross_platform_breakout",
            metric_value=float(run.platform_count),
            deactivate_below=2.0,
            triggered=False,
            trigger_data={
                "platform_count": run.platform_count,
                "spread_hours": round(spread_hours, 3),
            },
        )

    conviction = _conviction_from_components(
        support_component=min(run.doc_count / 15.0, 1.0),
        magnitude_component=min(run.platform_count / 4.0, 1.0),
        breadth_component=min(run.platform_count / 4.0, 1.0),
        authority_component=_avg_confidence_support(run),
    )
    recent_cutoff = _now() - timedelta(hours=2)
    recent = [b for b in buckets if b.bucket_start >= recent_cutoff]
    recent_docs = sum(b.doc_count for b in recent)
    recent_avg_authority = (
        sum(b.authority_sum for b in recent) / recent_docs
        if recent_docs > 0
        else 0.0
    )
    severity = "critical" if run.platform_count >= 4 or recent_avg_authority >= config.high_authority_threshold else "warning"
    return SignalEvaluation(
        trigger_type="cross_platform_breakout",
        metric_value=float(run.platform_count),
        deactivate_below=2.0,
        triggered=True,
        severity=severity,
        conviction_score=conviction,
        title=f"Cross-platform breakout: {run.label}",
        message=(
            f"{run.label} spread across {run.platform_count} platforms in "
            f"{spread_hours:.1f}h"
        ),
        trigger_data={
            "platform_count": run.platform_count,
            "spread_hours": round(spread_hours, 3),
            "recent_avg_authority": round(recent_avg_authority, 3),
            "platform_first_seen": run.platform_first_seen,
        },
    )


def evaluate_authority_divergence(
    run: NarrativeRun,
    buckets: list[NarrativeRunBucket],
    config: NarrativeConfig,
) -> SignalEvaluation:
    cutoff = _now() - timedelta(hours=config.authority_window_hours)
    recent = [b for b in buckets if b.bucket_start >= cutoff]
    high_weight = sum(b.high_authority_weight for b in recent)
    low_weight = sum(b.low_authority_weight for b in recent)
    high_docs = sum(b.high_authority_doc_count for b in recent)
    low_docs = sum(b.low_authority_doc_count for b in recent)
    if high_docs < 3 or low_docs < 3:
        return SignalEvaluation(
            trigger_type="authority_divergence",
            metric_value=0.0,
            deactivate_below=config.authority_gap_reset,
            triggered=False,
        )

    high_avg = _weighted_average(
        sum(b.high_authority_sentiment_sum for b in recent), high_weight
    )
    low_avg = _weighted_average(
        sum(b.low_authority_sentiment_sum for b in recent), low_weight
    )
    gap = abs(high_avg - low_avg)
    triggered = gap >= config.authority_gap_trigger
    if not triggered:
        return SignalEvaluation(
            trigger_type="authority_divergence",
            metric_value=gap,
            deactivate_below=config.authority_gap_reset,
            triggered=False,
            trigger_data={
                "high_authority_avg": round(high_avg, 3),
                "low_authority_avg": round(low_avg, 3),
                "gap": round(gap, 3),
            },
        )

    conviction = _conviction_from_components(
        support_component=min((high_weight + low_weight) / 12.0, 1.0),
        magnitude_component=min(gap / 0.8, 1.0),
        breadth_component=min(run.platform_count / 3.0, 1.0),
        authority_component=min(run.avg_authority + 0.2, 1.0),
    )
    severity = "critical" if gap >= 0.7 else "warning"
    return SignalEvaluation(
        trigger_type="authority_divergence",
        metric_value=gap,
        deactivate_below=config.authority_gap_reset,
        triggered=True,
        severity=severity,
        conviction_score=conviction,
        title=f"Authority divergence: {run.label}",
        message=(
            f"High-authority sentiment diverged from crowd sentiment "
            f"by {gap:.2f}"
        ),
        trigger_data={
            "high_authority_avg": round(high_avg, 3),
            "low_authority_avg": round(low_avg, 3),
            "gap": round(gap, 3),
            "high_authority_docs": high_docs,
            "low_authority_docs": low_docs,
        },
    )


def evaluate_sentiment_regime_shift(
    run: NarrativeRun,
    buckets: list[NarrativeRunBucket],
    config: NarrativeConfig,
) -> SignalEvaluation:
    now = _now()
    recent_cutoff = now - timedelta(hours=config.sentiment_recent_hours)
    prior_cutoff = now - timedelta(
        hours=config.sentiment_recent_hours + config.sentiment_prior_hours
    )
    recent = [b for b in buckets if b.bucket_start >= recent_cutoff]
    prior = [b for b in buckets if prior_cutoff <= b.bucket_start < recent_cutoff]
    recent_docs = sum(b.doc_count for b in recent)
    prior_docs = sum(b.doc_count for b in prior)
    if recent_docs < config.sentiment_min_docs or prior_docs < config.sentiment_min_docs:
        return SignalEvaluation(
            trigger_type="sentiment_regime_shift",
            metric_value=0.0,
            deactivate_below=config.sentiment_shift_trigger / 2.0,
            triggered=False,
        )

    recent_sentiment = _weighted_average(
        sum(b.sentiment_sum for b in recent), sum(b.sentiment_weight for b in recent)
    )
    prior_sentiment = _weighted_average(
        sum(b.sentiment_sum for b in prior), sum(b.sentiment_weight for b in prior)
    )
    recent_confidence = _weighted_average(
        sum(b.sentiment_confidence_sum for b in recent),
        sum(b.sentiment_doc_count for b in recent),
    )
    delta = recent_sentiment - prior_sentiment
    crossed_zero = (recent_sentiment >= 0 > prior_sentiment) or (
        recent_sentiment <= 0 < prior_sentiment
    )
    triggered = (
        crossed_zero
        and abs(delta) >= config.sentiment_shift_trigger
        and recent_confidence >= config.sentiment_shift_confidence
    )
    if not triggered:
        return SignalEvaluation(
            trigger_type="sentiment_regime_shift",
            metric_value=abs(delta),
            deactivate_below=config.sentiment_shift_trigger / 2.0,
            triggered=False,
            trigger_data={
                "recent_sentiment": round(recent_sentiment, 3),
                "prior_sentiment": round(prior_sentiment, 3),
                "delta": round(delta, 3),
                "recent_confidence": round(recent_confidence, 3),
            },
        )

    conviction = _conviction_from_components(
        support_component=min((recent_docs + prior_docs) / 20.0, 1.0),
        magnitude_component=min(abs(delta) / 0.6, 1.0),
        breadth_component=min(run.platform_count / 3.0, 1.0),
        authority_component=_avg_confidence_support(run),
    )
    severity = "critical" if abs(delta) >= 0.55 else "warning"
    direction = "bullish" if recent_sentiment > prior_sentiment else "bearish"
    return SignalEvaluation(
        trigger_type="sentiment_regime_shift",
        metric_value=abs(delta),
        deactivate_below=config.sentiment_shift_trigger / 2.0,
        triggered=True,
        severity=severity,
        conviction_score=conviction,
        title=f"Sentiment regime shift: {run.label}",
        message=(
            f"{run.label} flipped {direction} "
            f"(Δ={delta:+.2f} vs prior window)"
        ),
        trigger_data={
            "recent_sentiment": round(recent_sentiment, 3),
            "prior_sentiment": round(prior_sentiment, 3),
            "delta": round(delta, 3),
            "direction": direction,
            "recent_confidence": round(recent_confidence, 3),
        },
    )


def evaluate_all_signals(
    run: NarrativeRun,
    buckets: list[NarrativeRunBucket],
    config: NarrativeConfig,
) -> list[SignalEvaluation]:
    return [
        evaluate_narrative_surge(run, buckets, config),
        evaluate_cross_platform_breakout(run, buckets, config),
        evaluate_authority_divergence(run, buckets, config),
        evaluate_sentiment_regime_shift(run, buckets, config),
    ]
