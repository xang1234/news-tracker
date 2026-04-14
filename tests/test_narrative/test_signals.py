"""Unit tests for narrative momentum signal evaluation."""

from datetime import UTC, datetime, timedelta

import numpy as np

from src.narrative.config import NarrativeConfig
from src.narrative.schemas import NarrativeRun, NarrativeRunBucket
from src.narrative.signals import (
    evaluate_all_signals,
    evaluate_authority_divergence,
    evaluate_cross_platform_breakout,
    evaluate_narrative_surge,
    evaluate_sentiment_regime_shift,
)


def _make_run(**overrides) -> NarrativeRun:
    now = datetime(2026, 2, 5, 12, 0, 0, tzinfo=UTC)
    data = {
        "run_id": "run_123",
        "theme_id": "theme_123",
        "status": "active",
        "centroid": np.ones(768, dtype=np.float32),
        "label": "NVDA / AMD",
        "started_at": now - timedelta(hours=4),
        "last_document_at": now,
        "doc_count": 20,
        "platform_first_seen": {
            "news": (now - timedelta(hours=4)).isoformat(),
            "twitter": (now - timedelta(hours=3, minutes=30)).isoformat(),
            "reddit": (now - timedelta(hours=3)).isoformat(),
        },
        "ticker_counts": {"NVDA": 14, "AMD": 6},
        "avg_sentiment": 0.2,
        "avg_authority": 0.75,
        "platform_count": 3,
        "current_rate_per_hour": 8.0,
        "current_acceleration": 2.0,
        "conviction_score": 65.0,
        "metadata": {},
        "created_at": now - timedelta(hours=4),
        "updated_at": now,
    }
    data.update(overrides)
    return NarrativeRun(**data)


def _bucket(
    bucket_start: datetime,
    *,
    doc_count: int,
    sentiment_sum: float = 0.0,
    sentiment_weight: float = 0.0,
    sentiment_confidence_sum: float = 0.0,
    sentiment_doc_count: int = 0,
    authority_sum: float = 0.0,
    high_authority_sentiment_sum: float = 0.0,
    high_authority_weight: float = 0.0,
    high_authority_doc_count: int = 0,
    low_authority_sentiment_sum: float = 0.0,
    low_authority_weight: float = 0.0,
    low_authority_doc_count: int = 0,
) -> NarrativeRunBucket:
    return NarrativeRunBucket(
        run_id="run_123",
        bucket_start=bucket_start,
        doc_count=doc_count,
        sentiment_sum=sentiment_sum,
        sentiment_weight=sentiment_weight,
        sentiment_confidence_sum=sentiment_confidence_sum,
        sentiment_doc_count=sentiment_doc_count,
        authority_sum=authority_sum,
        high_authority_sentiment_sum=high_authority_sentiment_sum,
        high_authority_weight=high_authority_weight,
        high_authority_doc_count=high_authority_doc_count,
        low_authority_sentiment_sum=low_authority_sentiment_sum,
        low_authority_weight=low_authority_weight,
        low_authority_doc_count=low_authority_doc_count,
    )


def test_narrative_surge_triggers_on_rate_uplift(monkeypatch):
    now = datetime(2026, 2, 5, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("src.narrative.signals._now", lambda: now)
    config = NarrativeConfig()
    run = _make_run(doc_count=24)

    baseline = [
        _bucket(now - timedelta(hours=1, minutes=step * 5), doc_count=1)
        for step in range(12, 0, -1)
    ]
    recent = [_bucket(now - timedelta(minutes=step * 5), doc_count=3) for step in range(6, 0, -1)]

    result = evaluate_narrative_surge(run, baseline + recent, config)

    assert result.triggered is True
    assert result.trigger_type == "narrative_surge"
    assert result.conviction_score is not None


def test_cross_platform_breakout_escalates_with_recent_authority(monkeypatch):
    now = datetime(2026, 2, 5, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("src.narrative.signals._now", lambda: now)
    config = NarrativeConfig()
    run = _make_run()
    buckets = [
        _bucket(
            now - timedelta(minutes=30),
            doc_count=3,
            authority_sum=0.9,
            high_authority_doc_count=1,
        )
    ]

    result = evaluate_cross_platform_breakout(run, buckets, config)

    assert result.triggered is True
    assert result.severity == "critical"


def test_authority_divergence_requires_distinct_high_and_low_support(monkeypatch):
    now = datetime(2026, 2, 5, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("src.narrative.signals._now", lambda: now)
    config = NarrativeConfig()
    run = _make_run()
    buckets = [
        _bucket(
            now - timedelta(hours=1),
            doc_count=8,
            high_authority_sentiment_sum=3.8,
            high_authority_weight=4.0,
            high_authority_doc_count=4,
            low_authority_sentiment_sum=-3.6,
            low_authority_weight=4.0,
            low_authority_doc_count=4,
        )
    ]

    result = evaluate_authority_divergence(run, buckets, config)

    assert result.triggered is True
    assert result.metric_value >= config.authority_gap_trigger


def test_sentiment_regime_shift_requires_recent_confidence(monkeypatch):
    now = datetime(2026, 2, 5, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("src.narrative.signals._now", lambda: now)
    config = NarrativeConfig()
    run = _make_run()
    buckets = [
        _bucket(
            now - timedelta(hours=10),
            doc_count=6,
            sentiment_sum=-4.8,
            sentiment_weight=6.0,
            sentiment_confidence_sum=4.5,
            sentiment_doc_count=6,
        ),
        _bucket(
            now - timedelta(hours=2),
            doc_count=6,
            sentiment_sum=4.8,
            sentiment_weight=6.0,
            sentiment_confidence_sum=4.8,
            sentiment_doc_count=6,
        ),
    ]

    result = evaluate_sentiment_regime_shift(run, buckets, config)

    assert result.triggered is True
    assert result.severity == "critical"


def test_evaluate_all_signals_returns_all_trigger_types(monkeypatch):
    now = datetime(2026, 2, 5, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("src.narrative.signals._now", lambda: now)
    config = NarrativeConfig()
    run = _make_run()
    buckets = [
        _bucket(
            now - timedelta(minutes=5),
            doc_count=2,
            authority_sum=1.4,
            sentiment_sum=1.2,
            sentiment_weight=2.0,
            sentiment_confidence_sum=1.6,
            sentiment_doc_count=2,
            high_authority_sentiment_sum=0.8,
            high_authority_weight=1.0,
            high_authority_doc_count=1,
            low_authority_sentiment_sum=0.4,
            low_authority_weight=1.0,
            low_authority_doc_count=1,
        )
    ]

    evaluations = evaluate_all_signals(run, buckets, config)

    assert [evaluation.trigger_type for evaluation in evaluations] == [
        "narrative_surge",
        "cross_platform_breakout",
        "authority_divergence",
        "sentiment_regime_shift",
    ]
