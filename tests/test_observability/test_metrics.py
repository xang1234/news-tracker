"""Tests for Prometheus metrics helpers."""

from __future__ import annotations

from src.observability.metrics import MetricsCollector


class RecordingCounter:
    def __init__(self) -> None:
        self.labels_called_with: list[dict[str, str]] = []
        self.increments: list[int] = []

    def labels(self, **kwargs: str) -> RecordingCounter:
        self.labels_called_with.append(kwargs)
        return self

    def inc(self, amount: int = 1) -> None:
        self.increments.append(amount)


def test_rss_feed_document_metric_emits_zero_yield_series() -> None:
    counter = RecordingCounter()
    metrics = MetricsCollector.__new__(MetricsCollector)
    metrics.rss_feed_documents = counter  # type: ignore[attr-defined]

    metrics.record_rss_feed_documents("semiwiki", 0)

    assert counter.labels_called_with == [{"feed": "semiwiki"}]
    assert counter.increments == [0]
