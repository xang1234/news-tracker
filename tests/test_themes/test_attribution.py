"""Tests for document → theme-metric attribution (o59.1).

The decomposition is exact: per-document sentiment_contributions sum to the
aggregate's bullish_ratio − bearish_ratio, and volume_contributions sum to 1.
Pure tests use a deterministic weigher; one consistency test uses the real
SentimentAggregator to prove contributions reconcile with the displayed number.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.sentiment.aggregation import (
    DocumentSentiment,
    SentimentAggregator,
    document_sentiments_from_rows,
)
from src.themes.attribution import AttributionService, attribute_documents

REF = datetime(2026, 6, 1, tzinfo=UTC)


def _doc(doc_id: str, label: str, *, authority=None, ts=REF) -> DocumentSentiment:
    return DocumentSentiment(
        document_id=doc_id,
        timestamp=ts,
        label=label,
        confidence=0.8,
        scores={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
        authority_score=authority,
    )


def _uniform(_doc, _ref):  # deterministic weigher: every doc weight 1.0
    return 1.0


class TestAttributeDocuments:
    def test_empty_docs(self) -> None:
        assert attribute_documents([], weigher=_uniform, reference_time=REF) == []

    def test_zero_total_weight_returns_empty(self) -> None:
        docs = [_doc("a", "positive")]
        assert attribute_documents(docs, weigher=lambda d, r: 0.0, reference_time=REF) == []

    def test_exact_decomposition(self) -> None:
        # 2 positive, 1 negative, 1 neutral, uniform weights → net = (2-1)/4 = 0.25.
        docs = [
            _doc("p1", "positive"),
            _doc("p2", "positive"),
            _doc("n1", "negative"),
            _doc("z1", "neutral"),
        ]
        out = attribute_documents(docs, weigher=_uniform, reference_time=REF)
        assert pytest.approx(sum(c.sentiment_contribution for c in out)) == 0.25
        assert pytest.approx(sum(c.volume_contribution for c in out)) == 1.0
        # Each positive contributes +1/4, the negative −1/4, neutral 0.
        by_id = {c.document_id: c for c in out}
        assert by_id["p1"].sentiment_contribution == pytest.approx(0.25)
        assert by_id["n1"].sentiment_contribution == pytest.approx(-0.25)
        assert by_id["z1"].sentiment_contribution == 0.0

    def test_ranked_by_absolute_sentiment_then_volume(self) -> None:
        # Heavier-weighted docs move sentiment more; neutral (0 sentiment) ranks last.
        docs = [_doc("z", "neutral"), _doc("pos", "positive"), _doc("neg", "negative")]
        weights = {"z": 5.0, "pos": 3.0, "neg": 1.0}
        out = attribute_documents(
            docs, weigher=lambda d, r: weights[d.document_id], reference_time=REF
        )
        # pos (|0.333|) > neg (|0.111|) > z (0.0) despite z having the largest volume.
        assert [c.document_id for c in out] == ["pos", "neg", "z"]

    def test_limit_truncates(self) -> None:
        docs = [_doc(f"d{i}", "positive") for i in range(5)]
        out = attribute_documents(docs, weigher=_uniform, reference_time=REF, limit=2)
        assert len(out) == 2

    def test_reconciles_with_real_aggregator(self) -> None:
        # The decisive test: contributions computed with the aggregator's own weight
        # must sum to the aggregate net sentiment it reports.
        agg = SentimentAggregator()
        docs = [
            _doc("p", "positive", authority=0.9),
            _doc("n", "negative", authority=0.2),
            _doc("z", "neutral"),
        ]
        result = agg.aggregate_theme_sentiment("t1", None, docs, window_days=30, reference_time=REF)
        out = attribute_documents(docs, weigher=agg.document_weight, reference_time=REF, limit=None)
        net = sum(c.sentiment_contribution for c in out)
        # The aggregate rounds bullish/bearish to 4 dp; the decomposition is exact,
        # so they reconcile to within that rounding granularity.
        assert net == pytest.approx(result.bullish_ratio - result.bearish_ratio, abs=1e-4)


class _FakeRepo:
    def __init__(self, rows) -> None:
        self._rows = rows
        self.calls: list = []

    async def get_sentiments_for_theme(self, theme_id, *, since, until):
        self.calls.append((theme_id, since, until))
        return self._rows


def _row(doc_id, label, ts=REF):
    return {
        "document_id": doc_id,
        "timestamp": ts,
        "platform": "news",
        "authority_score": 0.5,
        "sentiment": {"label": label, "confidence": 0.8, "scores": {}},
    }


class TestAttributionService:
    @pytest.mark.asyncio
    async def test_attribute_theme_ranks_contributions(self) -> None:
        repo = _FakeRepo([_row("p", "positive"), _row("n", "negative")])
        svc = AttributionService(repo, SentimentAggregator())
        out = await svc.attribute_theme("t1", window_days=7, reference_time=REF)
        assert {c.document_id for c in out} == {"p", "n"}
        # Repo queried with the trailing window.
        theme_id, since, until = repo.calls[0]
        assert theme_id == "t1" and until == REF

    @pytest.mark.asyncio
    async def test_no_documents_returns_empty(self) -> None:
        svc = AttributionService(_FakeRepo([]), SentimentAggregator())
        assert await svc.attribute_theme("t1", reference_time=REF) == []


class TestDocumentSentimentsFromRows:
    def test_skips_missing_and_invalid_sentiment(self) -> None:
        rows = [
            _row("ok", "positive"),
            {"document_id": "no_sent", "timestamp": REF, "sentiment": None},
            {"document_id": "bad_label", "timestamp": REF, "sentiment": {"label": "??"}},
        ]
        docs = document_sentiments_from_rows(rows)
        assert [d.document_id for d in docs] == ["ok"]
