"""Document → theme-metric attribution (epic ``o59``, explainability).

Answers "why did this theme's sentiment/volume move?" by decomposing the
*displayed* aggregate into per-document contributions — so an alert or catalyst
can carry the specific documents that caused it.

The decomposition is exact, not heuristic. The theme's weighted net sentiment is

    net = Σ_i (w_i · s_i) / Σ_i w_i

where ``w_i`` is the aggregator's composite weight (recency · authority ·
confidence) and ``s_i ∈ {+1, 0, −1}`` mirrors the positive/negative/neutral
label buckets. So document *i*'s ``sentiment_contribution = w_i·s_i / Σw``
and these **sum to the aggregate ``bullish_ratio − bearish_ratio``**; its
``volume_contribution = w_i / Σw`` is its share of the weighted document volume
(summing to 1). Reusing the aggregator's own ``document_weight`` keeps the
contributions consistent with the sentiment number on screen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.sentiment.aggregation import DocumentSentiment, document_sentiments_from_rows

if TYPE_CHECKING:
    from src.sentiment.aggregation import SentimentAggregator
    from src.storage.repository import DocumentRepository

DEFAULT_ATTRIBUTION_LIMIT = 10

_LABEL_POLARITY = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# A document weigher: (doc, reference_time) -> composite weight (> 0).
Weigher = Callable[[DocumentSentiment, datetime], float]


@dataclass(frozen=True)
class DocumentContribution:
    """One document's share of a theme window's sentiment/volume."""

    document_id: str
    timestamp: datetime
    platform: str | None
    weight: float
    polarity: float  # +1 positive / 0 neutral / −1 negative
    sentiment_contribution: float  # signed; Σ over docs = bullish_ratio − bearish_ratio
    volume_contribution: float  # share of weighted volume; Σ over docs = 1.0


def attribute_documents(
    docs: list[DocumentSentiment],
    *,
    weigher: Weigher,
    reference_time: datetime,
    limit: int | None = DEFAULT_ATTRIBUTION_LIMIT,
) -> list[DocumentContribution]:
    """Decompose a window's weighted sentiment/volume into ranked contributions.

    Ranked by absolute sentiment contribution (which documents moved sentiment
    most), then by volume share. Returns ``[]`` when there is no weighted mass.
    """
    weighted = [(doc, weigher(doc, reference_time)) for doc in docs]
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        return []

    contributions = [
        DocumentContribution(
            document_id=doc.document_id,
            timestamp=doc.timestamp,
            platform=doc.platform,
            weight=weight,
            polarity=_LABEL_POLARITY.get(doc.label, 0.0),
            sentiment_contribution=weight * _LABEL_POLARITY.get(doc.label, 0.0) / total_weight,
            volume_contribution=weight / total_weight,
        )
        for doc, weight in weighted
    ]
    contributions.sort(
        key=lambda c: (abs(c.sentiment_contribution), c.volume_contribution),
        reverse=True,
    )
    return contributions[:limit] if limit is not None else contributions


class AttributionService:
    """Rank the documents that most moved a theme's sentiment/volume window."""

    def __init__(
        self,
        repository: DocumentRepository,
        aggregator: SentimentAggregator,
    ) -> None:
        self._repository = repository
        self._aggregator = aggregator

    async def attribute_theme(
        self,
        theme_id: str,
        *,
        window_days: int = 7,
        limit: int | None = DEFAULT_ATTRIBUTION_LIMIT,
        reference_time: datetime | None = None,
    ) -> list[DocumentContribution]:
        """Ranked document contributions to ``theme_id`` over the trailing window."""
        ref_time = reference_time or datetime.now(UTC)
        since = ref_time - timedelta(days=window_days)
        rows: list[Any] = await self._repository.get_sentiments_for_theme(
            theme_id, since=since, until=ref_time
        )
        docs = document_sentiments_from_rows(rows)
        return attribute_documents(
            docs, weigher=self._aggregator.document_weight, reference_time=ref_time, limit=limit
        )
