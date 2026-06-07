"""Supporting-evidence payloads for alerts (epic o59.2, explainability).

Turns the doc→metric attribution (epic o59.1) into the "receipt" attached to a
theme-metric alert: the documents whose sentiment/volume contribution drove the
metric that fired the alert. Pure builder here; the I/O orchestration (running
attribution, attaching to the alert) lives in ``AlertService``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.themes.attribution import DocumentContribution

# Theme-metric triggers whose cause is decomposable into contributing documents.
# Lifecycle/new-theme/propagated alerts aren't metric decompositions, so they
# don't get document evidence (their trigger_data already carries their cause).
EVIDENCE_TRIGGER_TYPES: frozenset[str] = frozenset(
    {"sentiment_velocity", "extreme_sentiment", "volume_surge"}
)


def document_evidence_payload(
    contributions: list[DocumentContribution],
    *,
    window_days: int,
) -> dict[str, Any]:
    """Build a supporting_evidence payload from ranked document contributions.

    Returns ``{}`` when there are no contributions, so an alert without
    attributable documents simply carries no receipt rather than an empty shell.
    """
    if not contributions:
        return {}
    return {
        "source": "doc_metric_attribution",
        "window_days": window_days,
        "documents": [
            {
                "document_id": c.document_id,
                "timestamp": c.timestamp.isoformat(),
                "platform": c.platform,
                "sentiment_contribution": round(c.sentiment_contribution, 6),
                "volume_contribution": round(c.volume_contribution, 6),
            }
            for c in contributions
        ],
    }
