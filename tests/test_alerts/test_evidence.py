"""Tests for the pure alert supporting-evidence payload builder (o59.2)."""

from __future__ import annotations

from datetime import UTC, datetime

from src.alerts.evidence import EVIDENCE_TRIGGER_TYPES, document_evidence_payload
from src.themes.attribution import DocumentContribution

REF = datetime(2026, 6, 1, tzinfo=UTC)


def _contrib(doc_id: str, sent: float, vol: float) -> DocumentContribution:
    return DocumentContribution(
        document_id=doc_id,
        timestamp=REF,
        platform="news",
        weight=1.0,
        polarity=1.0,
        sentiment_contribution=sent,
        volume_contribution=vol,
    )


def test_empty_contributions_yields_empty_payload() -> None:
    assert document_evidence_payload([], window_days=7) == {}


def test_builds_document_receipt() -> None:
    payload = document_evidence_payload(
        [_contrib("d1", 0.6, 0.5), _contrib("d2", -0.3, 0.25)], window_days=14
    )
    assert payload["source"] == "doc_metric_attribution"
    assert payload["window_days"] == 14
    assert [d["document_id"] for d in payload["documents"]] == ["d1", "d2"]
    assert payload["documents"][0]["sentiment_contribution"] == 0.6
    assert payload["documents"][1]["volume_contribution"] == 0.25
    assert payload["documents"][0]["platform"] == "news"


def test_evidence_trigger_types_are_metric_decompositions() -> None:
    # Only triggers whose cause decomposes into documents get evidence.
    assert "sentiment_velocity" in EVIDENCE_TRIGGER_TYPES
    assert "volume_surge" in EVIDENCE_TRIGGER_TYPES
    assert "extreme_sentiment" in EVIDENCE_TRIGGER_TYPES
    assert "new_theme" not in EVIDENCE_TRIGGER_TYPES
    assert "lifecycle_change" not in EVIDENCE_TRIGGER_TYPES
    assert "propagated_impact" not in EVIDENCE_TRIGGER_TYPES
