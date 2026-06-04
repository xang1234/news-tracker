"""Shared evidence payload helpers for publish-lane outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

EvidencePayload = dict[str, Any]
EvidencePayloadInput = Mapping[str, Any]


def copy_evidence(
    items: Sequence[EvidencePayloadInput] | None,
) -> list[EvidencePayload]:
    """Return detached evidence payload dictionaries for publication objects."""
    return [dict(item) for item in items or []]


def evidence_for_key(
    evidence_by_key: Mapping[str, Sequence[EvidencePayloadInput]] | None,
    key: str,
) -> list[EvidencePayload]:
    """Copy evidence payloads for a key from an optional lookup."""
    if evidence_by_key is None:
        return []
    return copy_evidence(evidence_by_key.get(key))
