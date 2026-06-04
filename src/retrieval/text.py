"""Compose the canonical text used to embed a claim for semantic retrieval.

Retrieval runs over the *structured* layer, so a claim is embedded as a
synthesized sentence built from its resolved triple (subject — humanized
predicate — object) plus any typed numeric fact, rather than its raw source
span (which can be long and noisy). The function is pure and deterministic:
the same claim always yields the same text, so embeddings are reproducible
and re-indexing is idempotent.
"""

from __future__ import annotations

from src.claims.schemas import EvidenceClaim


def _format_number(value: float) -> str:
    """Render a magnitude without trailing ``.0`` noise for whole numbers."""
    if value == int(value):
        return str(int(value))
    return str(value)


def claim_embedding_text(claim: EvidenceClaim) -> str:
    """Build the embedding text for a claim.

    Shape: ``"<subject> <predicate humanized> <object> <metric value unit
    period (modality)>"`` — trailing clauses are omitted when their fields
    are absent.
    """
    parts = [claim.subject_text, claim.predicate.replace("_", " ")]
    if claim.object_text:
        parts.append(claim.object_text)

    if claim.metric and claim.numeric_value is not None:
        fact = [claim.metric, _format_number(claim.numeric_value)]
        if claim.unit:
            fact.append(claim.unit)
        if claim.period:
            fact.append(claim.period)
        if claim.modality:
            fact.append(f"({claim.modality})")
        parts.extend(fact)

    return " ".join(p.strip() for p in parts if p and p.strip()).strip()
