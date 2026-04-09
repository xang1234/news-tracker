"""Narrative claim extractor — converts document events and entity
co-occurrences into evidence claims.

This is the missing LANE_NARRATIVE claim producer.  It reads documents
that already have ``events_extracted`` and ``entities_mentioned``
populated by the preprocessing pipeline and converts them into
``EvidenceClaim`` objects that feed into the assertion layer.

Two extraction paths:

1. **Event → Claim**: Each extracted event (SVO triplet from
   ``PatternExtractor``) becomes one claim whose predicate is derived
   from the event type.

2. **Entity co-occurrence → Claim**: When two COMPANY entities appear
   near each other alongside a supply-chain verb, a lower-confidence
   relationship claim is emitted.

All claims use deterministic keys (``make_claim_key``) so re-processing
the same document is idempotent.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key
from src.contracts.intelligence.lanes import LANE_NARRATIVE

logger = logging.getLogger(__name__)

# -- Event type → predicate mapping ----------------------------------------

EVENT_TYPE_TO_PREDICATE: dict[str, str] = {
    "capacity_expansion": "expands_capacity",
    "capacity_constraint": "constrains_capacity",
    "product_launch": "launches_product",
    "product_delay": "delays_product",
    "price_change": "changes_pricing",
    "guidance_change": "revises_guidance",
}

# -- Co-occurrence verb patterns -------------------------------------------

_SUPPLY_VERBS = re.compile(
    r"\b(?:suppl(?:y|ies|ied|ying)|provid(?:e|es|ed|ing)|manufactur(?:e|es|ed|ing)|"
    r"fabricat(?:e|es|ed|ing)|produc(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_CUSTOMER_VERBS = re.compile(
    r"\b(?:us(?:e|es|ed|ing)|adopt(?:s|ed|ing)?|purchas(?:e|es|ed|ing)|"
    r"bu(?:y|ys|ying|ought)|order(?:s|ed|ing)?|sourc(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_COMPETE_VERBS = re.compile(
    r"\b(?:compet(?:e|es|ed|ing)|rival(?:s|ed|ing)?|challeng(?:e|es|ed|ing)|"
    r"undercut(?:s|ting)?|displac(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_PARTNER_VERBS = re.compile(
    r"\b(?:partner(?:s|ed|ing)?|collaborat(?:e|es|ed|ing)|"
    r"joint(?:ly)?|co-develop(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)

_VERB_TO_PREDICATE: list[tuple[re.Pattern[str], str]] = [
    (_SUPPLY_VERBS, "supplies_to"),
    (_CUSTOMER_VERBS, "customer_of"),
    (_COMPETE_VERBS, "competes_with"),
    (_PARTNER_VERBS, "depends_on"),
]

# Maximum character window for co-occurrence detection
_COOCCURRENCE_WINDOW = 300


def extract_claims_from_events(
    doc_id: str,
    events: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    published_at: Any | None = None,
) -> list[EvidenceClaim]:
    """Convert extracted events into evidence claims.

    Each event with an ``actor`` becomes one claim.  The predicate is
    derived from the event type via ``EVENT_TYPE_TO_PREDICATE``.

    Args:
        doc_id: Source document ID.
        events: List of event dicts (as stored in ``events_extracted`` JSONB).
        run_id: Optional lane run identifier.
        published_at: Document publication timestamp.

    Returns:
        List of ``EvidenceClaim`` objects (may be empty).
    """
    claims: list[EvidenceClaim] = []

    for event in events:
        actor = (event.get("actor") or "").strip()
        if not actor:
            continue

        event_type = event.get("event_type", "")
        predicate = EVENT_TYPE_TO_PREDICATE.get(event_type)
        if predicate is None:
            continue

        object_text = (event.get("object") or "").strip() or None
        confidence = float(event.get("confidence", 0.7))
        span_start = event.get("span_start")
        span_end = event.get("span_end")

        claim_key = make_claim_key(
            LANE_NARRATIVE,
            doc_id,
            actor,
            predicate,
            object_text,
        )
        claim_id = make_claim_id(claim_key)

        claims.append(
            EvidenceClaim(
                claim_id=claim_id,
                claim_key=claim_key,
                lane=LANE_NARRATIVE,
                run_id=run_id,
                source_id=doc_id,
                source_type="document",
                source_span_start=span_start,
                source_span_end=span_end,
                source_text=event.get("action"),
                subject_text=actor,
                predicate=predicate,
                object_text=object_text,
                confidence=confidence,
                extraction_method="rule",
                source_published_at=published_at,
                metadata={
                    "event_type": event_type,
                    "tickers": event.get("tickers", []),
                    "time_ref": event.get("time_ref"),
                    "quantity": event.get("quantity"),
                },
            )
        )

    return claims


def extract_claims_from_cooccurrence(
    doc_id: str,
    entities: list[dict[str, Any]],
    content: str,
    *,
    run_id: str | None = None,
    published_at: Any | None = None,
) -> list[EvidenceClaim]:
    """Infer relationship claims from entity co-occurrence.

    When two COMPANY (or TICKER) entities appear within a text window
    that also contains a supply-chain verb, emit a lower-confidence
    relationship claim.

    Args:
        doc_id: Source document ID.
        entities: List of entity dicts from ``entities_mentioned``.
        content: Full document text for verb scanning.
        run_id: Optional lane run identifier.
        published_at: Document publication timestamp.

    Returns:
        List of ``EvidenceClaim`` objects (may be empty).
    """
    # Collect company/ticker entities with their text positions
    company_entities: list[tuple[str, int]] = []
    for ent in entities:
        ent_type = (ent.get("type") or ent.get("entity_type") or "").upper()
        if ent_type not in ("COMPANY", "TICKER", "ORG"):
            continue
        text = (ent.get("text") or ent.get("name") or "").strip()
        if not text:
            continue
        start = ent.get("start", 0) or 0
        company_entities.append((text, start))

    if len(company_entities) < 2:
        return []

    claims: list[EvidenceClaim] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    for i, (entity_a, pos_a) in enumerate(company_entities):
        for entity_b, pos_b in company_entities[i + 1 :]:
            if entity_a.lower() == entity_b.lower():
                continue

            # Check proximity
            if abs(pos_a - pos_b) > _COOCCURRENCE_WINDOW:
                continue

            # Extract text window covering both entities and surrounding context.
            # Extends to the full co-occurrence window so verbs appearing just
            # after the later entity are still captured.
            window_start = min(pos_a, pos_b)
            window_end = min(window_start + _COOCCURRENCE_WINDOW, len(content))
            window_text = content[window_start:window_end]

            # Scan for supply-chain verbs
            for pattern, predicate in _VERB_TO_PREDICATE:
                if not pattern.search(window_text):
                    continue

                # Normalize order so (A,B) and (B,A) are treated as the same pair
                low_a, low_b = entity_a.lower(), entity_b.lower()
                pair_key = (min(low_a, low_b), max(low_a, low_b), predicate)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                claim_key = make_claim_key(
                    LANE_NARRATIVE,
                    doc_id,
                    entity_a,
                    predicate,
                    entity_b,
                )
                claim_id = make_claim_id(claim_key)

                claims.append(
                    EvidenceClaim(
                        claim_id=claim_id,
                        claim_key=claim_key,
                        lane=LANE_NARRATIVE,
                        run_id=run_id,
                        source_id=doc_id,
                        source_type="document",
                        source_span_start=window_start,
                        source_span_end=window_end,
                        source_text=window_text[:200],
                        subject_text=entity_a,
                        predicate=predicate,
                        object_text=entity_b,
                        confidence=0.45,
                        extraction_method="rule",
                        source_published_at=published_at,
                        metadata={"detection": "cooccurrence"},
                    )
                )
                break  # one predicate per entity pair

    return claims


def extract_claims_from_document(
    doc_id: str,
    events: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    content: str,
    *,
    run_id: str | None = None,
    published_at: Any | None = None,
) -> list[EvidenceClaim]:
    """Extract all narrative claims from a single document.

    Combines event-based and co-occurrence-based extraction.

    Args:
        doc_id: Source document ID.
        events: ``events_extracted`` JSONB list.
        entities: ``entities_mentioned`` JSONB list.
        content: Full document text.
        run_id: Optional lane run identifier.
        published_at: Document publication timestamp.

    Returns:
        Combined list of claims (deduplicated by claim_key).
    """
    event_claims = extract_claims_from_events(
        doc_id,
        events,
        run_id=run_id,
        published_at=published_at,
    )
    cooccurrence_claims = extract_claims_from_cooccurrence(
        doc_id,
        entities,
        content,
        run_id=run_id,
        published_at=published_at,
    )

    # Deduplicate by claim_key (event claims take priority)
    seen_keys: set[str] = set()
    combined: list[EvidenceClaim] = []

    for claim in event_claims:
        if claim.claim_key not in seen_keys:
            seen_keys.add(claim.claim_key)
            combined.append(claim)

    for claim in cooccurrence_claims:
        if claim.claim_key not in seen_keys:
            seen_keys.add(claim.claim_key)
            combined.append(claim)

    return combined
