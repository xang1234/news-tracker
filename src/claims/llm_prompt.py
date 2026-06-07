"""Pure prompt construction + response parsing for the LLM claim extractor.

The LLM second pass (epic ``7th``) augments the regex ``narrative_extractor``
by reading raw document text and emitting the SAME ``EvidenceClaim`` shape —
catching implicit/multi-clause relationships the patterns miss (e.g. "surging
demand from NVIDIA for TSMC's packaging" ⇒ ``TSMC supplies_to NVIDIA``).

Both functions are pure (no I/O), mirroring ``briefing``/``qa``: the service
owns the breaker-guarded round-trip via the shared ``JsonLLMClient``.
``parse_extraction_response`` builds claims with the *same* deterministic
``make_claim_key`` the rule pass uses, so identical triples dedup-merge through
``ON CONFLICT`` regardless of which pass found them.
"""

from __future__ import annotations

import json
from typing import Any

from src.claims.narrative_extractor import NARRATIVE_PREDICATES
from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key
from src.contracts.intelligence.lanes import LANE_NARRATIVE

_EXTRACTION_PROMPT = """\
You are an information-extraction system for semiconductor-industry news. \
Extract every factual relationship the text asserts between companies or \
products as subject–predicate–object claims, INCLUDING relationships stated \
only implicitly (e.g. "surging demand from NVIDIA for TSMC's packaging" \
implies the claim TSMC supplies_to NVIDIA).

Use ONLY these predicates (snake_case):
{predicate_list}

Rules:
- subject and object are canonical company or product names drawn from the text.
- Set object to null only when the predicate is inherently unary.
- confidence is your 0.0-1.0 certainty that the text supports the claim.
- Do not invent entities or relationships the text does not support.
- Extract at most {max_claims} claims.

Reply with ONLY a JSON object of this shape:
{{"claims": [{{"subject": "<s>", "predicate": "<p>", "object": "<o>|null", \
"confidence": 0.0}}, ...]}}

Text:
{content}
"""


def build_extraction_prompt(content: str, *, max_claims: int) -> str:
    """Build the extraction prompt for a document's raw text."""
    predicate_list = "\n".join(f"- {p}" for p in sorted(NARRATIVE_PREDICATES))
    return _EXTRACTION_PROMPT.format(
        predicate_list=predicate_list, max_claims=max_claims, content=content
    )


def _confidence(value: Any) -> float:
    """Coerce a model-supplied confidence to a clamped float (default 0.5)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.5
    return max(0.0, min(1.0, float(value)))


def parse_extraction_response(
    payload: Any,
    *,
    doc_id: str,
    run_id: str | None = None,
    published_at: Any | None = None,
    allowed_predicates: frozenset[str] = NARRATIVE_PREDICATES,
    min_confidence: float = 0.0,
) -> list[EvidenceClaim]:
    """Parse an LLM extraction response into schema-valid ``EvidenceClaim``s.

    Keeps only entries with a non-empty subject and an in-vocabulary predicate;
    predicates are normalised to snake_case, confidence is clamped (entries
    below ``min_confidence`` dropped), and claim_keys dedupe within the response.
    Malformed input yields ``[]`` (caller degrades to the rule pass).
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(payload, dict):
        return []
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return []

    claims: list[EvidenceClaim] = []
    seen_keys: set[str] = set()
    for entry in raw_claims:
        if not isinstance(entry, dict):
            continue
        subject = entry.get("subject")
        predicate = entry.get("predicate")
        if not isinstance(subject, str) or not subject.strip():
            continue
        if not isinstance(predicate, str) or not predicate.strip():
            continue
        normalized_predicate = predicate.strip().lower().replace(" ", "_")
        if normalized_predicate not in allowed_predicates:
            continue
        confidence = _confidence(entry.get("confidence"))
        if confidence < min_confidence:
            continue

        subject = subject.strip()
        obj = entry.get("object")
        object_text = obj.strip() if isinstance(obj, str) and obj.strip() else None

        claim_key = make_claim_key(
            LANE_NARRATIVE, doc_id, subject, normalized_predicate, object_text
        )
        if claim_key in seen_keys:
            continue
        seen_keys.add(claim_key)

        claims.append(
            EvidenceClaim(
                claim_id=make_claim_id(claim_key),
                claim_key=claim_key,
                lane=LANE_NARRATIVE,
                run_id=run_id,
                source_id=doc_id,
                source_type="document",
                subject_text=subject,
                predicate=normalized_predicate,
                object_text=object_text,
                confidence=confidence,
                extraction_method="llm",
                source_published_at=published_at,
                metadata={"detection": "llm"},
            )
        )
    return claims
