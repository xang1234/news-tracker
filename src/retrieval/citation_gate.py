"""Shared grounding gate for cited-LLM responses.

Briefings and cited answers ask the LLM for the same JSON shape — a list of
entries under some key, each ``{"text": ..., "claim_ids": [...]}`` — and ground
them identically: keep only entries whose text is non-empty and that cite at
least one *retrieved* claim, stripping invented ids and dropping now-uncited
entries. This is that gate, parameterised by the JSON key and the value-object
factory, so neither feature can emit an uncited or hallucinated-cited assertion.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def parse_cited_entries(
    payload: Any,
    valid_claim_ids: set[str],
    *,
    key: str,
    factory: Callable[[str, list[str]], T],
) -> list[T]:
    """Parse + ground a ``{key: [{text, claim_ids}]}`` LLM response.

    Builds ``factory(text, grounded_ids)`` for each entry with non-empty text
    and at least one citation in ``valid_claim_ids``; invented ids are stripped,
    surviving ids deduped (order preserved), and now-uncited entries dropped.
    A string payload is JSON-decoded first; any malformed input yields ``[]``.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(payload, dict):
        return []
    raw_entries = payload.get(key)
    if not isinstance(raw_entries, list):
        return []

    grounded_entries: list[T] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        ids = entry.get("claim_ids")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(ids, list):
            continue
        grounded = list(
            dict.fromkeys(i for i in ids if isinstance(i, str) and i in valid_claim_ids)
        )
        if not grounded:
            continue
        grounded_entries.append(factory(text.strip(), grounded))
    return grounded_entries
