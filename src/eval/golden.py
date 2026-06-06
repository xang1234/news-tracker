"""Golden-set loading for the extraction eval.

The golden set is a checked-in JSON file (one labelled document per entry) so
the eval is fully reproducible and CI-runnable with no external data.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.schemas import GoldenClaim, GoldenDocument

#: Default checked-in golden set.
DEFAULT_GOLDEN_PATH = Path(__file__).parent / "golden" / "extraction_golden.json"


def load_golden_set(path: Path | str | None = None) -> list[GoldenDocument]:
    """Load labelled golden documents from a JSON file.

    Schema (per document):
        {"doc_id", "content", "events": [...], "entities": [...],
         "expected_claims": [{"subject", "predicate", "object"?}, ...]}
    """
    path = Path(path) if path is not None else DEFAULT_GOLDEN_PATH
    raw = json.loads(path.read_text())
    docs: list[GoldenDocument] = []
    for entry in raw["documents"]:
        docs.append(
            GoldenDocument(
                doc_id=entry["doc_id"],
                content=entry.get("content", ""),
                events=entry.get("events", []),
                entities=entry.get("entities", []),
                expected_claims=[
                    GoldenClaim(
                        subject=c["subject"],
                        predicate=c["predicate"],
                        object=c.get("object"),
                    )
                    for c in entry.get("expected_claims", [])
                ],
            )
        )
    return docs
