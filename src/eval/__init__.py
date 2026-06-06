"""Evaluation harnesses for the parsing/extraction layer.

The extraction eval (epic ``7th``) scores any extractor against a labelled
golden set by matching (subject, predicate, object) triples, reporting
recall/precision. It gates the hybrid-LLM rollout: the LLM pass must beat the
regex extractor on recall without regressing precision below the agreed floor.
"""

from __future__ import annotations

from src.eval.extraction import Extractor, evaluate, normalize_triple, rule_extractor
from src.eval.golden import DEFAULT_GOLDEN_PATH, load_golden_set
from src.eval.schemas import DocEval, ExtractionEval, GoldenClaim, GoldenDocument

__all__ = [
    "DEFAULT_GOLDEN_PATH",
    "DocEval",
    "Extractor",
    "ExtractionEval",
    "GoldenClaim",
    "GoldenDocument",
    "evaluate",
    "load_golden_set",
    "normalize_triple",
    "rule_extractor",
]
