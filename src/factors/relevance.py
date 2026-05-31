"""Theme-to-factor relevance mapping."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol


class ThemeWithFactorTags(Protocol):
    theme_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FactorTagRule:
    """Explicit word/phrase rule for deriving factor relevance tags."""

    tags: frozenset[str]
    terms: tuple[str, ...]


_WORD_RE = re.compile(r"[a-z0-9]+")
_SPACING_RE = re.compile(r"[-_]+")

_DERIVED_TAG_RULES: tuple[FactorTagRule, ...] = (
    FactorTagRule(
        tags=frozenset({"semiconductors"}),
        terms=(
            "semiconductor",
            "semiconductors",
            "chip",
            "chips",
            "gpu",
            "hbm",
            "nvidia",
            "amd",
            "tsmc",
            "foundry",
            "fab",
        ),
    ),
    FactorTagRule(tags=frozenset({"memory"}), terms=("hbm", "dram", "nand", "memory")),
    FactorTagRule(
        tags=frozenset({"ai_infrastructure", "energy"}),
        terms=("ai", "data center", "datacenter", "power", "electricity"),
    ),
    FactorTagRule(
        tags=frozenset({"energy"}),
        terms=("energy", "utility", "electricity", "power"),
    ),
    FactorTagRule(
        tags=frozenset({"rates", "yield_curve", "macro"}),
        terms=("rate", "rates", "yield", "treasury", "fed"),
    ),
    FactorTagRule(
        tags=frozenset({"inflation", "consumer", "macro"}),
        terms=("inflation", "cpi", "ppi", "consumer price"),
    ),
    FactorTagRule(
        tags=frozenset({"labor", "macro"}),
        terms=("jobs", "labor", "payroll", "employment"),
    ),
    FactorTagRule(
        tags=frozenset({"industry", "macro"}),
        terms=("industrial", "production", "manufacturing", "pmi"),
    ),
    FactorTagRule(tags=frozenset({"growth", "macro"}), terms=("gdp", "growth")),
    FactorTagRule(
        tags=frozenset({"profits", "earnings", "macro"}),
        terms=("profit", "profits", "earnings", "margin"),
    ),
    FactorTagRule(
        tags=frozenset({"trade", "supply_chain"}),
        terms=("trade", "import", "imports", "export", "tariff", "china"),
    ),
    FactorTagRule(tags=frozenset({"capex"}), terms=("capex", "equipment", "asml")),
)


def extract_theme_factor_tags(theme: ThemeWithFactorTags) -> set[str]:
    """Return explicit and inferred factor tags for a theme."""
    metadata = theme.metadata or {}
    raw_tags = metadata.get("factor_relevance_tags") or metadata.get("relevance_tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]

    tags = {normalise_factor_tag(str(tag)) for tag in raw_tags if str(tag).strip()}
    text = _theme_text(theme)
    tags.update(_derive_factor_tags(text))
    return {tag for tag in tags if tag}


def normalise_factor_tag(value: str) -> str:
    """Normalise factor relevance tags to the catalog convention."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _theme_text(theme: ThemeWithFactorTags) -> str:
    text_parts = [
        getattr(theme, "name", ""),
        getattr(theme, "description", ""),
        " ".join(str(keyword) for keyword in getattr(theme, "top_keywords", []) or []),
        " ".join(str(ticker) for ticker in getattr(theme, "top_tickers", []) or []),
    ]
    for entity in getattr(theme, "top_entities", []) or []:
        if isinstance(entity, dict):
            text_parts.append(str(entity.get("name") or entity.get("text") or ""))
        else:
            text_parts.append(str(entity))
    return _normalise_text(" ".join(text_parts))


def _derive_factor_tags(text: str) -> set[str]:
    tokens = set(_word_tokens(text))
    tags: set[str] = set()
    for rule in _DERIVED_TAG_RULES:
        if any(_term_matches(text, tokens, term) for term in rule.terms):
            tags.update(rule.tags)
    return tags


def _normalise_text(value: str) -> str:
    return _SPACING_RE.sub(" ", value.lower())


def _word_tokens(value: str) -> Iterable[str]:
    return _WORD_RE.findall(value)


def _term_matches(text: str, tokens: set[str], term: str) -> bool:
    term_text = _normalise_text(term)
    term_tokens = _WORD_RE.findall(term_text)
    if not term_tokens:
        return False
    if len(term_tokens) == 1:
        return term_tokens[0] in tokens

    pattern = r"\b" + r"\s+".join(re.escape(token) for token in term_tokens) + r"\b"
    return re.search(pattern, text) is not None
