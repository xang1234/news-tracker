"""Section alignment and diff logic for sequential filing comparison.

Aligns equivalent sections across two filings (e.g., consecutive 10-K
filings from the same issuer) and computes section-level diffs. Used
by drift/divergence scoring to detect meaningful changes.

Design:
    - SectionAligner: fuzzy-matches sections by normalized name
    - SectionDiff: captures what changed between aligned sections
    - align_sections(): pure function, no I/O
    - diff_aligned_sections(): pure function, no I/O
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from src.filing.persistence import FilingSectionRecord

# -- Normalization ---------------------------------------------------------

# Patterns to strip from section names for matching
_STRIP_PATTERNS = [
    re.compile(r"^item\s+\d+[a-z]?\.?\s*", re.IGNORECASE),
    re.compile(r"^part\s+[ivx]+\.?\s*", re.IGNORECASE),
    re.compile(r"\s+", re.IGNORECASE),
]


def normalize_section_name(name: str) -> str:
    """Normalize a section name for fuzzy matching.

    Strips item numbers, part numbers, collapses whitespace,
    and lowercases. E.g., "Item 1A. Risk Factors" → "risk factors".
    """
    result = name.strip()
    for pattern in _STRIP_PATTERNS:
        result = pattern.sub(" " if pattern.pattern == r"\s+" else "", result)
    return result.strip().lower()


# -- Alignment result dataclasses -----------------------------------------


@dataclass
class SectionAlignment:
    """A pair of aligned sections from two filings.

    If only_in_base or only_in_target is set, the other side is None
    (section was added or removed between filings).

    Attributes:
        base_section: Section from the earlier filing (None if added).
        target_section: Section from the later filing (None if removed).
        similarity: Name similarity score (0-1) from fuzzy matching.
        normalized_name: The normalized section name used for matching.
    """

    base_section: FilingSectionRecord | None
    target_section: FilingSectionRecord | None
    similarity: float = 1.0
    normalized_name: str = ""

    @property
    def is_matched(self) -> bool:
        """Both sides present."""
        return self.base_section is not None and self.target_section is not None

    @property
    def is_added(self) -> bool:
        """Only in target (new section)."""
        return self.base_section is None and self.target_section is not None

    @property
    def is_removed(self) -> bool:
        """Only in base (deleted section)."""
        return self.base_section is not None and self.target_section is None


@dataclass
class SectionDiff:
    """Diff result for an aligned section pair.

    Attributes:
        alignment: The section alignment this diff is based on.
        content_changed: Whether the content differs.
        word_count_delta: Change in word count (target - base).
        hash_changed: Whether the content hash changed.
        added_lines: Number of lines added.
        removed_lines: Number of lines removed.
        diff_ratio: Similarity ratio from SequenceMatcher (0-1).
        diff_summary: Human-readable summary of changes.
    """

    alignment: SectionAlignment
    content_changed: bool = False
    word_count_delta: int = 0
    hash_changed: bool = False
    added_lines: int = 0
    removed_lines: int = 0
    diff_ratio: float = 1.0
    diff_summary: str = ""


@dataclass
class FilingComparison:
    """Complete comparison result between two filings.

    Attributes:
        base_accession: Accession number of the earlier filing.
        target_accession: Accession number of the later filing.
        alignments: All section alignments.
        diffs: Diffs for matched sections.
        sections_added: Count of sections only in target.
        sections_removed: Count of sections only in base.
        sections_changed: Count of sections with content changes.
        sections_unchanged: Count of sections with identical content.
    """

    base_accession: str
    target_accession: str
    alignments: list[SectionAlignment] = field(default_factory=list)
    diffs: list[SectionDiff] = field(default_factory=list)
    sections_added: int = 0
    sections_removed: int = 0
    sections_changed: int = 0
    sections_unchanged: int = 0


# -- Alignment logic -------------------------------------------------------


def align_sections(
    base_sections: list[FilingSectionRecord],
    target_sections: list[FilingSectionRecord],
    *,
    min_similarity: float = 0.6,
) -> list[SectionAlignment]:
    """Align sections from two filings by normalized name.

    Uses fuzzy matching on normalized section names to pair up
    equivalent sections. Unpaired sections are returned as
    added (only in target) or removed (only in base).

    Args:
        base_sections: Sections from the earlier filing.
        target_sections: Sections from the later filing.
        min_similarity: Minimum name similarity to consider a match.

    Returns:
        List of SectionAlignment objects (matched, added, removed).
    """
    alignments: list[SectionAlignment] = []
    used_targets: set[int] = set()

    # Build normalized name index for target sections
    target_names = [
        normalize_section_name(s.section_name) for s in target_sections
    ]

    for base_sec in base_sections:
        base_name = normalize_section_name(base_sec.section_name)
        best_match_idx = -1
        best_similarity = 0.0

        for j, target_name in enumerate(target_names):
            if j in used_targets:
                continue
            sim = difflib.SequenceMatcher(
                None, base_name, target_name
            ).ratio()
            if sim > best_similarity and sim >= min_similarity:
                best_similarity = sim
                best_match_idx = j

        if best_match_idx >= 0:
            used_targets.add(best_match_idx)
            alignments.append(
                SectionAlignment(
                    base_section=base_sec,
                    target_section=target_sections[best_match_idx],
                    similarity=best_similarity,
                    normalized_name=base_name,
                )
            )
        else:
            alignments.append(
                SectionAlignment(
                    base_section=base_sec,
                    target_section=None,
                    similarity=0.0,
                    normalized_name=base_name,
                )
            )

    # Add unmatched target sections as "added"
    for j, target_sec in enumerate(target_sections):
        if j not in used_targets:
            alignments.append(
                SectionAlignment(
                    base_section=None,
                    target_section=target_sec,
                    similarity=0.0,
                    normalized_name=normalize_section_name(
                        target_sec.section_name
                    ),
                )
            )

    return alignments


# -- Diff logic ------------------------------------------------------------


def diff_aligned_sections(
    alignments: list[SectionAlignment],
) -> list[SectionDiff]:
    """Compute diffs for aligned section pairs.

    Only produces diffs for matched pairs (both sides present).
    Added/removed sections are captured in the alignment itself.

    Args:
        alignments: Output from align_sections().

    Returns:
        List of SectionDiff objects for matched pairs.
    """
    diffs: list[SectionDiff] = []

    for alignment in alignments:
        if not alignment.is_matched:
            continue

        base = alignment.base_section
        target = alignment.target_section
        assert base is not None and target is not None

        hash_changed = base.content_hash != target.content_hash
        content_changed = base.content != target.content
        word_delta = target.word_count - base.word_count

        # Compute line-level diff stats
        base_lines = base.content.splitlines()
        target_lines = target.content.splitlines()
        matcher = difflib.SequenceMatcher(None, base_lines, target_lines)
        ratio = matcher.ratio()

        added = 0
        removed = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                added += j2 - j1
            elif tag == "delete":
                removed += i2 - i1
            elif tag == "replace":
                added += j2 - j1
                removed += i2 - i1

        # Build summary
        if not content_changed:
            summary = "No changes"
        elif word_delta > 0:
            summary = f"+{word_delta} words, +{added}/-{removed} lines"
        elif word_delta < 0:
            summary = f"{word_delta} words, +{added}/-{removed} lines"
        else:
            summary = f"Rewritten (+{added}/-{removed} lines, same word count)"

        diffs.append(
            SectionDiff(
                alignment=alignment,
                content_changed=content_changed,
                word_count_delta=word_delta,
                hash_changed=hash_changed,
                added_lines=added,
                removed_lines=removed,
                diff_ratio=ratio,
                diff_summary=summary,
            )
        )

    return diffs


# -- Full comparison -------------------------------------------------------


def compare_filings(
    base_accession: str,
    target_accession: str,
    base_sections: list[FilingSectionRecord],
    target_sections: list[FilingSectionRecord],
    *,
    min_similarity: float = 0.6,
) -> FilingComparison:
    """Compare two filings at the section level.

    This is the main entry point for filing comparison. It aligns
    sections, computes diffs, and returns a complete comparison result.

    Args:
        base_accession: Accession number of the earlier filing.
        target_accession: Accession number of the later filing.
        base_sections: Sections from the earlier filing.
        target_sections: Sections from the later filing.
        min_similarity: Minimum name similarity for alignment.

    Returns:
        FilingComparison with alignments, diffs, and summary stats.
    """
    alignments = align_sections(
        base_sections, target_sections, min_similarity=min_similarity
    )
    diffs = diff_aligned_sections(alignments)

    added = sum(1 for a in alignments if a.is_added)
    removed = sum(1 for a in alignments if a.is_removed)
    changed = sum(1 for d in diffs if d.content_changed)
    unchanged = sum(1 for d in diffs if not d.content_changed)

    return FilingComparison(
        base_accession=base_accession,
        target_accession=target_accession,
        alignments=alignments,
        diffs=diffs,
        sections_added=added,
        sections_removed=removed,
        sections_changed=changed,
        sections_unchanged=unchanged,
    )
