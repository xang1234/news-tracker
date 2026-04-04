"""Map source passages to narrative_frame and theme_concept identities.

Separates two conflated concerns in the existing narrative system:
    - narrative_frame: the specific angle expressed in a passage
      (e.g., "TSMC HBM Expansion Bottleneck")
    - theme_concept: the broader thematic category it informs
      (e.g., "High Bandwidth Memory")

Current narrative runs use a single theme_id for both. This module
introduces explicit mapping so downstream work (filing comparisons,
divergence analysis) can target canonical concepts across narratives.

Mapping flow:
    1. Source passage arrives with text, source_id, and span offsets
    2. Mapper resolves the narrative_frame concept (specific angle)
    3. Mapper resolves the theme_concept (broader category)
    4. PassageMapping records both with traceability back to source
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.security_master.concept_schemas import make_concept_id


# -- PassageMapping --------------------------------------------------------


@dataclass
class PassageMapping:
    """Maps a source passage to its narrative frame and theme concept.

    Preserves traceability from raw text through to both concept
    layers. Later explanation and divergence work operates on these
    mappings rather than raw theme_ids.

    Attributes:
        mapping_id: Deterministic ID from source + frame + theme.
        source_id: Document/section that contains the passage.
        source_span_start: Character offset start in source text.
        source_span_end: Character offset end in source text.
        passage_text: The extracted text span (for audit/display).
        narrative_frame_id: Concept ID of the specific narrative angle.
        narrative_frame_name: Human-readable frame name.
        theme_concept_id: Concept ID of the broader theme.
        theme_concept_name: Human-readable theme name.
        narrative_run_id: The narrative run this passage belongs to.
        confidence: How confidently this mapping was assigned (0-1).
        metadata: Extensible metadata.
    """

    mapping_id: str
    source_id: str
    narrative_frame_id: str
    theme_concept_id: str
    narrative_frame_name: str = ""
    theme_concept_name: str = ""
    source_span_start: int | None = None
    source_span_end: int | None = None
    passage_text: str | None = None
    narrative_run_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def make_mapping_id(
    source_id: str,
    narrative_frame_id: str,
    theme_concept_id: str,
) -> str:
    """Generate a deterministic mapping ID.

    Same source + frame + theme always produces the same ID,
    enabling idempotent upserts.
    """
    parts = [source_id, narrative_frame_id, theme_concept_id]
    key_input = "\x00".join(parts)
    return f"pmap_{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"


# -- PassageMapper ---------------------------------------------------------


class PassageMapper:
    """Maps passages to narrative_frame and theme_concept identities.

    Resolves the specific narrative angle (frame) and the broader
    thematic concept (theme) for each passage. Uses the canonical
    identity layer for concept creation and lookup.

    Usage:
        mapper = PassageMapper()
        mapping = mapper.map_passage(
            source_id="doc_123",
            frame_name="TSMC HBM Expansion Bottleneck",
            theme_name="High Bandwidth Memory",
        )
    """

    def map_passage(
        self,
        *,
        source_id: str,
        frame_name: str,
        theme_name: str,
        passage_text: str | None = None,
        source_span_start: int | None = None,
        source_span_end: int | None = None,
        narrative_run_id: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> PassageMapping:
        """Map a source passage to its frame and theme concepts.

        Creates deterministic concept IDs for both the narrative
        frame and the theme, then builds a PassageMapping that
        links them to the source passage.

        Args:
            source_id: Document/section containing the passage.
            frame_name: The specific narrative angle.
            theme_name: The broader thematic category.
            passage_text: Extracted text span (optional).
            source_span_start: Character offset start.
            source_span_end: Character offset end.
            narrative_run_id: Associated narrative run.
            confidence: Mapping confidence (0-1).
            metadata: Extra context.

        Returns:
            PassageMapping with resolved concept IDs.
        """
        frame_id = make_concept_id("narrative_frame", frame_name)
        theme_id = make_concept_id("theme", theme_name)
        mapping_id = make_mapping_id(source_id, frame_id, theme_id)

        return PassageMapping(
            mapping_id=mapping_id,
            source_id=source_id,
            narrative_frame_id=frame_id,
            narrative_frame_name=frame_name,
            theme_concept_id=theme_id,
            theme_concept_name=theme_name,
            source_span_start=source_span_start,
            source_span_end=source_span_end,
            passage_text=passage_text,
            narrative_run_id=narrative_run_id,
            confidence=confidence,
            metadata=metadata or {},
        )

    def map_from_narrative_run(
        self,
        *,
        source_id: str,
        frame_name: str,
        theme_name: str,
        narrative_run_id: str,
        passage_text: str | None = None,
        confidence: float = 1.0,
    ) -> PassageMapping:
        """Convenience: map a passage from a narrative run context.

        Shorthand for map_passage when the narrative_run_id is known.
        """
        return self.map_passage(
            source_id=source_id,
            frame_name=frame_name,
            theme_name=theme_name,
            narrative_run_id=narrative_run_id,
            passage_text=passage_text,
            confidence=confidence,
        )

    @staticmethod
    def resolve_frame_concept_id(frame_name: str) -> str:
        """Get the deterministic concept ID for a narrative frame."""
        return make_concept_id("narrative_frame", frame_name)

    @staticmethod
    def resolve_theme_concept_id(theme_name: str) -> str:
        """Get the deterministic concept ID for a theme."""
        return make_concept_id("theme", theme_name)
