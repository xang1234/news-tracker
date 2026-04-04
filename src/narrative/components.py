"""Decomposed narrative component scores.

Breaks narrative conviction into four explicit, inspectable
components. Each component is independently testable and explainable
so later ranking and UI work can answer "why is this narrative hot?"

Components:
    - Attention: velocity and acceleration of document flow
    - Corroboration: source diversity and platform spread
    - Confirmation: authority/crowd sentiment agreement
    - Novelty/Persistence: recency vs sustained duration

All component functions are stateless — they take narrative run
metrics as input and return a scored breakdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime, timezone


# -- Component scores ------------------------------------------------------


@dataclass(frozen=True)
class AttentionScore:
    """How much attention the narrative is receiving.

    Attributes:
        velocity: Normalized document rate (0-1).
        acceleration: Rate of change in velocity (0-1).
        doc_mass: Volume saturation (0-1).
        score: Weighted composite (0-1).
    """

    velocity: float
    acceleration: float
    doc_mass: float
    score: float


@dataclass(frozen=True)
class CorroborationScore:
    """How broadly the narrative is corroborated across sources.

    Attributes:
        platform_spread: Proportion of known platforms (0-1).
        source_diversity: Distinct source type saturation (0-1).
        cross_platform_speed: How quickly it spread (0-1).
        score: Weighted composite (0-1).
    """

    platform_spread: float
    source_diversity: float
    cross_platform_speed: float
    score: float


@dataclass(frozen=True)
class ConfirmationScore:
    """How much authority supports the narrative direction.

    Attributes:
        authority_alignment: High-authority sentiment agreement (0-1).
        crowd_agreement: Overall sentiment coherence (0-1).
        authority_weight: Proportion of high-authority sources (0-1).
        score: Weighted composite (0-1).
    """

    authority_alignment: float
    crowd_agreement: float
    authority_weight: float
    score: float


@dataclass(frozen=True)
class NoveltyPersistenceScore:
    """Balance between freshness and sustained duration.

    Attributes:
        recency: Exponential decay from last evidence (0-1).
        persistence: How long the narrative has been active (0-1).
        novelty_ratio: Recency vs total duration balance (0-1).
        score: Weighted composite (0-1).
    """

    recency: float
    persistence: float
    novelty_ratio: float
    score: float


@dataclass(frozen=True)
class NarrativeComponents:
    """Full decomposed narrative scoring.

    The composite score is a weighted blend of all four components,
    scaled to 0-100 for compatibility with existing conviction_score.

    Attributes:
        attention: Document flow velocity/acceleration.
        corroboration: Source diversity/platform spread.
        confirmation: Authority/crowd agreement.
        novelty_persistence: Freshness vs sustained duration.
        composite: Weighted blend (0-100).
    """

    attention: AttentionScore
    corroboration: CorroborationScore
    confirmation: ConfirmationScore
    novelty_persistence: NoveltyPersistenceScore
    composite: float

    def to_dict(self) -> dict[str, object]:
        """Serialize for storage in metadata or published payloads."""
        return {
            "attention": asdict(self.attention),
            "corroboration": asdict(self.corroboration),
            "confirmation": asdict(self.confirmation),
            "novelty_persistence": asdict(self.novelty_persistence),
            "composite": self.composite,
        }


# -- Component weights (tune these) ----------------------------------------

ATTENTION_WEIGHT = 0.30
CORROBORATION_WEIGHT = 0.25
CONFIRMATION_WEIGHT = 0.25
NOVELTY_WEIGHT = 0.20

# Normalization targets
MAX_RATE_PER_HOUR = 60.0  # docs/hr for velocity=1.0
MAX_ACCELERATION = 30.0  # docs/hr² for acceleration=1.0
MAX_DOC_COUNT = 50  # docs for doc_mass=1.0
MAX_PLATFORMS = 5  # platforms for platform_spread=1.0
DIVERSITY_TARGET = 4  # distinct source types for diversity=1.0
MAX_SPREAD_HOURS = 6.0  # hours for max cross-platform speed score
PERSISTENCE_HALFLIFE_HOURS = 48.0  # hours for half persistence credit
RECENCY_DECAY = 0.02  # exponential decay per hour


# -- Compute functions (stateless) -----------------------------------------


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_attention(
    current_rate_per_hour: float,
    current_acceleration: float,
    doc_count: int,
) -> AttentionScore:
    """Compute attention from document flow metrics."""
    velocity = _clamp01(current_rate_per_hour / MAX_RATE_PER_HOUR)
    acceleration = _clamp01(current_acceleration / MAX_ACCELERATION)
    doc_mass = _clamp01(doc_count / MAX_DOC_COUNT)

    score = _clamp01(
        velocity * 0.5 + acceleration * 0.3 + doc_mass * 0.2
    )
    return AttentionScore(
        velocity=round(velocity, 4),
        acceleration=round(acceleration, 4),
        doc_mass=round(doc_mass, 4),
        score=round(score, 4),
    )


def compute_corroboration(
    platform_count: int,
    source_type_count: int,
    spread_hours: float | None,
) -> CorroborationScore:
    """Compute corroboration from source diversity metrics."""
    platform_spread = _clamp01(platform_count / MAX_PLATFORMS)
    source_diversity = _clamp01(source_type_count / DIVERSITY_TARGET)

    if spread_hours is not None and spread_hours > 0:
        cross_platform_speed = _clamp01(
            1.0 - (spread_hours / MAX_SPREAD_HOURS)
        )
    else:
        cross_platform_speed = 0.0

    score = _clamp01(
        platform_spread * 0.4
        + source_diversity * 0.35
        + cross_platform_speed * 0.25
    )
    return CorroborationScore(
        platform_spread=round(platform_spread, 4),
        source_diversity=round(source_diversity, 4),
        cross_platform_speed=round(cross_platform_speed, 4),
        score=round(score, 4),
    )


def compute_confirmation(
    avg_sentiment: float,
    avg_authority: float,
    high_authority_doc_ratio: float,
) -> ConfirmationScore:
    """Compute confirmation from authority/crowd sentiment alignment.

    Uses absolute sentiment as a proxy for directional conviction.
    High authority alignment means experts agree with the narrative.
    """
    crowd_agreement = _clamp01(abs(avg_sentiment))
    authority_alignment = _clamp01(avg_authority)
    authority_weight = _clamp01(high_authority_doc_ratio)

    score = _clamp01(
        authority_alignment * 0.45
        + crowd_agreement * 0.30
        + authority_weight * 0.25
    )
    return ConfirmationScore(
        authority_alignment=round(authority_alignment, 4),
        crowd_agreement=round(crowd_agreement, 4),
        authority_weight=round(authority_weight, 4),
        score=round(score, 4),
    )


def compute_novelty_persistence(
    last_document_at: datetime,
    started_at: datetime,
    *,
    now: datetime | None = None,
) -> NoveltyPersistenceScore:
    """Compute novelty/persistence balance.

    Recency decays exponentially from last evidence. Persistence
    grows logarithmically with duration (diminishing returns).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    hours_since_last = max(
        0.0, (now - last_document_at).total_seconds() / 3600
    )
    recency = math.exp(-RECENCY_DECAY * hours_since_last)

    duration_hours = max(
        0.0, (now - started_at).total_seconds() / 3600
    )
    persistence = _clamp01(
        math.log1p(duration_hours) / math.log1p(PERSISTENCE_HALFLIFE_HOURS)
    )

    novelty_ratio = _clamp01(recency * (1.0 - 0.3 * persistence))

    score = _clamp01(
        recency * 0.5 + persistence * 0.25 + novelty_ratio * 0.25
    )
    return NoveltyPersistenceScore(
        recency=round(recency, 4),
        persistence=round(persistence, 4),
        novelty_ratio=round(novelty_ratio, 4),
        score=round(score, 4),
    )


def compute_narrative_components(
    *,
    current_rate_per_hour: float,
    current_acceleration: float,
    doc_count: int,
    platform_count: int,
    source_type_count: int = 1,
    spread_hours: float | None = None,
    avg_sentiment: float,
    avg_authority: float,
    high_authority_doc_ratio: float = 0.0,
    last_document_at: datetime,
    started_at: datetime,
    now: datetime | None = None,
) -> NarrativeComponents:
    """Compute all four narrative components and the composite score.

    This is the main entry point. Takes raw narrative run metrics
    and returns the full decomposed scoring.

    Args:
        current_rate_per_hour: Current document arrival rate.
        current_acceleration: Rate of change in arrival rate.
        doc_count: Total documents in the narrative run.
        platform_count: Distinct platforms.
        source_type_count: Distinct source types.
        spread_hours: Hours for first 3 platforms to appear.
        avg_sentiment: Average sentiment score (-1 to 1).
        avg_authority: Average authority score (0 to 1).
        high_authority_doc_ratio: Fraction of high-authority docs.
        last_document_at: Most recent document timestamp.
        started_at: When the narrative run started.
        now: Current time (defaults to utcnow).

    Returns:
        NarrativeComponents with all four sub-scores and composite.
    """
    attention = compute_attention(
        current_rate_per_hour, current_acceleration, doc_count
    )
    corroboration = compute_corroboration(
        platform_count, source_type_count, spread_hours
    )
    confirmation = compute_confirmation(
        avg_sentiment, avg_authority, high_authority_doc_ratio
    )
    novelty_persistence = compute_novelty_persistence(
        last_document_at, started_at, now=now
    )

    composite = (
        attention.score * ATTENTION_WEIGHT
        + corroboration.score * CORROBORATION_WEIGHT
        + confirmation.score * CONFIRMATION_WEIGHT
        + novelty_persistence.score * NOVELTY_WEIGHT
    ) * 100.0
    composite = round(min(composite, 100.0), 2)

    return NarrativeComponents(
        attention=attention,
        corroboration=corroboration,
        confirmation=confirmation,
        novelty_persistence=novelty_persistence,
        composite=composite,
    )
