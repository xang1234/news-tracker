"""LLM-powered compellingness scoring for themes.

Three-tier pipeline: rule-based (free) → GPT-4o-mini (cheap) → Claude (expensive).
Evaluates theme quality across six dimensions: authority, evidence, reasoning,
risk assessment, actionability, and technical depth.

Usage:
    from src.scoring import CompellingnessService

    service = CompellingnessService()
    score = await service.score_theme(theme)
    theme.metadata["compellingness"] = score.overall_score
"""

from src.scoring.circuit_breaker import CircuitOpenError, GenericCircuitBreaker
from src.scoring.compellingness import CompellingnessService
from src.scoring.config import ScoringConfig
from src.scoring.schemas import CompellingnessScore, DimensionScores, EvidenceQuote

__all__ = [
    "CircuitOpenError",
    "CompellingnessScore",
    "CompellingnessService",
    "DimensionScores",
    "EvidenceQuote",
    "GenericCircuitBreaker",
    "ScoringConfig",
]
