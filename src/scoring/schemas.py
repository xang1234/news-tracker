"""Data models for the compellingness scoring pipeline.

Defines the structured output from each scoring tier (rule-based, GPT, Claude)
and the thesis input format used to aggregate theme content for evaluation.

Follows the MS-PS (Multi-Signal Persuasion Score) framework with six dimensions:
authority, evidence, reasoning, risk assessment, actionability, and technical depth.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceQuote(BaseModel):
    """An extracted quote from theme content with relevance annotation."""

    text: str = Field(description="Verbatim quote from source content")
    relevance: str = Field(description="Why this quote supports the thesis")


class DimensionScores(BaseModel):
    """Six-dimension scoring breakdown (MS-PS framework).

    Each dimension is scored 0-10:
    - authority: Source credibility and expertise signals
    - evidence: Concrete data points, numbers, citations
    - reasoning: Logical coherence and causal chain quality
    - risk: Quality of risk/downside acknowledgment
    - actionability: Clarity of investment thesis and time horizon
    - technical: Semiconductor domain-specific depth
    """

    authority: float = Field(default=0.0, ge=0.0, le=10.0)
    evidence: float = Field(default=0.0, ge=0.0, le=10.0)
    reasoning: float = Field(default=0.0, ge=0.0, le=10.0)
    risk: float = Field(default=0.0, ge=0.0, le=10.0)
    actionability: float = Field(default=0.0, ge=0.0, le=10.0)
    technical: float = Field(default=0.0, ge=0.0, le=10.0)

    @property
    def mean(self) -> float:
        """Compute the arithmetic mean of all six dimensions."""
        return (
            self.authority
            + self.evidence
            + self.reasoning
            + self.risk
            + self.actionability
            + self.technical
        ) / 6.0


class CompellingnessScore(BaseModel):
    """Full compellingness assessment for a theme.

    Returned by all scoring tiers. The tier_used field indicates which
    level of analysis produced the score (rule/gpt/claude).
    """

    overall_score: float = Field(ge=0.0, le=10.0, description="Composite score 0-10")
    dimensions: DimensionScores = Field(default_factory=DimensionScores)
    summary: str = Field(default="", description="Brief scoring rationale")
    evidence_quotes: list[EvidenceQuote] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    time_horizon: str = Field(default="unknown", description="e.g. short/medium/long-term")
    key_risks: list[str] = Field(default_factory=list)
    flags: list[str] = Field(
        default_factory=list,
        description="Quality flags: hype_language, no_evidence, too_short, needs_human_review",
    )
    tier_used: Literal["rule", "gpt", "claude"] = "rule"
    model_version: str = Field(default="", description="Model ID that produced this score")
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cached: bool = Field(default=False, description="Whether this result came from cache")


class ThesisInput(BaseModel):
    """Aggregated theme content prepared for LLM scoring.

    Built from Theme fields — does not fetch documents. Contains everything
    an LLM needs to evaluate narrative quality.
    """

    theme_id: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    lifecycle_stage: str = "emerging"
    document_count: int = 0
