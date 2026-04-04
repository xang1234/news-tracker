"""Configuration for evidence claims and LLM fallback gating.

Controls when and how the resolver cascade is allowed to invoke
an LLM as a fallback tier. All settings can be overridden via
CLAIMS_* environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClaimsConfig(BaseSettings):
    """Configuration for the claims extraction pipeline.

    The LLM fallback gate enforces disciplined escalation: LLM calls
    are only permitted when all deterministic tiers fail AND the
    passage meets high-value criteria AND budget limits allow it.

    Example:
        CLAIMS_LLM_FALLBACK_ENABLED=true
        CLAIMS_MIN_PASSAGE_LENGTH=80
        CLAIMS_DAILY_LLM_BUDGET=50
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAIMS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Master gate --
    llm_fallback_enabled: bool = Field(
        default=False,
        description="Master switch for LLM fallback tier. Off by default.",
    )

    # -- High-value passage criteria --
    min_passage_length: int = Field(
        default=80,
        ge=1,
        description=(
            "Minimum character length for a passage to qualify as "
            "high-value. Short fragments rarely justify LLM cost."
        ),
    )
    high_value_predicates: list[str] = Field(
        default_factory=lambda: [
            "supplies_to",
            "competes_with",
            "acquires",
            "invests_in",
            "partners_with",
            "revenue_exposure",
        ],
        description=(
            "Predicates that indicate high-value claims worth LLM "
            "fallback. Empty list means all predicates qualify."
        ),
    )

    # -- Budget caps --
    daily_llm_budget: int = Field(
        default=50,
        ge=0,
        description=(
            "Maximum LLM fallback invocations per calendar day. "
            "0 effectively disables fallback without flipping the master switch."
        ),
    )
    per_run_llm_budget: int = Field(
        default=10,
        ge=0,
        description="Maximum LLM fallback invocations per lane run.",
    )

    # -- Confidence thresholds --
    llm_proposed_confidence: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description=(
            "Default confidence assigned to LLM-proposed resolutions. "
            "Lower than fuzzy (0.6) to signal review is needed."
        ),
    )
    auto_approve_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence above which an LLM resolution can skip review. "
            "Set very high by default to make auto-approve rare."
        ),
    )
