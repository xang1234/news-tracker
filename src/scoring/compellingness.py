"""Main compellingness scoring service with three-tier pipeline.

Evaluates theme quality through:
  Tier 1: Rule-based keyword density scoring (free, always runs)
  Tier 2: GPT-4o-mini evaluation (cheap, for promising themes)
  Tier 3: Claude validation (expensive, for top-scoring themes only)

The service is pure — takes a Theme, returns a CompellingnessScore.
It does NOT modify Theme or touch the database. The caller is responsible
for writing ``theme.metadata["compellingness"] = score.overall_score``.

Follows the same constructor pattern as EmbeddingService/SentimentService:
  ``(config?, redis_client?)`` with lazy initialization.
"""

import hashlib
import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Any

from src.scoring.circuit_breaker import CircuitOpenError
from src.scoring.config import ScoringConfig
from src.scoring.llm_client import LLMClient
from src.scoring.prompts import (
    ACTION_KEYWORDS,
    AUTHORITY_KEYWORDS,
    EVIDENCE_KEYWORDS,
    HYPE_KEYWORDS,
    RISK_KEYWORDS,
    TECHNICAL_KEYWORDS,
)
from src.scoring.schemas import CompellingnessScore, DimensionScores, ThesisInput

logger = logging.getLogger(__name__)

# Minimum text length for meaningful scoring
MIN_TEXT_LENGTH = 50

# Maximum items to include in thesis text aggregation
MAX_KEYWORDS = 20
MAX_TICKERS = 10
MAX_ENTITIES = 15


class CompellingnessService:
    """Three-tier compellingness scoring for themes.

    Constructor: ``(config?, redis_client?)`` — matches embedding/sentiment pattern.

    Methods:
      - ``build_thesis_text(theme)`` — aggregate theme fields into scorable text
      - ``score_rule_based(thesis_text)`` — Tier 1 keyword density scoring
      - ``score_theme(theme)`` — full pipeline (Tier 1 → 2 → 3)
      - ``score_themes_batch(themes)`` — batch with per-theme error isolation
      - ``get_stats()`` — budget and circuit breaker status
      - ``close()`` — cleanup

    Args:
        config: Scoring configuration. Defaults to ScoringConfig().
        redis_client: Optional async Redis client for caching and budget tracking.
    """

    def __init__(
        self,
        config: ScoringConfig | None = None,
        redis_client: Any = None,
    ) -> None:
        self._config = config or ScoringConfig()
        self._redis = redis_client
        self._llm_client: LLMClient | None = None
        # In-memory budget fallback when Redis is unavailable
        self._memory_budgets: dict[str, float] = {}
        self._stats = {
            "total_scored": 0,
            "tier1_only": 0,
            "tier2_used": 0,
            "tier3_used": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    def _get_llm_client(self) -> LLMClient:
        """Lazy-initialize the LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient(self._config)
        return self._llm_client

    # ── Thesis Building ──────────────────────────────────

    def build_thesis_text(self, theme: Any) -> str:
        """Aggregate theme fields into a single text for scoring.

        Uses description, keywords, tickers, entities, and lifecycle stage.
        Does NOT fetch documents — theme metadata is sufficient.

        Args:
            theme: A Theme dataclass instance.

        Returns:
            Aggregated text suitable for scoring.
        """
        parts: list[str] = []

        if theme.description:
            parts.append(theme.description)

        if theme.top_keywords:
            keywords = theme.top_keywords[:MAX_KEYWORDS]
            parts.append(f"Keywords: {', '.join(keywords)}")

        if theme.top_tickers:
            tickers = theme.top_tickers[:MAX_TICKERS]
            parts.append(f"Tickers: {', '.join(tickers)}")

        if theme.top_entities:
            entity_names = []
            for ent in theme.top_entities[:MAX_ENTITIES]:
                if isinstance(ent, dict):
                    name = ent.get("normalized", ent.get("name", ""))
                    if name:
                        entity_names.append(name)
            if entity_names:
                parts.append(f"Entities: {', '.join(entity_names)}")

        parts.append(f"Lifecycle: {theme.lifecycle_stage}")
        parts.append(f"Documents: {theme.document_count}")

        return "\n".join(parts)

    def _build_context(self, theme: Any) -> dict[str, Any]:
        """Build context dict for LLM prompt formatting."""
        return {
            "tickers": theme.top_tickers[:MAX_TICKERS],
            "keywords": theme.top_keywords[:MAX_KEYWORDS],
            "lifecycle_stage": theme.lifecycle_stage,
            "document_count": theme.document_count,
        }

    # ── Content Hashing ──────────────────────────────────

    def _content_hash(self, thesis_text: str) -> str:
        """Compute deterministic hash for cache key.

        Uses SHA-256 truncated to 32 chars, matching the embedding/sentiment
        content-hash caching pattern.
        """
        return hashlib.sha256(thesis_text.encode("utf-8")).hexdigest()[:32]

    # ── Tier 1: Rule-Based Scoring ───────────────────────

    def score_rule_based(self, thesis_text: str) -> CompellingnessScore:
        """Score using keyword density analysis (Tier 1, free).

        Counts keyword matches per dimension, normalizes by text length,
        and detects quality flags (hype language, missing evidence, too short).

        Args:
            thesis_text: Aggregated theme text to score.

        Returns:
            CompellingnessScore with tier_used="rule".
        """
        text_lower = thesis_text.lower()
        words = re.findall(r"\w+", text_lower)
        word_count = max(len(words), 1)  # Avoid division by zero

        flags: list[str] = []

        # Check minimum length
        if len(thesis_text) < MIN_TEXT_LENGTH:
            flags.append("too_short")
            return CompellingnessScore(
                overall_score=0.0,
                dimensions=DimensionScores(),
                summary="Text too short for meaningful analysis",
                flags=flags,
                tier_used="rule",
                model_version="rule_v1",
            )

        # Count keyword matches per dimension
        def _density(keyword_set: frozenset[str]) -> float:
            count = sum(1 for kw in keyword_set if kw in text_lower)
            return count / word_count

        authority_density = _density(AUTHORITY_KEYWORDS)
        evidence_density = _density(EVIDENCE_KEYWORDS)
        hype_density = _density(HYPE_KEYWORDS)
        risk_density = _density(RISK_KEYWORDS)
        action_density = _density(ACTION_KEYWORDS)
        technical_density = _density(TECHNICAL_KEYWORDS)

        # Scale densities to 0-10 scores (calibrated for typical theme texts)
        # Multiplier chosen so that ~5% keyword density ≈ score of 7
        scale = 140.0
        authority_score = min(10.0, authority_density * scale)
        evidence_score = min(10.0, evidence_density * scale)
        risk_score = min(10.0, risk_density * scale)
        action_score = min(10.0, action_density * scale)
        technical_score = min(10.0, technical_density * scale)

        # Reasoning is harder to detect with keywords; use a blend
        reasoning_score = min(10.0, (evidence_density + risk_density) * scale * 0.5)

        # Flag detection
        if hype_density > 0.01:
            flags.append("hype_language")
        if evidence_density < 0.005:
            flags.append("no_evidence")

        # Hype penalty: reduce overall score when hype language is detected
        hype_penalty = min(3.0, hype_density * scale * 0.5)

        dimensions = DimensionScores(
            authority=round(authority_score, 2),
            evidence=round(evidence_score, 2),
            reasoning=round(reasoning_score, 2),
            risk=round(risk_score, 2),
            actionability=round(action_score, 2),
            technical=round(technical_score, 2),
        )

        overall = max(0.0, dimensions.mean - hype_penalty)
        overall = round(min(10.0, overall), 2)

        return CompellingnessScore(
            overall_score=overall,
            dimensions=dimensions,
            summary=f"Rule-based scoring: {word_count} words analyzed, {len(flags)} flags",
            flags=flags,
            tier_used="rule",
            model_version="rule_v1",
        )

    # ── Budget Tracking ──────────────────────────────────

    async def _check_budget(self, provider: str) -> bool:
        """Check if the daily budget allows another API call.

        Uses Redis INCRBYFLOAT when available, falls back to in-memory tracking.
        Estimates ~$0.01 per call for simplicity.
        """
        limit = (
            self._config.daily_budget_openai
            if provider == "openai"
            else self._config.daily_budget_anthropic
        )
        today = date.today().isoformat()
        key = f"{self._config.cache_key_prefix}budget:{provider}:{today}"

        if self._redis:
            try:
                current = await self._redis.get(key)
                current_val = float(current) if current else 0.0
                return current_val < limit
            except Exception:
                pass

        # In-memory fallback
        mem_key = f"{provider}:{today}"
        current_val = self._memory_budgets.get(mem_key, 0.0)
        return current_val < limit

    async def _record_spend(self, provider: str, amount: float = 0.01) -> None:
        """Record API spend against the daily budget."""
        today = date.today().isoformat()
        key = f"{self._config.cache_key_prefix}budget:{provider}:{today}"

        if self._redis:
            try:
                await self._redis.incrbyfloat(key, amount)
                # Set 48h TTL if this is a new key
                ttl = await self._redis.ttl(key)
                if ttl < 0:
                    await self._redis.expire(key, 172800)  # 48 hours
                return
            except Exception:
                pass

        # In-memory fallback
        mem_key = f"{provider}:{today}"
        self._memory_budgets[mem_key] = self._memory_budgets.get(mem_key, 0.0) + amount

    # ── Caching ──────────────────────────────────────────

    async def _get_cached(self, content_hash: str) -> CompellingnessScore | None:
        """Retrieve a cached score by content hash."""
        if not self._config.cache_enabled or not self._redis:
            return None

        key = f"{self._config.cache_key_prefix}{content_hash}"
        try:
            raw = await self._redis.get(key)
            if raw:
                data = json.loads(raw)
                score = CompellingnessScore(**data)
                score.cached = True
                self._stats["cache_hits"] += 1
                return score
        except Exception:
            logger.debug("Cache read failed for %s", content_hash)
        return None

    async def _set_cached(self, content_hash: str, score: CompellingnessScore) -> None:
        """Store a score in cache."""
        if not self._config.cache_enabled or not self._redis:
            return

        key = f"{self._config.cache_key_prefix}{content_hash}"
        try:
            data = score.model_dump(mode="json")
            await self._redis.set(
                key,
                json.dumps(data),
                ex=self._config.cache_ttl_seconds,
            )
        except Exception:
            logger.debug("Cache write failed for %s", content_hash)

    # ── Main Pipeline ────────────────────────────────────

    async def score_theme(self, theme: Any) -> CompellingnessScore:
        """Score a theme through the three-tier pipeline.

        Pipeline:
          1. Build thesis text → check cache
          2. Tier 1: rule-based scoring (always)
          3. If rule_score < tier2_min → return rule score
          4. Tier 2: GPT-4o-mini (if budget + circuit allow)
          5. If gpt_score < tier3_min → return gpt score
          6. Tier 3: Claude validation (if budget + circuit allow)
          7. Consensus check → cache → return

        Falls back to the best available tier on any failure.

        Args:
            theme: A Theme dataclass instance.

        Returns:
            CompellingnessScore (always succeeds; worst case = rule-based).
        """
        self._stats["total_scored"] += 1

        thesis_text = self.build_thesis_text(theme)
        content_hash = self._content_hash(thesis_text)
        context = self._build_context(theme)

        # Check cache
        cached = await self._get_cached(content_hash)
        if cached is not None:
            return cached

        # Tier 1: Rule-based
        rule_score = self.score_rule_based(thesis_text)

        if rule_score.overall_score < self._config.tier2_min_rule_score:
            self._stats["tier1_only"] += 1
            await self._set_cached(content_hash, rule_score)
            return rule_score

        # Tier 2: GPT-4o-mini
        gpt_score: CompellingnessScore | None = None
        if await self._check_budget("openai"):
            try:
                llm = self._get_llm_client()
                gpt_score = await llm.score_with_openai(thesis_text, context)
                if gpt_score is not None:
                    await self._record_spend("openai")
                    self._stats["tier2_used"] += 1
            except CircuitOpenError:
                logger.info("OpenAI circuit open, staying at Tier 1")
            except Exception as e:
                logger.warning("Tier 2 scoring failed: %s", e)
                self._stats["errors"] += 1

        # If Tier 2 failed, return rule score
        if gpt_score is None:
            self._stats["tier1_only"] += 1
            await self._set_cached(content_hash, rule_score)
            return rule_score

        if gpt_score.overall_score < self._config.tier3_min_gpt_score:
            await self._set_cached(content_hash, gpt_score)
            return gpt_score

        # Tier 3: Claude validation
        claude_score: CompellingnessScore | None = None
        if await self._check_budget("anthropic"):
            try:
                llm = self._get_llm_client()
                previous = {
                    "overall_score": gpt_score.overall_score,
                    "dimensions": gpt_score.dimensions.model_dump(),
                    "summary": gpt_score.summary,
                }
                claude_score = await llm.score_with_anthropic(
                    thesis_text, context, previous_scores=previous,
                )
                if claude_score is not None:
                    await self._record_spend("anthropic")
                    self._stats["tier3_used"] += 1
            except CircuitOpenError:
                logger.info("Anthropic circuit open, staying at Tier 2")
            except Exception as e:
                logger.warning("Tier 3 scoring failed: %s", e)
                self._stats["errors"] += 1

        # If Tier 3 failed, return GPT score
        if claude_score is None:
            await self._set_cached(content_hash, gpt_score)
            return gpt_score

        # Consensus check
        delta = abs(claude_score.overall_score - gpt_score.overall_score)
        if delta > self._config.consensus_tolerance:
            # Disagreement: average scores, flag for review
            avg_overall = (claude_score.overall_score + gpt_score.overall_score) / 2.0
            claude_score.overall_score = round(avg_overall, 2)
            if "needs_human_review" not in claude_score.flags:
                claude_score.flags.append("needs_human_review")
            claude_score.summary = (
                f"GPT/Claude disagreement (delta={delta:.1f}). "
                f"GPT={gpt_score.overall_score}, Claude={claude_score.overall_score}. "
                f"{claude_score.summary}"
            )

        await self._set_cached(content_hash, claude_score)
        return claude_score

    # ── Batch Scoring ────────────────────────────────────

    async def score_themes_batch(
        self, themes: list[Any],
    ) -> list[CompellingnessScore]:
        """Score multiple themes with per-theme error isolation.

        Each theme is scored independently. Failures fall back to rule-based
        scoring rather than propagating to other themes.

        Args:
            themes: List of Theme dataclass instances.

        Returns:
            List of CompellingnessScore, one per theme (same order).
        """
        results: list[CompellingnessScore] = []
        for theme in themes:
            try:
                score = await self.score_theme(theme)
                results.append(score)
            except Exception as e:
                logger.warning(
                    "Batch scoring failed for theme %s: %s",
                    getattr(theme, "theme_id", "unknown"),
                    e,
                )
                self._stats["errors"] += 1
                # Fallback to rule-based
                thesis_text = self.build_thesis_text(theme)
                fallback = self.score_rule_based(thesis_text)
                results.append(fallback)
        return results

    # ── Stats & Cleanup ──────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return scoring statistics and circuit breaker states."""
        llm = self._llm_client
        return {
            **self._stats,
            "openai_circuit": (
                llm.openai_breaker.state.value if llm else "not_initialized"
            ),
            "anthropic_circuit": (
                llm.anthropic_breaker.state.value if llm else "not_initialized"
            ),
            "memory_budgets": dict(self._memory_budgets),
        }

    async def close(self) -> None:
        """Clean up LLM clients."""
        if self._llm_client is not None:
            await self._llm_client.close()
            self._llm_client = None
