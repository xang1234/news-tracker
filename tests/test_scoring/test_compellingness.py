"""Tests for the CompellingnessService."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.scoring.circuit_breaker import CircuitOpenError, CircuitState
from src.scoring.compellingness import CompellingnessService
from src.scoring.config import ScoringConfig
from src.scoring.schemas import CompellingnessScore, DimensionScores
from src.themes.schemas import Theme


# ── Thesis Building ──────────────────────────────────────


class TestThesisBuilding:
    """Tests for build_thesis_text()."""

    def test_full_theme(self, scoring_config: ScoringConfig, sample_theme: Theme) -> None:
        service = CompellingnessService(config=scoring_config)
        text = service.build_thesis_text(sample_theme)

        # Should include all major components
        assert "NVIDIA" in text
        assert "chiplet" in text or "gpu" in text
        assert "NVDA" in text
        assert "emerging" in text
        assert "42" in text

    def test_minimal_theme(self, scoring_config: ScoringConfig, minimal_theme: Theme) -> None:
        service = CompellingnessService(config=scoring_config)
        text = service.build_thesis_text(minimal_theme)

        assert "fading" in text
        assert "Keywords:" not in text  # No keywords
        assert "Tickers:" not in text  # No tickers
        assert "Entities:" not in text  # No entities

    def test_includes_keywords_and_tickers(
        self, scoring_config: ScoringConfig, sample_theme: Theme,
    ) -> None:
        service = CompellingnessService(config=scoring_config)
        text = service.build_thesis_text(sample_theme)

        assert "Keywords:" in text
        assert "Tickers:" in text
        assert "AMD" in text
        assert "hbm" in text

    def test_truncates_long_lists(self, scoring_config: ScoringConfig, sample_centroid: np.ndarray) -> None:
        """Keywords/tickers/entities are capped at max limits."""
        theme = Theme(
            theme_id="theme_long_lists_00",
            name="many_keywords",
            centroid=sample_centroid,
            top_keywords=[f"kw{i}" for i in range(50)],
            top_tickers=[f"T{i}" for i in range(20)],
            lifecycle_stage="mature",
            document_count=100,
        )
        service = CompellingnessService(config=scoring_config)
        text = service.build_thesis_text(theme)

        # Should not include all 50 keywords
        assert "kw0" in text
        assert "kw49" not in text


# ── Rule-Based Scoring ───────────────────────────────────


class TestRuleBasedScoring:
    """Tests for score_rule_based()."""

    def test_evidence_rich_text_scores_higher(self, scoring_config: ScoringConfig) -> None:
        service = CompellingnessService(config=scoring_config)
        evidence_text = (
            "TSMC reported revenue growth of 25% YoY with capacity expansion. "
            "Analyst report from Bloomberg confirms foundry capex estimate of $30 billion. "
            "Wafer shipments increased 15% QoQ according to management guidance. "
            "Key risk is geopolitical uncertainty around export controls. "
            "Target price set by institutional research at $180 with medium-term horizon."
        )
        score = service.score_rule_based(evidence_text)

        assert score.overall_score > 2.0
        assert score.tier_used == "rule"
        assert score.dimensions.evidence > 0

    def test_hype_text_scores_lower(self, scoring_config: ScoringConfig) -> None:
        service = CompellingnessService(config=scoring_config)
        hype_text = (
            "NVDA to the moon! This is a no brainer, guaranteed 100x returns. "
            "Diamond hands, YOLO, easy money. Can't lose on this rocket ship. "
            "Trust me bro, this is a once in a lifetime opportunity. Free money! "
            "Parabolic gains incoming, insane upside potential, skyrocket guaranteed."
        )
        score = service.score_rule_based(hype_text)

        assert "hype_language" in score.flags
        # Hype penalty should suppress the score
        assert score.overall_score < 5.0

    def test_empty_text_returns_zero(self, scoring_config: ScoringConfig) -> None:
        service = CompellingnessService(config=scoring_config)
        score = service.score_rule_based("Hi.")

        assert score.overall_score == 0.0
        assert "too_short" in score.flags

    def test_flags_no_evidence(self, scoring_config: ScoringConfig) -> None:
        service = CompellingnessService(config=scoring_config)
        # Text with no evidence keywords but long enough (avoids AUTHORITY/EVIDENCE keywords)
        no_evidence = (
            "People are talking about things that happened recently. "
            "There are many opinions floating around online about this topic. "
            "Some folks think one way while others disagree completely. "
            "The conversation keeps going with more thoughts and ideas shared."
        )
        score = service.score_rule_based(no_evidence)

        assert "no_evidence" in score.flags

    def test_model_version_set(self, scoring_config: ScoringConfig) -> None:
        service = CompellingnessService(config=scoring_config)
        text = "A sufficiently long text about semiconductor technology and processes for scoring."
        score = service.score_rule_based(text)
        assert score.model_version == "rule_v1"


# ── Tiered Pipeline ──────────────────────────────────────


class TestTieredPipeline:
    """Tests for score_theme() tier gating logic."""

    async def test_low_rule_score_skips_llm(
        self, scoring_config: ScoringConfig, minimal_theme: Theme,
    ) -> None:
        """Themes scoring below tier2_min_rule_score stay at Tier 1."""
        service = CompellingnessService(config=scoring_config)
        score = await service.score_theme(minimal_theme)

        assert score.tier_used == "rule"
        assert service._stats["tier1_only"] >= 1

    async def test_tier2_gating(
        self, scoring_config: ScoringConfig, sample_theme: Theme, mock_gpt_score: CompellingnessScore,
    ) -> None:
        """Theme with high rule score advances to Tier 2, but GPT < tier3_min stays at Tier 2."""
        service = CompellingnessService(config=scoring_config)

        with patch.object(service, "_get_llm_client") as mock_llm:
            client = MagicMock()
            client.score_with_openai = AsyncMock(return_value=mock_gpt_score)
            mock_llm.return_value = client

            score = await service.score_theme(sample_theme)

        # GPT score is 7.5 < 8.5 (tier3_min), so should stop at Tier 2
        assert score.tier_used == "gpt"
        assert score.overall_score == 7.5

    async def test_tier3_consensus_pass(
        self,
        scoring_config: ScoringConfig,
        sample_theme: Theme,
        mock_gpt_score: CompellingnessScore,
        mock_claude_score: CompellingnessScore,
    ) -> None:
        """High GPT score triggers Tier 3; consensus within tolerance passes."""
        # Set GPT score above tier3 threshold
        mock_gpt_score.overall_score = 9.0
        # Claude score is 8.0, delta = 1.0 < 1.5 tolerance
        mock_claude_score.overall_score = 8.5

        service = CompellingnessService(config=scoring_config)

        with patch.object(service, "_get_llm_client") as mock_llm:
            client = MagicMock()
            client.score_with_openai = AsyncMock(return_value=mock_gpt_score)
            client.score_with_anthropic = AsyncMock(return_value=mock_claude_score)
            mock_llm.return_value = client

            score = await service.score_theme(sample_theme)

        assert score.tier_used == "claude"
        assert "needs_human_review" not in score.flags

    async def test_consensus_disagreement_flags_review(
        self,
        scoring_config: ScoringConfig,
        sample_theme: Theme,
        mock_gpt_score: CompellingnessScore,
        mock_claude_score: CompellingnessScore,
    ) -> None:
        """Large GPT-Claude disagreement flags for human review and averages scores."""
        mock_gpt_score.overall_score = 9.0
        mock_claude_score.overall_score = 6.0  # delta = 3.0 > 1.5

        service = CompellingnessService(config=scoring_config)

        with patch.object(service, "_get_llm_client") as mock_llm:
            client = MagicMock()
            client.score_with_openai = AsyncMock(return_value=mock_gpt_score)
            client.score_with_anthropic = AsyncMock(return_value=mock_claude_score)
            mock_llm.return_value = client

            score = await service.score_theme(sample_theme)

        assert "needs_human_review" in score.flags
        assert score.overall_score == 7.5  # (9.0 + 6.0) / 2

    async def test_llm_failure_falls_back_to_rule(
        self, scoring_config: ScoringConfig, sample_theme: Theme,
    ) -> None:
        """When LLM call fails, fallback to rule-based score."""
        service = CompellingnessService(config=scoring_config)

        with patch.object(service, "_get_llm_client") as mock_llm:
            client = MagicMock()
            client.score_with_openai = AsyncMock(side_effect=RuntimeError("API down"))
            mock_llm.return_value = client

            score = await service.score_theme(sample_theme)

        assert score.tier_used == "rule"


# ── Budget Tracking ──────────────────────────────────────


class TestBudgetTracking:
    """Tests for budget enforcement."""

    async def test_under_budget_allows_call(self, scoring_config: ScoringConfig) -> None:
        service = CompellingnessService(config=scoring_config)
        assert await service._check_budget("openai") is True

    async def test_over_budget_blocks_call(self, scoring_config: ScoringConfig) -> None:
        scoring_config.daily_budget_openai = 0.005  # Very low budget
        service = CompellingnessService(config=scoring_config)

        # Record enough spend to exceed budget
        await service._record_spend("openai", 0.01)

        assert await service._check_budget("openai") is False

    async def test_graceful_without_redis(self, scoring_config: ScoringConfig) -> None:
        """Budget tracking works with in-memory fallback when Redis is None."""
        service = CompellingnessService(config=scoring_config, redis_client=None)
        assert await service._check_budget("openai") is True

        await service._record_spend("openai", 0.01)
        # Should still track in memory
        assert service._memory_budgets

    async def test_auto_downgrade_on_budget_exhaustion(
        self, scoring_config: ScoringConfig, sample_theme: Theme,
    ) -> None:
        """Exhausted budget causes tier downgrade to rule-based."""
        scoring_config.daily_budget_openai = 0.0  # Zero budget
        service = CompellingnessService(config=scoring_config)

        score = await service.score_theme(sample_theme)

        assert score.tier_used == "rule"

    async def test_redis_budget_tracking(
        self, scoring_config: ScoringConfig, mock_redis: AsyncMock,
    ) -> None:
        """Budget check uses Redis when available."""
        scoring_config.daily_budget_openai = 5.0  # Set explicit limit for test
        scoring_config.cache_enabled = False  # Disable cache but keep Redis for budget
        service = CompellingnessService(config=scoring_config, redis_client=mock_redis)

        mock_redis.get.return_value = "4.99"  # Under $5 limit
        assert await service._check_budget("openai") is True

        mock_redis.get.return_value = "5.01"  # Over $5 limit
        assert await service._check_budget("openai") is False


# ── Caching ──────────────────────────────────────────────


class TestCaching:
    """Tests for content-hash caching."""

    def test_hash_determinism(self, scoring_config: ScoringConfig) -> None:
        """Same text always produces same hash."""
        service = CompellingnessService(config=scoring_config)
        h1 = service._content_hash("test content")
        h2 = service._content_hash("test content")
        assert h1 == h2
        assert len(h1) == 32

    def test_hash_differs_for_different_content(self, scoring_config: ScoringConfig) -> None:
        service = CompellingnessService(config=scoring_config)
        h1 = service._content_hash("content A")
        h2 = service._content_hash("content B")
        assert h1 != h2

    async def test_cache_hit(
        self, scoring_config: ScoringConfig, mock_redis: AsyncMock, sample_theme: Theme,
    ) -> None:
        """Cached score is returned without LLM calls."""
        scoring_config.cache_enabled = True
        service = CompellingnessService(config=scoring_config, redis_client=mock_redis)

        cached_data = CompellingnessScore(
            overall_score=7.0,
            tier_used="gpt",
            model_version="gpt-4o-mini",
        ).model_dump(mode="json")
        mock_redis.get.return_value = json.dumps(cached_data)

        score = await service.score_theme(sample_theme)

        assert score.cached is True
        assert score.overall_score == 7.0

    async def test_cache_disabled_skips(
        self, scoring_config: ScoringConfig, sample_theme: Theme,
    ) -> None:
        """With cache disabled, no Redis calls are made."""
        scoring_config.cache_enabled = False
        service = CompellingnessService(config=scoring_config, redis_client=None)

        score = await service.score_theme(sample_theme)
        # Should compute fresh (rule-based since no LLM configured)
        assert score.cached is False


# ── Circuit Breaker Integration ──────────────────────────


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker fallback in the pipeline."""

    async def test_open_circuit_falls_back_to_rule(
        self, scoring_config: ScoringConfig, sample_theme: Theme,
    ) -> None:
        """Open OpenAI circuit causes fallback to rule-based scoring."""
        service = CompellingnessService(config=scoring_config)

        with patch.object(service, "_get_llm_client") as mock_llm:
            client = MagicMock()
            client.score_with_openai = AsyncMock(side_effect=CircuitOpenError("OPEN"))
            client.openai_breaker = MagicMock()
            client.openai_breaker.state = CircuitState.OPEN
            mock_llm.return_value = client

            score = await service.score_theme(sample_theme)

        assert score.tier_used == "rule"


# ── Batch Scoring ────────────────────────────────────────


class TestBatchScoring:
    """Tests for score_themes_batch()."""

    async def test_batch_returns_one_per_theme(
        self, scoring_config: ScoringConfig, sample_theme: Theme, minimal_theme: Theme,
    ) -> None:
        service = CompellingnessService(config=scoring_config)
        results = await service.score_themes_batch([sample_theme, minimal_theme])
        assert len(results) == 2

    async def test_error_isolation(
        self, scoring_config: ScoringConfig, sample_theme: Theme, minimal_theme: Theme,
    ) -> None:
        """One theme failing doesn't affect others."""
        service = CompellingnessService(config=scoring_config)

        # Make score_theme raise for the first call, succeed for second
        call_count = 0
        original = service.score_theme

        async def _flaky(theme: Theme) -> CompellingnessScore:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient failure")
            return await original(theme)

        with patch.object(service, "score_theme", side_effect=_flaky):
            results = await service.score_themes_batch([sample_theme, minimal_theme])

        assert len(results) == 2
        # First should be fallback (rule-based)
        assert results[0].tier_used == "rule"


# ── Stats ────────────────────────────────────────────────


class TestStats:
    """Tests for get_stats()."""

    async def test_stats_after_scoring(
        self, scoring_config: ScoringConfig, sample_theme: Theme,
    ) -> None:
        service = CompellingnessService(config=scoring_config)
        await service.score_theme(sample_theme)

        stats = service.get_stats()
        assert stats["total_scored"] >= 1
        assert "openai_circuit" in stats

    async def test_close_cleans_up(self, scoring_config: ScoringConfig) -> None:
        service = CompellingnessService(config=scoring_config)
        await service.close()
        assert service._llm_client is None
