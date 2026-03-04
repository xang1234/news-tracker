"""
Twitter adapter for financial content ingestion.

Primary path: xui browser ingestion with adaptive guardrails.
Backup path: Twitter API v2 (when token is configured).

Guardrails implemented for xui:
- Randomized polling cadence
- Randomized source order with short source-cooldown ring
- Randomized inter-source pauses
- Scroll/page budget enforcement using xui runtime constraints
- Block/challenge detection with exponential backoff + circuit breaker
"""

import asyncio
import json
import logging
import os
import random
import re
import shlex
import shutil
import tempfile
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from src.config.settings import get_settings
from src.config.tickers import SEMICONDUCTOR_TICKERS
from src.config.twitter_accounts import parse_usernames
from src.ingestion.base_adapter import (
    BaseAdapter,
    clean_text,
    expand_twitter_abbreviations,
    extract_cashtags,
    translate_emoji_sentiment,
)
from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform

logger = logging.getLogger(__name__)

# Twitter API v2 endpoints
TWITTER_API_BASE = "https://api.twitter.com/2"
TWEETS_SEARCH_RECENT = f"{TWITTER_API_BASE}/tweets/search/recent"

_XUI_BLOCK_MARKERS = (
    "rate limit",
    "rate_limit",
    "challenge",
    "login_wall",
    "blocked_",
    "suspend",
    "missing storage_state",
    "missing_storage_state",
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class XuiGuardrailConfig:
    enabled: bool
    command: tuple[str, ...]
    config_path: Path
    config_path_explicit: bool
    profile: str
    usernames: tuple[str, ...]
    poll_min_seconds: int
    poll_max_seconds: int
    cycle_jitter_ratio: float
    limit_per_source: int
    scroll_pause_min_ms: int
    scroll_pause_max_ms: int
    max_scroll_rounds: int
    max_page_loads: int
    timeout_ms: int
    shuffle_sources: bool
    source_cooldown_cycles: int
    source_pause_min_seconds: float
    source_pause_max_seconds: float
    block_backoff_initial_seconds: int
    block_backoff_max_seconds: int
    block_circuit_threshold: int
    block_circuit_open_seconds: int


@dataclass
class XuiGuardrailState:
    consecutive_block_events: int = 0
    current_backoff_seconds: int = 0
    circuit_open_until: datetime | None = None
    recent_lead_sources: deque[str] = field(default_factory=deque)


@dataclass(frozen=True)
class XuiInvocationResult:
    return_code: int
    stdout: str
    stderr: str
    payload: dict[str, Any]

    @property
    def items(self) -> list[dict[str, Any]]:
        raw_items = self.payload.get("items", [])
        return [i for i in raw_items if isinstance(i, dict)] if isinstance(raw_items, list) else []

    @property
    def succeeded_sources(self) -> int:
        return _coerce_non_negative_int(self.payload.get("succeeded_sources"), default=0)

    @property
    def failed_sources(self) -> int:
        return _coerce_non_negative_int(self.payload.get("failed_sources"), default=0)

    @property
    def page_loads(self) -> int:
        return _coerce_non_negative_int(self.payload.get("page_loads"), default=0)

    @property
    def scroll_rounds(self) -> int:
        return _coerce_non_negative_int(self.payload.get("scroll_rounds"), default=0)


class TwitterAdapter(BaseAdapter):
    """
    Twitter adapter for fetching financial posts.

    Fetch order:
    1) xui browser ingestion (primary)
    2) Twitter API v2 (backup when xui is unavailable/blocked)

    xui guardrails are tuned to reduce bot-like traffic patterns and
    challenge/rate-limit risk.
    """

    def __init__(
        self,
        bearer_token: str | None = None,
        tickers: set[str] | None = None,
        rate_limit: int = 30,
        max_results_per_request: int = 100,
        xui_usernames: list[str] | None = None,
    ):
        super().__init__(rate_limit=rate_limit)

        settings = get_settings()
        self._bearer_token = bearer_token or settings.twitter_bearer_token
        self._tickers = tickers or SEMICONDUCTOR_TICKERS
        self._max_results = min(max_results_per_request, 100)

        self._xui = self._build_xui_guardrail_config(settings, xui_usernames)
        cooldown_len = max(1, self._xui.source_cooldown_cycles)
        self._xui_state = XuiGuardrailState(
            recent_lead_sources=deque(maxlen=cooldown_len)
        )
        self._rng = random.Random()

        # Track seen tweet IDs to avoid duplicates
        self._seen_tweet_ids: set[str] = set()
        self._twitter_api_unavailable = False

        if self._xui.enabled and not self._xui.usernames:
            logger.warning("xui is enabled but no usernames are configured")

        if not self._bearer_token:
            if self._xui.enabled:
                logger.info("Twitter API token missing; using xui primary ingestion")
            else:
                logger.warning(
                    "Twitter bearer token not configured and xui disabled. "
                    "Adapter will not fetch data."
                )

    @property
    def platform(self) -> Platform:
        return Platform.TWITTER

    def next_poll_delay_seconds(self, default_interval_seconds: float) -> float:
        """Adapter-specific jittered polling delay with block-aware backoff."""
        if not self._xui.enabled:
            return default_interval_seconds

        now = _utc_now()
        if self._xui_state.circuit_open_until and self._xui_state.circuit_open_until > now:
            return max(1.0, (self._xui_state.circuit_open_until - now).total_seconds())

        if self._xui_state.circuit_open_until and self._xui_state.circuit_open_until <= now:
            self._xui_state.circuit_open_until = None

        base_delay = self._compute_cycle_delay_seconds()
        if self._xui_state.current_backoff_seconds <= 0:
            return float(base_delay)

        backoff = self._xui_state.current_backoff_seconds
        backoff_jitter = max(1, int(round(backoff * self._xui.cycle_jitter_ratio)))
        jittered_backoff = self._rng.randint(backoff, backoff + backoff_jitter)
        return float(max(base_delay, jittered_backoff))

    def _build_query(self, tickers: list[str]) -> str:
        cashtag_query = " OR ".join(f"${t}" for t in tickers)
        return f"({cashtag_query}) -is:retweet lang:en"

    async def _fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch tweets via xui primary, with Twitter API fallback.
        """
        self._seen_tweet_ids.clear()
        self._twitter_api_unavailable = False

        xui_cycle_success = False
        if self._xui.enabled:
            raw_items, xui_cycle_success = await self._collect_xui_items()
            for item in raw_items:
                yield item

        if xui_cycle_success:
            return

        if self._bearer_token:
            async for item in self._fetch_twitter_api():
                yield item
            return

        if not self._xui.enabled:
            logger.error("Twitter bearer token not configured and xui disabled")
        else:
            logger.warning(
                "xui cycle did not complete successfully and no API token is configured"
            )

    async def _collect_xui_items(self) -> tuple[list[dict[str, Any]], bool]:
        """Run one guarded xui collection cycle and return raw items + success flag."""
        if self._is_xui_circuit_open():
            until = self._xui_state.circuit_open_until
            logger.warning(
                "Skipping xui cycle: circuit breaker is open",
                extra={"circuit_open_until": until.isoformat() if until else None},
            )
            return [], False

        ordered_sources = self._ordered_sources_for_cycle()
        if not ordered_sources:
            return [], False

        config_path, cleanup_path = self._prepare_xui_runtime_config()
        collected: list[dict[str, Any]] = []
        total_page_loads = 0
        total_scroll_rounds = 0
        succeeded_sources = 0
        block_detected = False

        try:
            for index, source in enumerate(ordered_sources):
                if total_page_loads >= self._xui.max_page_loads:
                    logger.info(
                        "xui page-load budget reached; ending cycle early",
                        extra={
                            "page_loads": total_page_loads,
                            "budget": self._xui.max_page_loads,
                        },
                    )
                    break

                result = await self._invoke_xui_read(source=source, config_path=config_path)

                total_page_loads += result.page_loads
                total_scroll_rounds += result.scroll_rounds
                succeeded_sources += result.succeeded_sources

                is_blocked, block_reason = self._detect_xui_block(result)
                if is_blocked:
                    self._register_xui_block(block_reason or "block_signal", result)
                    block_detected = True
                    break

                for item in result.items:
                    tweet_id = str(item.get("tweet_id", "")).strip()
                    if not tweet_id or tweet_id in self._seen_tweet_ids:
                        continue
                    self._seen_tweet_ids.add(tweet_id)
                    collected.append(
                        {
                            "source": "xui",
                            "tweet": item,
                            "username": source,
                        }
                    )

                if index < len(ordered_sources) - 1:
                    await asyncio.sleep(self._compute_source_pause_seconds())

        except Exception as exc:
            logger.error("xui collection cycle failed", extra={"error": str(exc)})
            return collected, False
        finally:
            if cleanup_path is not None:
                try:
                    cleanup_path.unlink(missing_ok=True)
                except Exception:
                    logger.debug("Failed to delete temporary xui config", exc_info=True)

        if block_detected:
            return collected, False

        success = succeeded_sources > 0
        if success:
            self._register_xui_success(
                succeeded_sources=succeeded_sources,
                page_loads=total_page_loads,
                scroll_rounds=total_scroll_rounds,
                emitted_items=len(collected),
            )
        else:
            logger.warning(
                "xui cycle produced no successful source reads",
                extra={
                    "page_loads": total_page_loads,
                    "scroll_rounds": total_scroll_rounds,
                },
            )

        return collected, success

    def _ordered_sources_for_cycle(self) -> list[str]:
        sources = list(self._xui.usernames)
        if not sources:
            return []

        if self._xui.shuffle_sources:
            self._rng.shuffle(sources)

        if self._xui.source_cooldown_cycles > 0 and len(sources) > 1:
            recent = set(self._xui_state.recent_lead_sources)
            primary = [source for source in sources if source not in recent]
            deferred = [source for source in sources if source in recent]
            if primary:
                sources = primary + deferred

        self._xui_state.recent_lead_sources.append(sources[0])
        return sources

    def _prepare_xui_runtime_config(self) -> tuple[Path | None, Path | None]:
        """
        Prepare a temporary runtime config with randomized scroll behavior.

        Returns:
            (config_path_for_command, temporary_file_to_delete)
        """
        base_config = self._xui.config_path
        scroll_pause_ms = self._rng.randint(
            self._xui.scroll_pause_min_ms,
            self._xui.scroll_pause_max_ms,
        )

        try:
            base_config.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Could not create xui config directory; using xui defaults without overrides",
                extra={"path": str(base_config.parent), "error": str(exc)},
            )
            return None, None

        if base_config.exists():
            try:
                original = base_config.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "Could not read xui config; using synthesized runtime config",
                    extra={"path": str(base_config), "error": str(exc)},
                )
                original = _default_xui_runtime_config_toml(self._xui.profile)
        else:
            original = _default_xui_runtime_config_toml(self._xui.profile)

        overridden = original
        overridden = _upsert_toml_value(
            overridden,
            section="collection",
            key="scroll_delay_ms",
            value=scroll_pause_ms,
        )
        overridden = _upsert_toml_value(
            overridden,
            section="collection",
            key="max_scrolls",
            value=self._xui.max_scroll_rounds,
        )
        overridden = _upsert_toml_value(
            overridden,
            section="search",
            key="scroll_pause_ms",
            value=scroll_pause_ms,
        )
        overridden = _upsert_toml_value(
            overridden,
            section="search",
            key="max_scrolls",
            value=self._xui.max_scroll_rounds,
        )
        overridden = _upsert_toml_value(
            overridden,
            section="browser",
            key="navigation_timeout_ms",
            value=self._xui.timeout_ms,
        )
        overridden = _upsert_toml_value(
            overridden,
            section="browser",
            key="action_timeout_ms",
            value=min(self._xui.timeout_ms, 30_000),
        )

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".toml",
                prefix="news-tracker-xui-",
                dir=base_config.parent,
                delete=False,
            ) as tmp:
                tmp.write(overridden)
            temp_path = Path(tmp.name)
            return temp_path, temp_path
        except OSError as exc:
            logger.warning(
                "Could not create temporary xui config; using base/default config path",
                extra={"path": str(base_config), "error": str(exc)},
            )
            return (base_config if base_config.exists() else None), None

    async def _invoke_xui_read(
        self,
        *,
        source: str,
        config_path: Path | None,
    ) -> XuiInvocationResult:
        command = list(self._xui.command)
        command.extend(
            [
                "--timeout-ms",
                str(self._xui.timeout_ms),
            ]
        )
        command.extend(
            [
                "read",
                "--profile",
                self._xui.profile,
                "--limit",
                str(self._xui.limit_per_source),
                "--checkpoint-mode",
                "auto",
                "--new",
                "--json",
                "--sources",
                f"user:{source}",
            ]
        )
        if config_path is not None:
            command.extend(["--path", str(config_path)])

        logger.debug("Invoking xui", extra={"command": command, "source": source})

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=180)
            return_code = int(process.returncode or 0)
        except TimeoutError:
            process.kill()
            await process.communicate()
            return XuiInvocationResult(
                return_code=124,
                stdout="",
                stderr="xui invocation timed out",
                payload={},
            )

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        payload = _extract_json_payload(stdout_text)

        if return_code != 0:
            logger.warning(
                "xui read exited non-zero",
                extra={
                    "return_code": return_code,
                    "source": source,
                    "stderr": stderr_text.strip()[:400],
                },
            )

        return XuiInvocationResult(
            return_code=return_code,
            stdout=stdout_text,
            stderr=stderr_text,
            payload=payload,
        )

    def _detect_xui_block(self, result: XuiInvocationResult) -> tuple[bool, str | None]:
        candidate_messages: list[str] = []
        outcomes = result.payload.get("outcomes", [])
        if isinstance(outcomes, list):
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                error_msg = outcome.get("error")
                if isinstance(error_msg, str) and error_msg.strip():
                    candidate_messages.append(error_msg)

        payload_error = result.payload.get("error")
        if isinstance(payload_error, str) and payload_error.strip():
            candidate_messages.append(payload_error)

        if result.stderr.strip():
            candidate_messages.append(result.stderr)

        for message in candidate_messages:
            marker = _find_block_marker(message)
            if marker is not None:
                return True, marker

        return False, None

    def _register_xui_block(
        self,
        reason: str,
        result: XuiInvocationResult,
    ) -> None:
        self._xui_state.consecutive_block_events += 1
        attempts = self._xui_state.consecutive_block_events

        backoff = self._xui.block_backoff_initial_seconds * (2 ** (attempts - 1))
        self._xui_state.current_backoff_seconds = min(
            backoff,
            self._xui.block_backoff_max_seconds,
        )

        circuit_opened = False
        if attempts >= self._xui.block_circuit_threshold:
            self._xui_state.circuit_open_until = _utc_now() + timedelta(
                seconds=self._xui.block_circuit_open_seconds
            )
            circuit_opened = True

        logger.warning(
            "xui block/challenge detected",
            extra={
                "reason": reason,
                "attempts": attempts,
                "backoff_seconds": self._xui_state.current_backoff_seconds,
                "circuit_opened": circuit_opened,
                "circuit_open_until": (
                    self._xui_state.circuit_open_until.isoformat()
                    if self._xui_state.circuit_open_until
                    else None
                ),
                "return_code": result.return_code,
            },
        )

    def _register_xui_success(
        self,
        *,
        succeeded_sources: int,
        page_loads: int,
        scroll_rounds: int,
        emitted_items: int,
    ) -> None:
        had_guardrail_state = (
            self._xui_state.consecutive_block_events > 0
            or self._xui_state.current_backoff_seconds > 0
            or self._xui_state.circuit_open_until is not None
        )

        self._xui_state.consecutive_block_events = 0
        self._xui_state.current_backoff_seconds = 0
        self._xui_state.circuit_open_until = None

        logger.info(
            "xui cycle completed",
            extra={
                "succeeded_sources": succeeded_sources,
                "page_loads": page_loads,
                "scroll_rounds": scroll_rounds,
                "emitted_items": emitted_items,
                "cleared_guardrail_state": had_guardrail_state,
            },
        )

    def _is_xui_circuit_open(self) -> bool:
        until = self._xui_state.circuit_open_until
        if until is None:
            return False
        if until > _utc_now():
            return True
        self._xui_state.circuit_open_until = None
        return False

    def _compute_cycle_delay_seconds(self) -> int:
        base = self._rng.uniform(
            float(self._xui.poll_min_seconds),
            float(self._xui.poll_max_seconds),
        )
        spread = base * self._xui.cycle_jitter_ratio
        jittered = base + self._rng.uniform(-spread, spread)
        lower = float(self._xui.poll_min_seconds)
        upper = float(self._xui.poll_max_seconds)
        return int(round(max(lower, min(upper, jittered))))

    def _compute_source_pause_seconds(self) -> float:
        return self._rng.uniform(
            self._xui.source_pause_min_seconds,
            self._xui.source_pause_max_seconds,
        )

    @staticmethod
    def _build_xui_guardrail_config(
        settings,
        xui_usernames: list[str] | None,
    ) -> XuiGuardrailConfig:
        command_tokens = tuple(shlex.split(settings.twitter_xui_command or "xui")) or ("xui",)

        configured_usernames = (
            xui_usernames
            if xui_usernames is not None
            else parse_usernames(settings.twitter_xui_usernames)
        )
        usernames = tuple(u.lstrip("@").strip() for u in configured_usernames if u.strip())

        config_path, config_path_explicit = _resolve_xui_config_path(
            settings.twitter_xui_config_path
        )

        poll_min = min(
            settings.twitter_xui_poll_min_seconds,
            settings.twitter_xui_poll_max_seconds,
        )
        poll_max = max(
            settings.twitter_xui_poll_min_seconds,
            settings.twitter_xui_poll_max_seconds,
        )

        scroll_min = min(
            settings.twitter_xui_scroll_pause_min_ms,
            settings.twitter_xui_scroll_pause_max_ms,
        )
        scroll_max = max(
            settings.twitter_xui_scroll_pause_min_ms,
            settings.twitter_xui_scroll_pause_max_ms,
        )

        source_pause_min = min(
            settings.twitter_xui_source_pause_min_seconds,
            settings.twitter_xui_source_pause_max_seconds,
        )
        source_pause_max = max(
            settings.twitter_xui_source_pause_min_seconds,
            settings.twitter_xui_source_pause_max_seconds,
        )

        return XuiGuardrailConfig(
            enabled=bool(settings.twitter_xui_enabled),
            command=command_tokens,
            config_path=config_path,
            config_path_explicit=config_path_explicit,
            profile=settings.twitter_xui_profile,
            usernames=usernames,
            poll_min_seconds=poll_min,
            poll_max_seconds=poll_max,
            cycle_jitter_ratio=settings.twitter_xui_cycle_jitter_ratio,
            limit_per_source=settings.twitter_xui_limit_per_source,
            scroll_pause_min_ms=scroll_min,
            scroll_pause_max_ms=scroll_max,
            max_scroll_rounds=settings.twitter_xui_max_scroll_rounds,
            max_page_loads=settings.twitter_xui_max_page_loads,
            timeout_ms=settings.twitter_xui_timeout_ms,
            shuffle_sources=settings.twitter_xui_shuffle_sources,
            source_cooldown_cycles=settings.twitter_xui_source_cooldown_cycles,
            source_pause_min_seconds=source_pause_min,
            source_pause_max_seconds=source_pause_max,
            block_backoff_initial_seconds=settings.twitter_xui_block_backoff_initial_seconds,
            block_backoff_max_seconds=settings.twitter_xui_block_backoff_max_seconds,
            block_circuit_threshold=settings.twitter_xui_block_circuit_threshold,
            block_circuit_open_seconds=settings.twitter_xui_block_circuit_open_seconds,
        )

    async def _fetch_twitter_api(self) -> AsyncIterator[dict[str, Any]]:
        """Fetch tweets from Twitter API v2."""
        headers = {
            "Authorization": f"Bearer {self._bearer_token}",
            "User-Agent": "NewsTracker/1.0",
        }

        ticker_list = list(self._tickers)
        batch_size = 10

        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(0, len(ticker_list), batch_size):
                batch = ticker_list[i: i + batch_size]
                query = self._build_query(batch)

                params = {
                    "query": query,
                    "max_results": self._max_results,
                    "tweet.fields": "created_at,public_metrics,author_id",
                    "expansions": "author_id",
                    "user.fields": "name,username,public_metrics,verified",
                }

                next_token = None
                pages_fetched = 0
                max_pages = 3

                while pages_fetched < max_pages:
                    if next_token:
                        params["next_token"] = next_token

                    try:
                        await self._rate_limiter.acquire()

                        response = await client.get(
                            TWEETS_SEARCH_RECENT,
                            headers=headers,
                            params=params,
                        )

                        if response.status_code == 429:
                            logger.warning("Twitter API rate limit hit")
                            return

                        response.raise_for_status()
                        data = response.json()

                    except httpx.HTTPStatusError as exc:
                        status_code = exc.response.status_code
                        logger.error("Twitter API error", extra={"status_code": status_code})
                        if status_code in (401, 403):
                            self._twitter_api_unavailable = True
                            return
                        continue

                    except Exception as exc:
                        logger.error("Twitter API request failed", extra={"error": str(exc)})
                        continue

                    authors: dict[str, dict[str, Any]] = {}
                    includes = data.get("includes", {})
                    users = includes.get("users", []) if isinstance(includes, dict) else []
                    for user in users:
                        if isinstance(user, dict) and "id" in user:
                            authors[str(user["id"])] = user

                    tweets = data.get("data", [])
                    for tweet in tweets:
                        if not isinstance(tweet, dict):
                            continue
                        tweet_id = str(tweet.get("id", "")).strip()
                        if not tweet_id or tweet_id in self._seen_tweet_ids:
                            continue
                        self._seen_tweet_ids.add(tweet_id)

                        author = authors.get(str(tweet.get("author_id", "")), {})
                        yield {
                            "source": "twitter_api",
                            "tweet": tweet,
                            "author": author,
                        }

                    meta = data.get("meta", {})
                    next_token = meta.get("next_token") if isinstance(meta, dict) else None
                    pages_fetched += 1
                    if not next_token:
                        break

    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        source = raw.get("source", "twitter_api")
        if source == "xui":
            return self._transform_xui(raw)
        return self._transform_twitter_api(raw)

    def _transform_twitter_api(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        try:
            tweet = raw["tweet"]
            author = raw.get("author", {})

            created_at = tweet.get("created_at", "")
            if created_at:
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                timestamp = _utc_now()

            content = tweet.get("text", "")
            if not content:
                return None

            content = translate_emoji_sentiment(content)
            content = expand_twitter_abbreviations(content)
            content = clean_text(content)

            tickers = extract_cashtags(tweet.get("text", ""))

            metrics = tweet.get("public_metrics", {})
            engagement = EngagementMetrics(
                likes=_coerce_non_negative_int(metrics.get("like_count"), default=0),
                shares=_coerce_non_negative_int(metrics.get("retweet_count"), default=0),
                comments=_coerce_non_negative_int(metrics.get("reply_count"), default=0),
                views=_coerce_optional_non_negative_int(metrics.get("impression_count")),
            )

            author_metrics = author.get("public_metrics", {})
            author_followers = _coerce_optional_non_negative_int(
                author_metrics.get("followers_count")
            )

            tweet_id = str(tweet.get("id", "")).strip()
            if not tweet_id:
                return None

            return NormalizedDocument(
                id=f"twitter_{tweet_id}",
                platform=Platform.TWITTER,
                url=f"https://twitter.com/i/status/{tweet_id}",
                timestamp=timestamp,
                author_id=str(tweet.get("author_id", "")),
                author_name=author.get("username", "unknown"),
                author_followers=author_followers,
                author_verified=bool(author.get("verified", False)),
                content=content,
                content_type="post",
                engagement=engagement,
                tickers_mentioned=tickers,
                raw_data={
                    "source": "twitter_api",
                    "ingestion_method": "api",
                },
            )

        except Exception as exc:
            logger.debug("Failed to transform Twitter API tweet", extra={"error": str(exc)})
            return None

    def _transform_xui(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        try:
            tweet = raw.get("tweet", {})
            if not isinstance(tweet, dict):
                return None

            tweet_id = str(tweet.get("tweet_id", "")).strip()
            if not tweet_id:
                return None

            content = str(tweet.get("text", "")).strip()
            if not content:
                return None

            timestamp = _parse_iso_utc(tweet.get("created_at")) or _utc_now()

            handle = str(
                tweet.get("author_handle") or raw.get("username") or ""
            ).strip().lstrip("@")
            display_name = str(tweet.get("author_display_name") or handle or "unknown").strip()

            processed = translate_emoji_sentiment(content)
            processed = expand_twitter_abbreviations(processed)
            processed = clean_text(processed)

            tickers = extract_cashtags(content)

            engagement = EngagementMetrics(
                likes=_coerce_non_negative_int(tweet.get("likes"), default=0),
                shares=_coerce_non_negative_int(tweet.get("retweets"), default=0),
                comments=_coerce_non_negative_int(tweet.get("replies"), default=0),
                views=_coerce_optional_non_negative_int(tweet.get("views")),
            )

            if handle:
                url = f"https://x.com/{handle}/status/{tweet_id}"
                author_id = handle
            else:
                url = f"https://twitter.com/i/status/{tweet_id}"
                author_id = "unknown"

            return NormalizedDocument(
                id=f"twitter_{tweet_id}",
                platform=Platform.TWITTER,
                url=url,
                timestamp=timestamp,
                author_id=author_id,
                author_name=display_name,
                author_followers=None,
                author_verified=bool(tweet.get("author_verified", False)),
                content=processed,
                content_type="post",
                engagement=engagement,
                tickers_mentioned=tickers,
                raw_data={
                    "source": "xui",
                    "ingestion_method": "xui",
                    "source_id": tweet.get("source_id"),
                    "extraction_method": tweet.get("extraction_method"),
                    "quality_tier": tweet.get("quality_tier"),
                    "quality_score": tweet.get("quality_score"),
                },
            )

        except Exception as exc:
            logger.debug("Failed to transform xui tweet", extra={"error": str(exc)})
            return None

    async def health_check(self) -> bool:
        """
        Check if Twitter ingestion is reachable via xui or API.
        """
        xui_ok = self._xui_runtime_healthy()
        circuit_open = self._is_xui_circuit_open()
        if xui_ok and not circuit_open:
            return True

        if self._bearer_token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"{TWITTER_API_BASE}/users/me",
                        headers={"Authorization": f"Bearer {self._bearer_token}"},
                    )
                    if response.status_code in (200, 401, 403):
                        return True
            except Exception:
                pass

        return False

    def _xui_runtime_healthy(self) -> bool:
        if not self._xui.enabled:
            return False
        if not self._xui.usernames:
            return False

        executable = self._xui.command[0]
        if "/" in executable:
            if not Path(executable).expanduser().exists():
                return False
        elif shutil.which(executable) is None:
            return False

        if self._xui.config_path_explicit and not self._xui.config_path.exists():
            logger.warning(
                "Configured xui config path does not exist",
                extra={"path": str(self._xui.config_path)},
            )
        return True


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else None
    except (TypeError, ValueError):
        return None


def _parse_iso_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError):
        return None


def _extract_json_payload(stdout_text: str) -> dict[str, Any]:
    raw = stdout_text.strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return {}

    snippet = raw[start:end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _find_block_marker(text: str) -> str | None:
    normalized = text.lower()
    if re.search(r"(?<!\d)429(?!\d)", normalized):
        return "429"
    if re.search(r"(?<!\d)403(?!\d)", normalized):
        return "403"
    for marker in _XUI_BLOCK_MARKERS:
        if marker in normalized:
            return marker
    return None


def _resolve_xui_config_path(configured_value: str | None) -> tuple[Path, bool]:
    explicit_value = configured_value or os.getenv("XUI_CONFIG")
    if explicit_value:
        return Path(explicit_value).expanduser(), True
    try:
        from platformdirs import user_config_dir

        base_dir = Path(user_config_dir("xui-reader", appauthor=False))
    except ModuleNotFoundError:
        base_dir = Path.home() / ".config" / "xui-reader"
    return base_dir / "config.toml", False


def _default_xui_runtime_config_toml(profile: str) -> str:
    safe_profile = profile.replace("\\", "\\\\").replace('"', '\\"')
    return (
        "[app]\\n"
        f'default_profile = "{safe_profile}"\\n'
        'default_format = "json"\\n\\n'
        "[collection]\\n"
        "max_scrolls = 10\\n"
        "scroll_delay_ms = 1250\\n\\n"
        "[search]\\n"
        "scroll_pause_ms = 1500\\n"
        "max_scrolls = 12\\n"
    )


def _upsert_toml_value(
    text: str,
    *,
    section: str,
    key: str,
    value: int | float | bool | str,
) -> str:
    lines = text.splitlines()
    had_trailing_newline = text.endswith("\n")
    section_header = f"[{section}]"

    section_start = None
    section_end = len(lines)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section_header:
            section_start = idx
            continue
        if section_start is not None and stripped.startswith("[") and stripped.endswith("]"):
            section_end = idx
            break

    rendered_value = _render_toml_value(value)

    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(section_header)
        lines.append(f"{key} = {rendered_value}")
    else:
        replaced = False
        for idx in range(section_start + 1, section_end):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            existing_key = stripped.split("=", 1)[0].strip()
            if existing_key == key:
                prefix = lines[idx].split("=", 1)[0].rstrip()
                lines[idx] = f"{prefix} = {rendered_value}"
                replaced = True
                break
        if not replaced:
            lines.insert(section_end, f"{key} = {rendered_value}")

    output = "\n".join(lines)
    if had_trailing_newline:
        output += "\n"
    return output


def _render_toml_value(value: int | float | bool | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
