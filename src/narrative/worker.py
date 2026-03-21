"""Narrative worker for run assignment and real-time alerts."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import redis.asyncio as redis

from src.alerts.config import AlertConfig
from src.alerts.repository import AlertRepository
from src.alerts.schemas import Alert
from src.alerts.service import AlertService
from src.config.settings import get_settings
from src.narrative.config import NarrativeConfig
from src.narrative.queue import NarrativeJob, NarrativeQueue
from src.narrative.repository import NarrativeRepository, _row_to_bucket, _row_to_run
from src.narrative.schemas import NarrativeRun, NarrativeSignalState
from src.narrative.signals import SignalEvaluation, evaluate_all_signals
from src.observability.metrics import get_metrics
from src.queues.backoff import ExponentialBackoff
from src.storage.database import Database
from src.storage.repository import DocumentRepository
from src.themes.repository import ThemeRepository

logger = logging.getLogger(__name__)


class NarrativeWorker:
    """Consumes narrative jobs and updates narrative run state."""

    def __init__(
        self,
        queue: NarrativeQueue | None = None,
        database: Database | None = None,
        config: NarrativeConfig | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._config = config or NarrativeConfig()
        self._queue = queue or NarrativeQueue(config=self._config)
        self._database = database or Database()
        self._batch_size = batch_size or self._config.batch_size

        self._doc_repo: DocumentRepository | None = None
        self._theme_repo: ThemeRepository | None = None
        self._narrative_repo: NarrativeRepository | None = None
        self._alert_repo: AlertRepository | None = None
        self._alert_service: AlertService | None = None
        self._redis: redis.Redis | None = None
        self._running = False
        self._metrics = get_metrics()

    async def _connect_dependencies(self) -> None:
        settings = get_settings()
        await self._queue.connect()
        await self._database.connect()
        self._redis = redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )
        self._doc_repo = DocumentRepository(self._database)
        self._theme_repo = ThemeRepository(self._database)
        self._narrative_repo = NarrativeRepository(self._database)
        self._alert_repo = AlertRepository(self._database)
        self._alert_service = AlertService(
            config=AlertConfig(),
            alert_repo=self._alert_repo,
            redis_client=self._redis,
            dispatcher=None,
        )

    async def start(self) -> None:
        self._running = True
        settings = get_settings()
        backoff = ExponentialBackoff(
            base_delay=settings.worker_backoff_base_delay,
            max_delay=settings.worker_backoff_max_delay,
        )
        logger.info("Starting narrative worker")
        maintenance_task: asyncio.Task | None = None
        while self._running:
            try:
                await self._connect_dependencies()
                maintenance_task = asyncio.create_task(self._maintenance_loop())
                await self._process_loop()
                if not self._running:
                    break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                delay = backoff.next_delay()
                logger.warning("Narrative worker error: %s; retrying in %.1fs", exc, delay)
                await self._cleanup()
                await asyncio.sleep(delay)
            else:
                backoff.reset()
            finally:
                if maintenance_task:
                    maintenance_task.cancel()
                    try:
                        await maintenance_task
                    except asyncio.CancelledError:
                        pass
                    maintenance_task = None
        await self._cleanup()

    async def stop(self) -> None:
        self._running = False

    async def _cleanup(self) -> None:
        await self._queue.close()
        await self._database.close()
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def _process_loop(self) -> None:
        batch: list[NarrativeJob] = []
        batch_start = asyncio.get_event_loop().time()
        async for job in self._queue.consume(
            count=self._batch_size,
            block_ms=5000,
        ):
            if not self._running:
                break
            batch.append(job)
            if (
                len(batch) >= self._batch_size
                or (asyncio.get_event_loop().time() - batch_start) > self._config.worker_batch_timeout
            ):
                await self._process_batch(batch)
                batch = []
                batch_start = asyncio.get_event_loop().time()
                try:
                    pending = await self._queue.get_pending_count()
                    self._metrics.set_narrative_queue_depth(pending)
                except Exception:
                    pass
        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[NarrativeJob]) -> None:
        processed = 0
        skipped = 0
        errors = 0
        for job in batch:
            try:
                doc = await self._doc_repo.get_by_id(job.document_id)
                if doc is None or doc.embedding is None:
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue
                run = await self.process_document_for_theme(
                    document=doc,
                    theme_id=job.theme_id,
                    theme_similarity=job.theme_similarity,
                )
                if run is None:
                    skipped += 1
                else:
                    processed += 1
                await self._queue.ack(job.message_id)
            except Exception as exc:
                logger.exception("Narrative job failed for %s", job.document_id)
                errors += 1
                await self._queue.nack(job.message_id, str(exc))
        self._metrics.record_narrative_batch(processed, skipped, errors)

    async def process_document_for_theme(
        self,
        document: Any,
        theme_id: str,
        theme_similarity: float,
    ) -> NarrativeRun | None:
        embedding = np.array(document.embedding, dtype=np.float32)
        async with self._database.transaction() as conn:
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", theme_id)
            rows = await conn.fetch(
                """
                SELECT * FROM narrative_runs
                WHERE theme_id = $1
                  AND status IN ('active', 'cooling')
                ORDER BY last_document_at DESC
                LIMIT $2
                """,
                theme_id,
                self._config.candidate_limit,
            )
            candidates = [_row_to_run(row) for row in rows]
            target = self._select_run(candidates, embedding)
            if target is None:
                run = await self._create_run(conn, theme_id, embedding, document)
                similarity = 1.0
            else:
                run = await self._attach_to_run(conn, target, embedding, document)
                similarity = float(self._cosine_similarity(target.centroid, embedding))
            await conn.execute(
                """
                INSERT INTO narrative_run_documents (run_id, document_id, theme_id, similarity, assigned_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (run_id, document_id) DO NOTHING
                """,
                run.run_id,
                document.id,
                theme_id,
                similarity,
                document.timestamp,
            )
        buckets = await self._narrative_repo.get_recent_buckets(run.run_id)
        refreshed = await self._narrative_repo.get_by_id(run.run_id)
        if refreshed is None:
            return None
        await self._evaluate_and_publish_alerts(refreshed, buckets, document)
        return refreshed

    def _select_run(
        self,
        candidates: list[NarrativeRun],
        embedding: np.ndarray,
    ) -> NarrativeRun | None:
        best_run: NarrativeRun | None = None
        best_similarity = 0.0
        now = datetime.now(timezone.utc)
        for candidate in candidates:
            if candidate.status == "closed":
                continue
            similarity = self._cosine_similarity(candidate.centroid, embedding)
            if candidate.status == "cooling":
                if now - candidate.last_document_at > timedelta(hours=self._config.close_hours):
                    continue
            if similarity >= self._config.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_run = candidate
        return best_run

    async def _create_run(
        self,
        conn: Any,
        theme_id: str,
        embedding: np.ndarray,
        document: Any,
    ) -> NarrativeRun:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        platform_first_seen = json.dumps({str(document.platform): document.timestamp.isoformat()})
        ticker_counts = json.dumps({ticker: 1 for ticker in document.tickers_mentioned})
        sentiment_value = self._sentiment_value(document.sentiment)
        authority = float(document.authority_score or 0.0)
        label = self._label_from_doc(document)
        bucket_start = self._bucket_start(document.timestamp, self._config.bucket_minutes)
        await conn.execute(
            """
            INSERT INTO narrative_runs (
                run_id, theme_id, status, centroid, label, started_at, last_document_at,
                doc_count, platform_first_seen, ticker_counts, avg_sentiment, avg_authority,
                platform_count, current_rate_per_hour, current_acceleration, conviction_score,
                metadata
            ) VALUES (
                $1, $2, 'active', $3, $4, $5, $5,
                1, $6::jsonb, $7::jsonb, $8, $9,
                1, 12.0, 12.0, 0.0,
                '{}'::jsonb
            )
            """,
            run_id,
            theme_id,
            self._to_pgvector(embedding),
            label,
            document.timestamp,
            platform_first_seen,
            ticker_counts,
            sentiment_value,
            authority,
        )
        await self._upsert_bucket(conn, run_id, bucket_start, document)
        row = await conn.fetchrow("SELECT * FROM narrative_runs WHERE run_id = $1", run_id)
        return _row_to_run(row)

    async def _attach_to_run(
        self,
        conn: Any,
        run: NarrativeRun,
        embedding: np.ndarray,
        document: Any,
    ) -> NarrativeRun:
        new_doc_count = run.doc_count + 1
        new_centroid = self._ema_centroid_update(run.centroid, embedding, 0.2)
        platform_first_seen = dict(run.platform_first_seen)
        platform_key = str(document.platform)
        if platform_key not in platform_first_seen:
            platform_first_seen[platform_key] = document.timestamp.isoformat()
        ticker_counts = dict(run.ticker_counts)
        for ticker in document.tickers_mentioned:
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        sentiment_value = self._sentiment_value(document.sentiment)
        authority = float(document.authority_score or 0.0)
        avg_sentiment = ((run.avg_sentiment * run.doc_count) + sentiment_value) / new_doc_count
        avg_authority = ((run.avg_authority * run.doc_count) + authority) / new_doc_count
        label = self._label_from_counts(ticker_counts, fallback=run.label)
        bucket_start = self._bucket_start(document.timestamp, self._config.bucket_minutes)
        await self._upsert_bucket(conn, run.run_id, bucket_start, document)
        buckets_rows = await conn.fetch(
            """
            SELECT * FROM narrative_run_buckets
            WHERE run_id = $1
              AND bucket_start >= $2
            ORDER BY bucket_start ASC
            """,
            run.run_id,
            datetime.now(timezone.utc) - timedelta(hours=self._config.surge_baseline_hours),
        )
        buckets = [_row_to_bucket(row) for row in buckets_rows]
        current_rate, current_acceleration = self._compute_rate_metrics(buckets)
        await conn.execute(
            """
            UPDATE narrative_runs
            SET status = 'active',
                centroid = $2,
                label = $3,
                last_document_at = $4,
                doc_count = $5,
                platform_first_seen = $6::jsonb,
                ticker_counts = $7::jsonb,
                avg_sentiment = $8,
                avg_authority = $9,
                platform_count = $10,
                current_rate_per_hour = $11,
                current_acceleration = $12
            WHERE run_id = $1
            """,
            run.run_id,
            self._to_pgvector(new_centroid),
            label,
            document.timestamp,
            new_doc_count,
            json.dumps(platform_first_seen),
            json.dumps(ticker_counts),
            avg_sentiment,
            avg_authority,
            len(platform_first_seen),
            current_rate,
            current_acceleration,
        )
        row = await conn.fetchrow("SELECT * FROM narrative_runs WHERE run_id = $1", run.run_id)
        return _row_to_run(row)

    async def _upsert_bucket(
        self,
        conn: Any,
        run_id: str,
        bucket_start: datetime,
        document: Any,
    ) -> None:
        sentiment_value = self._sentiment_value(document.sentiment)
        sentiment_confidence = self._sentiment_confidence(document.sentiment)
        authority = float(document.authority_score or 0.0)
        weight = max(authority, 0.1)
        high_weight = weight if authority >= self._config.high_authority_threshold else 0.0
        low_weight = weight if authority < self._config.high_authority_threshold else 0.0
        high_doc_count = 1 if authority >= self._config.high_authority_threshold else 0
        low_doc_count = 1 if authority < self._config.high_authority_threshold else 0
        existing_row = await conn.fetchrow(
            """
            SELECT * FROM narrative_run_buckets
            WHERE run_id = $1 AND bucket_start = $2
            """,
            run_id,
            bucket_start,
        )
        platform_counts = {str(document.platform): 1}
        ticker_counts = {ticker: 1 for ticker in document.tickers_mentioned}
        if existing_row:
            existing_bucket = _row_to_bucket(existing_row)
            for key, value in existing_bucket.platform_counts.items():
                platform_counts[key] = platform_counts.get(key, 0) + value
            for key, value in existing_bucket.ticker_counts.items():
                ticker_counts[key] = ticker_counts.get(key, 0) + value

        await conn.execute(
            """
            INSERT INTO narrative_run_buckets (
                run_id, bucket_start, doc_count, platform_counts, ticker_counts,
                sentiment_sum, sentiment_weight, sentiment_confidence_sum,
                sentiment_doc_count, authority_sum,
                high_authority_sentiment_sum, high_authority_weight, high_authority_doc_count,
                low_authority_sentiment_sum, low_authority_weight, low_authority_doc_count
            ) VALUES (
                $1, $2, 1, $3::jsonb, $4::jsonb,
                $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            )
            ON CONFLICT (run_id, bucket_start) DO UPDATE
            SET doc_count = narrative_run_buckets.doc_count + 1,
                platform_counts = EXCLUDED.platform_counts,
                ticker_counts = EXCLUDED.ticker_counts,
                sentiment_sum = narrative_run_buckets.sentiment_sum + EXCLUDED.sentiment_sum,
                sentiment_weight = narrative_run_buckets.sentiment_weight + EXCLUDED.sentiment_weight,
                sentiment_confidence_sum = narrative_run_buckets.sentiment_confidence_sum + EXCLUDED.sentiment_confidence_sum,
                sentiment_doc_count = narrative_run_buckets.sentiment_doc_count + EXCLUDED.sentiment_doc_count,
                authority_sum = narrative_run_buckets.authority_sum + EXCLUDED.authority_sum,
                high_authority_sentiment_sum = narrative_run_buckets.high_authority_sentiment_sum + EXCLUDED.high_authority_sentiment_sum,
                high_authority_weight = narrative_run_buckets.high_authority_weight + EXCLUDED.high_authority_weight,
                high_authority_doc_count = narrative_run_buckets.high_authority_doc_count + EXCLUDED.high_authority_doc_count,
                low_authority_sentiment_sum = narrative_run_buckets.low_authority_sentiment_sum + EXCLUDED.low_authority_sentiment_sum,
                low_authority_weight = narrative_run_buckets.low_authority_weight + EXCLUDED.low_authority_weight,
                low_authority_doc_count = narrative_run_buckets.low_authority_doc_count + EXCLUDED.low_authority_doc_count
            """,
            run_id,
            bucket_start,
            json.dumps(platform_counts),
            json.dumps(ticker_counts),
            sentiment_value * weight,
            weight,
            sentiment_confidence,
            1 if document.sentiment else 0,
            authority,
            sentiment_value * high_weight,
            high_weight,
            high_doc_count,
            sentiment_value * low_weight,
            low_weight,
            low_doc_count,
        )

    async def _evaluate_and_publish_alerts(
        self,
        run: NarrativeRun,
        buckets: list[Any],
        document: Any,
    ) -> None:
        evaluations = evaluate_all_signals(run, buckets, self._config)
        states = await self._narrative_repo.get_signal_states(run.run_id)
        alerts_to_publish: list[Alert] = []
        async with self._database.transaction() as conn:
            for evaluation in evaluations:
                state = states.get(evaluation.trigger_type)
                should_publish = self._should_publish(state, evaluation)
                if should_publish and evaluation.title and evaluation.message:
                    alert = Alert(
                        theme_id=run.theme_id,
                        subject_type="narrative_run",
                        subject_id=run.run_id,
                        trigger_type=evaluation.trigger_type,
                        severity=evaluation.severity or "warning",
                        conviction_score=evaluation.conviction_score,
                        title=evaluation.title,
                        message=evaluation.message,
                        trigger_data={
                            **evaluation.trigger_data,
                            "run_id": run.run_id,
                            "run_label": run.label,
                            "theme_id": run.theme_id,
                            "top_document_ids": [document.id],
                            "support_count": run.doc_count,
                            "platform_timeline": run.platform_first_seen,
                            "top_tickers": list(run.ticker_counts)[:5],
                        },
                    )
                    alerts_to_publish.append(alert)
                    await conn.execute(
                        """
                        UPDATE narrative_runs
                        SET conviction_score = $2,
                            last_signal_at = $3
                        WHERE run_id = $1
                        """,
                        run.run_id,
                        float(evaluation.conviction_score or run.conviction_score),
                        datetime.now(timezone.utc),
                    )
                await self._upsert_signal_state(conn, run.run_id, state, evaluation, should_publish)
        if alerts_to_publish and self._alert_service:
            await self._alert_service.publish_alerts(alerts_to_publish)

    def _should_publish(
        self,
        state: NarrativeSignalState | None,
        evaluation: SignalEvaluation,
    ) -> bool:
        now = datetime.now(timezone.utc)
        if not evaluation.triggered:
            return False
        if state is None:
            return True
        if state.cooldown_until and state.cooldown_until > now:
            return False
        if state.state == "inactive":
            return True
        return evaluation.metric_value > (state.last_score + 0.25)

    async def _upsert_signal_state(
        self,
        conn: Any,
        run_id: str,
        state: NarrativeSignalState | None,
        evaluation: SignalEvaluation,
        published: bool,
    ) -> None:
        now = datetime.now(timezone.utc)
        next_state = "active" if evaluation.triggered else "inactive"
        last_score = evaluation.metric_value if evaluation.triggered else 0.0
        cooldown_until = now + timedelta(minutes=30) if published else (state.cooldown_until if state else None)
        metadata = dict(state.metadata) if state else {}
        if evaluation.triggered and evaluation.trigger_type == "cross_platform_breakout":
            metadata["platform_count"] = evaluation.metric_value
        await conn.execute(
            """
            INSERT INTO narrative_signal_state (
                run_id, trigger_type, state, last_score, last_alert_at,
                last_transition_at, cooldown_until, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            ON CONFLICT (run_id, trigger_type) DO UPDATE
            SET state = EXCLUDED.state,
                last_score = EXCLUDED.last_score,
                last_alert_at = COALESCE(EXCLUDED.last_alert_at, narrative_signal_state.last_alert_at),
                last_transition_at = EXCLUDED.last_transition_at,
                cooldown_until = EXCLUDED.cooldown_until,
                metadata = EXCLUDED.metadata
            """,
            run_id,
            evaluation.trigger_type,
            next_state if (evaluation.triggered or evaluation.metric_value <= evaluation.deactivate_below) else (state.state if state else "inactive"),
            last_score,
            now if published else (state.last_alert_at if state else None),
            now,
            cooldown_until,
            json.dumps(metadata),
        )

    async def _maintenance_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._config.maintenance_interval_seconds)
                await self._run_maintenance()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Narrative maintenance failed")

    async def _run_maintenance(self) -> None:
        now = datetime.now(timezone.utc)
        cooling_cutoff = now - timedelta(hours=self._config.cooling_hours)
        close_cutoff = now - timedelta(hours=self._config.close_hours)
        await self._database.execute(
            """
            UPDATE narrative_runs
            SET status = 'cooling'
            WHERE status = 'active'
              AND last_document_at < $1
            """,
            cooling_cutoff,
        )
        await self._database.execute(
            """
            UPDATE narrative_runs
            SET status = 'closed', closed_at = COALESCE(closed_at, $2)
            WHERE status IN ('active', 'cooling')
              AND last_document_at < $1
            """,
            close_cutoff,
            now,
        )
        # Same-theme merge decisions stay in maintenance only. For now this is
        # conservative: only merge if both runs are still open and very similar.
        rows = await self._database.fetch(
            """
            SELECT * FROM narrative_runs
            WHERE status IN ('active', 'cooling')
            ORDER BY theme_id, last_document_at DESC
            """
        )
        by_theme: dict[str, list[NarrativeRun]] = {}
        for row in rows:
            run = _row_to_run(row)
            by_theme.setdefault(run.theme_id, []).append(run)
        for theme_runs in by_theme.values():
            for index, left in enumerate(theme_runs):
                for right in theme_runs[index + 1:]:
                    if self._cosine_similarity(left.centroid, right.centroid) < self._config.merge_threshold:
                        continue
                    await self._database.execute(
                        """
                        UPDATE narrative_runs
                        SET status = 'closed',
                            closed_at = COALESCE(closed_at, $2),
                            metadata = metadata || $3::jsonb
                        WHERE run_id = $1
                        """,
                        right.run_id,
                        now,
                        json.dumps({"merged_into": left.run_id}),
                    )

    @staticmethod
    def _bucket_start(ts: datetime, bucket_minutes: int = 5) -> datetime:
        ts = ts.astimezone(timezone.utc)
        minute = (ts.minute // bucket_minutes) * bucket_minutes
        return ts.replace(minute=minute, second=0, microsecond=0)

    @staticmethod
    def _to_pgvector(vec: np.ndarray) -> str:
        return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    @staticmethod
    def _ema_centroid_update(
        centroid: np.ndarray,
        embedding: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        return ((1.0 - learning_rate) * centroid + learning_rate * embedding).astype(np.float32)

    def _compute_rate_metrics(self, buckets: list[Any]) -> tuple[float, float]:
        rates = [
            float(bucket.doc_count) * (60.0 / self._config.bucket_minutes)
            for bucket in buckets[-6:]
        ]
        if not rates:
            return 0.0, 0.0
        current = rates[-1]
        acceleration = current - rates[-2] if len(rates) > 1 else current
        return current, acceleration

    @staticmethod
    def _sentiment_value(sentiment: dict[str, Any] | None) -> float:
        label = (sentiment or {}).get("label")
        return {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0,
        }.get(label, 0.0)

    @staticmethod
    def _sentiment_confidence(sentiment: dict[str, Any] | None) -> float:
        if not sentiment:
            return 0.0
        return float(sentiment.get("confidence") or 0.0)

    @staticmethod
    def _label_from_doc(document: Any) -> str:
        if document.tickers_mentioned:
            return " / ".join(document.tickers_mentioned[:3])
        if document.title:
            return document.title[:80]
        return document.id

    @staticmethod
    def _label_from_counts(ticker_counts: dict[str, int], fallback: str) -> str:
        if not ticker_counts:
            return fallback
        ordered = sorted(ticker_counts.items(), key=lambda item: (-item[1], item[0]))
        return " / ".join(ticker for ticker, _count in ordered[:3])
